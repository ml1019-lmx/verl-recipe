"""Recipe-side DAPO trainer with predictor-driven rollout reordering."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

from verl import DataProto
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .predictor_utils import snake_sort_indices


class PredictorRayDAPOTrainer(RayDAPOTrainer):
    """DAPO trainer that only injects the predictor-specific reorder steps.

    Most heavy lifting is still reused from the current `RayDAPOTrainer` / `RayPPOTrainer` stack:
    reward computation, KL/ref/value computation, actor/critic updates, checkpointing, metrics,
    and rollout manager orchestration all stay on the upstream path.
    """

    def _predictor_cfg(self):
        return self.config.trainer.get("predictor_reorder", {})

    def _predictor_enabled(self) -> bool:
        return self._predictor_cfg().get("enable", False)

    def _build_predictor_order(self, gen_batch: DataProto) -> torch.Tensor:
        """Compute predictor scores and build snake-sort reorder indices."""
        predictor_scores = self.actor_rollout_wg.compute_predictor_score(gen_batch)
        gen_batch = gen_batch.union(predictor_scores)
        dp_world_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        return torch.tensor(
            snake_sort_indices(
                gen_batch.batch["predictor_scores"].tolist(),
                n_samples_per_prompt=self.config.actor_rollout_ref.rollout.n,
                dp_world_size=dp_world_size,
            ),
            dtype=torch.long,
        )

    def _apply_predictor_order(self, batch: DataProto, predictor_order: torch.Tensor | None) -> DataProto:
        """Apply predictor-derived reorder indices to a DataProto batch."""
        if predictor_order is not None:
            if batch.batch is not None:
                batch.reorder(predictor_order)
            else:
                indices_np = predictor_order.detach().cpu().numpy()
                batch.non_tensor_batch = {k: v[indices_np] for k, v in batch.non_tensor_batch.items()}
        return batch

    def _repeat_and_tag_uid(self, batch: DataProto) -> DataProto:
        """Tag each row with a UID and repeat the batch n times per prompt."""
        batch.non_tensor_batch["uid"] = np.array([str(i) for i in range(len(batch.batch))], dtype=object)
        return batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    @staticmethod
    def _ensure_gen_batch_has_tensors(gen_batch: DataProto, source_batch: DataProto) -> DataProto:
        if gen_batch.batch is None and source_batch.batch is not None:
            gen_batch.batch = source_batch.batch
        return gen_batch

    def _hydrate_gen_batch_model_inputs(self, gen_batch: DataProto) -> DataProto:
        """Ensure gen_batch has input_ids, attention_mask, and position_ids tensors.

        Tokenizes from raw prompts/messages if the model inputs are missing.
        """
        import uuid

        from tensordict import TensorDict

        from verl.workers.rollout.schemas import (
            AsyncRolloutRequest,
            AsyncRolloutRequestStateEnum,
            TokenizationSanityCheckModeEnum,
        )

        if gen_batch.batch is None:
            gen_batch.batch = {}

        batch_keys = set(gen_batch.batch.keys())
        if {"input_ids", "attention_mask", "position_ids"}.issubset(batch_keys):
            return gen_batch

        if "input_ids" not in batch_keys and "prompts" in batch_keys:
            gen_batch.batch["input_ids"] = gen_batch.batch["prompts"]
            batch_keys.add("input_ids")

        if "input_ids" not in batch_keys:
            seqs = None
            messages = None
            if "raw_prompt_ids" in gen_batch.non_tensor_batch:
                raw_prompt_ids = gen_batch.non_tensor_batch["raw_prompt_ids"]
                seqs = raw_prompt_ids.tolist() if isinstance(raw_prompt_ids, np.ndarray) else raw_prompt_ids
            if "messages" in gen_batch.non_tensor_batch:
                messages = gen_batch.non_tensor_batch["messages"]
            elif "raw_prompt" in gen_batch.non_tensor_batch:
                messages = gen_batch.non_tensor_batch["raw_prompt"]
            if messages is not None and len(messages) == 0:
                messages = None
            if messages is not None:
                multi_modal_batch = gen_batch.non_tensor_batch.get("multi_modal_data", None)
                tool_schema_batch = gen_batch.non_tensor_batch.get("tool_schemas", None)
                input_id_list = []
                attn_mask_list = []
                pos_id_list = []
                max_prompt_len = int(self.config.data.get("max_prompt_length", 32768))
                max_response_len = int(self.config.data.get("max_response_length", 8192))
                max_model_len = int(self.config.actor_rollout_ref.rollout.get("max_model_len") or 32768)
                for i, msg in enumerate(messages):
                    multi_modal_data = {"image": [], "video": []}
                    if multi_modal_batch is not None:
                        mm_val = (
                            multi_modal_batch[i] if isinstance(multi_modal_batch, np.ndarray) else multi_modal_batch
                        )
                        if isinstance(mm_val, dict):
                            multi_modal_data.update(mm_val)
                    tools = None
                    if tool_schema_batch is not None:
                        tool_schema_val = (
                            tool_schema_batch[i] if isinstance(tool_schema_batch, np.ndarray) else tool_schema_batch
                        )
                        if tool_schema_val:
                            tools = [
                                tool.model_dump() if hasattr(tool, "model_dump") else tool for tool in tool_schema_val
                            ]
                    request = AsyncRolloutRequest.model_validate(
                        {
                            "request_id": str(uuid.uuid4()),
                            "state": AsyncRolloutRequestStateEnum.PENDING,
                            "messages": msg,
                            "multi_modal_data": multi_modal_data,
                            "tool_schemas": tools,
                            "reward_scores": {},
                            "max_prompt_len": max_prompt_len,
                            "max_response_len": max_response_len,
                            "max_model_len": max_model_len,
                            "use_inference_chat_template": False,
                            "tokenization_sanity_check_mode": TokenizationSanityCheckModeEnum.DISABLE,
                            "processing_class": self.tokenizer,
                        }
                    )
                    input_ids = request.input_ids.squeeze(0)
                    attention_mask = request.attention_mask.squeeze(0)
                    position_ids = request.position_ids
                    if position_ids.dim() == 2 and position_ids.shape[0] == 1:
                        position_ids = position_ids.squeeze(0)
                    input_id_list.append(input_ids)
                    attn_mask_list.append(attention_mask)
                    pos_id_list.append(position_ids)

                if input_id_list:
                    max_len = max(x.shape[-1] for x in input_id_list)
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    input_ids = torch.full((len(input_id_list), max_len), fill_value=pad_token_id, dtype=torch.long)
                    attention_mask = torch.zeros((len(attn_mask_list), max_len), dtype=torch.long)
                    is_3d_pos = pos_id_list[0].dim() == 2
                    if is_3d_pos:
                        pos_channels = pos_id_list[0].shape[0]
                        position_ids = torch.zeros((len(pos_id_list), pos_channels, max_len), dtype=torch.long)
                    else:
                        position_ids = torch.zeros((len(pos_id_list), max_len), dtype=torch.long)

                    for i, (iid, am, pid) in enumerate(zip(input_id_list, attn_mask_list, pos_id_list, strict=True)):
                        input_ids[i, : iid.shape[-1]] = iid
                        attention_mask[i, : am.shape[-1]] = am
                        if is_3d_pos:
                            position_ids[i, :, : pid.shape[-1]] = pid
                        else:
                            position_ids[i, : pid.shape[-1]] = pid

                    gen_batch.batch = TensorDict(
                        source={
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "position_ids": position_ids,
                        },
                        batch_size=(len(input_id_list),),
                    )
                    batch_keys.update({"input_ids", "attention_mask", "position_ids"})

            if seqs is not None:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                max_len = max((len(s) for s in seqs), default=0)
                input_ids = torch.full((len(seqs), max_len), fill_value=pad_token_id, dtype=torch.long)
                for i, seq in enumerate(seqs):
                    if len(seq) > 0:
                        input_ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
                gen_batch.batch["input_ids"] = input_ids
                batch_keys.add("input_ids")

        if "attention_mask" not in batch_keys and "input_ids" in batch_keys:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            gen_batch.batch["attention_mask"] = (gen_batch.batch["input_ids"] != pad_token_id).long()
            batch_keys.add("attention_mask")

        if "position_ids" not in batch_keys and "attention_mask" in batch_keys:
            gen_batch.batch["position_ids"] = (
                (torch.cumsum(gen_batch.batch["attention_mask"], dim=-1) - 1).clamp_min(0).long()
            )

        return gen_batch

    @staticmethod
    def _build_reverse_idx_from_uid(before_uid: np.ndarray, after_uid: np.ndarray) -> torch.Tensor:
        """Build a reverse index mapping original positions to post-reorder positions.

        Used to restore data order after DP balancing. Tracks duplicate UIDs
        by counting occurrences so that repeated prompts can be matched correctly.
        """
        before_counts = defaultdict(int)
        before_slots: dict[tuple[str, int], int] = {}
        for idx, uid in enumerate(before_uid.tolist()):
            key = (uid, before_counts[uid])
            before_slots[key] = idx
            before_counts[uid] += 1

        after_counts = defaultdict(int)
        orig_pos_of_after = []
        for uid in after_uid.tolist():
            key = (uid, after_counts[uid])
            if key not in before_slots:
                raise ValueError(f"Cannot restore predictor order: missing uid key {key}")
            orig_pos_of_after.append(before_slots[key])
            after_counts[uid] += 1

        reverse_idx = torch.empty(len(orig_pos_of_after), dtype=torch.long)
        for after_pos, orig_pos in enumerate(orig_pos_of_after):
            reverse_idx[orig_pos] = after_pos
        return reverse_idx

    @staticmethod
    def _prepare_predictor_gen_batch(source_batch: DataProto) -> DataProto:
        """Pop a lightweight gen batch containing only predictor-required keys."""
        batch_keys = []
        if source_batch.batch is not None:
            preferred_batch_keys = ["input_ids", "attention_mask", "position_ids", "prompts"]
            source_batch_keys = set(source_batch.batch.keys())
            batch_keys = [k for k in preferred_batch_keys if k in source_batch_keys]
            if not batch_keys:
                batch_keys = list(source_batch_keys)

        preferred_non_tensor_keys = ["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"]
        non_tensor_batch_keys = [k for k in preferred_non_tensor_keys if k in source_batch.non_tensor_batch]

        gen_batch = source_batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)
        return gen_batch

    @staticmethod
    def _extract_restore_keys(batch: DataProto) -> np.ndarray:
        """Extract unique identifier keys from non_tensor_batch for order restoration."""
        if "uid" in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch["uid"], dtype=object)
        if "extra_info" in batch.non_tensor_batch:
            extra_info = batch.non_tensor_batch["extra_info"]
            keys = []
            for item in extra_info:
                if isinstance(item, dict) and "index" in item:
                    keys.append(item["index"])
                else:
                    keys.append(str(item))
            return np.asarray(keys, dtype=object)
        raise ValueError("Cannot restore order: neither `uid` nor `extra_info.index` found in non_tensor_batch")

    def _maybe_update_predictor(self, batch: DataProto, timing_raw):
        with marked_timer("update_predictor", timing_raw, "orange"):
            prompt_length = batch.batch["prompts"].shape[-1]
            prompt_input_ids = batch.batch["prompts"]
            prompt_attention_mask = batch.batch["attention_mask"][:, :prompt_length]
            prompt_position_ids = batch.batch["position_ids"][:, :prompt_length]

            prompt_batch = DataProto.from_dict(
                {
                    "input_ids": prompt_input_ids,
                    "attention_mask": prompt_attention_mask,
                    "position_ids": prompt_position_ids,
                },
                meta_info=batch.meta_info,
            )

            predictor_output = self.actor_rollout_wg.update_predictor(prompt_batch, batch)
        return reduce_metrics(predictor_output.meta_info.get("metrics", {}))

    @staticmethod
    def _invert_reorder_indices(order: torch.Tensor) -> torch.Tensor:
        reverse_order = torch.empty_like(order)
        reverse_order[order] = torch.arange(len(order), dtype=order.dtype, device=order.device)
        return reverse_order

    @contextmanager
    def _predictor_runtime_hooks(self):
        """Temporarily inject predictor logic into the inherited DAPO fit loop."""
        orig_generate_sequences = self.async_rollout_manager.generate_sequences
        orig_balance_batch = self._balance_batch
        orig_update_actor = self._update_actor
        self._predictor_balance_reverse_idx = None

        def wrapped_generate_sequences(gen_batch_output):
            is_baseline = gen_batch_output.meta_info.get("do_sample") is False
            if is_baseline:
                return orig_generate_sequences(gen_batch_output)

            predictor_timing = {}
            with marked_timer("predictor_score", predictor_timing, "purple"):
                with marked_timer("predictor_hydrate", predictor_timing, "purple"):
                    predictor_input_batch = gen_batch_output.select(deepcopy=True)
                    predictor_input_batch = self._hydrate_gen_batch_model_inputs(predictor_input_batch)
                predictor_order = self._build_predictor_order(predictor_input_batch)
                reverse_order = self._invert_reorder_indices(predictor_order)
                self._apply_predictor_order(gen_batch_output, predictor_order)

            output = orig_generate_sequences(gen_batch_output)
            self._apply_predictor_order(output, reverse_order)
            output.meta_info.setdefault("timing", {}).update(predictor_timing)
            return output

        def wrapped_balance_batch(batch, metrics, **kwargs):
            uid_before_balance = batch.non_tensor_batch["uid"].copy()
            result = orig_balance_batch(batch, metrics=metrics, **kwargs)
            self._predictor_balance_reverse_idx = self._build_reverse_idx_from_uid(
                uid_before_balance,
                batch.non_tensor_batch["uid"],
            )
            return result

        def wrapped_update_actor(batch):
            actor_output = orig_update_actor(batch)
            reverse_idx = self._predictor_balance_reverse_idx
            self._predictor_balance_reverse_idx = None
            if reverse_idx is not None:
                batch.reorder(reverse_idx)
            predictor_metrics = self._maybe_update_predictor(batch, {})
            actor_output.meta_info.setdefault("metrics", {}).update(predictor_metrics)
            return actor_output

        self.async_rollout_manager.generate_sequences = wrapped_generate_sequences
        self._balance_batch = wrapped_balance_batch
        self._update_actor = wrapped_update_actor
        try:
            yield
        finally:
            self.async_rollout_manager.generate_sequences = orig_generate_sequences
            self._balance_batch = orig_balance_batch
            self._update_actor = orig_update_actor
            self._predictor_balance_reverse_idx = None

    def fit(self):
        if not self._predictor_enabled():
            return super().fit()
        with self._predictor_runtime_hooks():
            return super().fit()
