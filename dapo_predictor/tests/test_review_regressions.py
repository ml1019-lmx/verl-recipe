import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "dapo_predictor" / "predictor_dapo_trainer.py"
WORKER = REPO_ROOT / "dapo_predictor" / "predictor_worker.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in {path}")


class PredictorReviewRegressionTests(unittest.TestCase):
    def test_predictor_fit_delegates_to_base_fit_through_runtime_hooks(self):
        fit_source = _function_source(TRAINER, "fit")

        self.assertIn("with self._predictor_runtime_hooks():", fit_source)
        self.assertIn("return super().fit()", fit_source)
        self.assertNotIn("for epoch in", fit_source)
        self.assertNotIn("for batch_dict in", fit_source)
        self.assertNotIn("Tracking(", fit_source)

    def test_predictor_runtime_hooks_wrap_and_restore_base_call_sites(self):
        hook_source = _function_source(TRAINER, "_predictor_runtime_hooks")

        self.assertIn("orig_generate_sequences", hook_source)
        self.assertIn("orig_balance_batch", hook_source)
        self.assertIn("orig_update_actor", hook_source)
        self.assertIn("self.async_rollout_manager.generate_sequences = wrapped_generate_sequences", hook_source)
        self.assertIn("self._balance_batch = wrapped_balance_batch", hook_source)
        self.assertIn("self._update_actor = wrapped_update_actor", hook_source)
        self.assertIn("finally:", hook_source)
        self.assertIn("self.async_rollout_manager.generate_sequences = orig_generate_sequences", hook_source)
        self.assertIn("self._balance_batch = orig_balance_batch", hook_source)
        self.assertIn("self._update_actor = orig_update_actor", hook_source)

    def test_predictor_update_returns_metrics_for_actor_wrapper(self):
        update_source = _function_source(TRAINER, "_maybe_update_predictor")

        self.assertIn("return reduce_metrics", update_source)
        self.assertNotIn("metrics.update", update_source)

    def test_predictor_worker_training_guards_small_response_length_and_uses_stable_shuffle(self):
        update_source = _function_source(WORKER, "update_predictor")

        self.assertIn("max(1,", update_source)
        self.assertIn("torch.Generator()", update_source)
        self.assertIn("generator=", update_source)
        self.assertIn("listmle_generator", update_source)
        self.assertIn('cfg.get("epochs", 10)', update_source)

    def test_predictor_worker_avoids_global_seed_and_debug_prints(self):
        source = _source(WORKER)
        tree = ast.parse(source)

        manual_seed_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "manual_seed"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        ]
        print_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        ]

        self.assertEqual(manual_seed_calls, [])
        self.assertEqual(print_calls, [])

    def test_kendall_tau_is_optional_when_scipy_is_missing(self):
        update_source = _function_source(WORKER, "update_predictor")

        self.assertIn("try:", update_source)
        self.assertIn("ImportError", update_source)
        self.assertIn("kendalltau", update_source)


if __name__ == "__main__":
    unittest.main()
