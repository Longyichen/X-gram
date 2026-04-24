from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .callback import Callback
from .evaluator_callback import DownstreamEvaluator, DownstreamEvaluatorCallbackConfig, EvaluatorCallback

log = logging.getLogger(__name__)


class FilteredDownstreamEvaluator(DownstreamEvaluator):
    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        metric_type_to_value = self.metric.compute()

        print(f"\n{'='*80}")
        print(f"Task: {self.label}")
        print(f"Available metrics: {list(metric_type_to_value.keys())}")
        print(f"{'='*80}\n")

        outputs: Dict[str, torch.Tensor] = {}

        acc_metric = None
        if "acc_v2" in metric_type_to_value:
            acc_metric = "acc_v2"
        elif "acc_v1" in metric_type_to_value:
            acc_metric = "acc_v1"

        len_norm_metric = None
        if "acc_len_norm_v2" in metric_type_to_value:
            len_norm_metric = "acc_len_norm_v2"
        elif "len_norm_v2" in metric_type_to_value:
            len_norm_metric = "len_norm_v2"
        elif "acc_len_norm_v1" in metric_type_to_value:
            len_norm_metric = "acc_len_norm_v1"
        elif "len_norm_v1" in metric_type_to_value:
            len_norm_metric = "len_norm_v1"

        if acc_metric or len_norm_metric:
            if acc_metric:
                value = metric_type_to_value[acc_metric]
                key = f"{self.label} ({self.metric_type_to_label[acc_metric]})"
                outputs[key] = value

            if len_norm_metric:
                value = metric_type_to_value[len_norm_metric]
                key = f"{self.label} ({self.metric_type_to_label[len_norm_metric]})"
                outputs[key] = value

            return outputs

        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label.get(metric_type, metric_type)})"
            outputs[key] = value
        return outputs


class AveragingEvaluatorCallback(EvaluatorCallback):
    """Evaluator callback that adds MMLU and global average metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_metrics = {}

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log raw metrics and append aggregate benchmark summaries."""
        super().log_metrics(step, metrics)

        if not metrics:
            return

        mmlu_tasks = [
            "mmlu_stem_mc_5shot",
            "mmlu_humanities_mc_5shot",
            "mmlu_social_sciences_mc_5shot",
            "mmlu_other_mc_5shot",
        ]

        def _to_float(val: Any) -> Optional[float]:
            if isinstance(val, torch.Tensor):
                return val.item()
            if isinstance(val, (int, float)):
                return float(val)
            return None

        def _parse_downstream_key(key: str) -> Optional[tuple]:
            prefixes = ("eval/downstream/", "downstream/")
            matched_prefix = next((p for p in prefixes if key.startswith(p)), None)
            if matched_prefix is None:
                return None
            stripped = key[len(matched_prefix):]
            if " (" in stripped and stripped.endswith(")"):
                task_name, _, label = stripped.partition(" (")
                label = label[:-1]
            else:
                task_name = stripped
                label = ""
            label_lower = label.lower()
            metric_type = None
            if "len_norm_v2" in label_lower or "length-normalized accuracy v2" in label_lower:
                metric_type = "len_norm_v2"
            elif "acc_v2" in label_lower or "accuracy v2" in label_lower:
                metric_type = "acc_v2"
            return (task_name, metric_type)

        def _select_primary_metric(entries: Dict[str, tuple]) -> Optional[tuple]:
            for alias in ("len_norm_v2", "acc_v2"):
                if alias in entries:
                    return entries[alias]
            return None

        downstream_metrics: Dict[str, Dict[str, tuple]] = {}
        for key, value in list(metrics.items()):
            parsed = _parse_downstream_key(key)
            if parsed is None:
                continue
            task_name, metric_type = parsed
            if task_name is None or metric_type is None:
                continue
            numeric_value = _to_float(value)
            if numeric_value is None:
                continue
            downstream_metrics.setdefault(task_name, {})[metric_type] = (numeric_value, key)

        mmlu_values: List[float] = []
        mmlu_details: List[tuple] = []
        for task in mmlu_tasks:
            task_entries = downstream_metrics.get(task)
            if not task_entries:
                continue
            metric_choice = None
            for alias in ("len_norm_v2", "acc_v2"):
                if task_entries.get(alias):
                    metric_choice = task_entries[alias]
                    break
            if metric_choice is None:
                continue
            metric_value, metric_key = metric_choice
            mmlu_values.append(metric_value)
            mmlu_details.append((task, metric_key, metric_value))

        mmlu_avg = None
        if len(mmlu_values) == len(mmlu_tasks):
            mmlu_avg = sum(mmlu_values) / len(mmlu_values)
            metrics["downstream/mmlu_average"] = mmlu_avg
            metrics["eval/downstream/mmlu_average"] = mmlu_avg

            log.info("\n%s", "=" * 80)
            log.info(
                "MMLU Average: %.4f (based on %d tasks)",
                mmlu_avg,
                len(mmlu_values),
            )
            log.info("Included metrics:")
            for task_name, metric_key, metric_value in mmlu_details:
                log.info("  - %s: %.4f", metric_key, metric_value)
            log.info("%s\n", "=" * 80)

        all_task_values: List[float] = []
        all_task_details: List[tuple] = []
        counted_tasks = set()

        if mmlu_avg is not None:
            all_task_values.append(mmlu_avg)
            all_task_details.append(("downstream/mmlu_average", mmlu_avg))
            counted_tasks.update(mmlu_tasks)

        for task_name, task_entries in downstream_metrics.items():
            if task_name in counted_tasks:
                continue
            selected = _select_primary_metric(task_entries)
            if selected is None:
                continue
            metric_value, metric_key = selected
            all_task_values.append(metric_value)
            all_task_details.append((metric_key, metric_value))
            counted_tasks.add(task_name)

        if all_task_values:
            global_avg = sum(all_task_values) / len(all_task_values)
            metrics["downstream/global_average"] = global_avg
            metrics["eval/downstream/global_average"] = global_avg

            log.info("\n%s", "=" * 80)
            log.info(
                "Global Average: %.4f (across %d tasks)",
                global_avg,
                len(all_task_values),
            )
            log.info("Included metrics:")
            for metric_key, metric_value in all_task_details:
                log.info("  - %s: %.4f", metric_key, metric_value)
            log.info("%s\n", "=" * 80)

    def post_train(self):
        super().post_train()
        if hasattr(self, "trainer") and self.trainer is not None:
            self.trainer._log_metrics()


@dataclass
class FilteredDownstreamEvaluatorCallbackConfig(DownstreamEvaluatorCallbackConfig):
    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_core.exceptions import OLMoConfigurationError
        from olmo_core.train.trainer import Trainer

        try:
            from olmo_eval import HFTokenizer
        except ImportError as exc:
            raise OLMoConfigurationError(
                "Downstream evaluation is enabled, but the optional 'olmo_eval' package is not installed. "
                "Install packages/olmo_in_loop_evals (or its published equivalent), or disable "
                "evaluation.downstream.enabled in your config."
            ) from exc

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError("Tokenizer 'identifier' required to build a concrete tokenizer")

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List["Evaluator"] = []
        for task in self.tasks:
            evaluators.append(
                FilteredDownstreamEvaluator(
                    name="downstream",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        return AveragingEvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
