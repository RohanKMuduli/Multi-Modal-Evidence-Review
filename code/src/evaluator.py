"""
Evaluation module. Computes accuracy metrics, compiles confusion matrices,
and generates the operational analysis and evaluation report.
"""

import json
import logging
import os
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger("damage_claim_system.evaluator")


class Evaluator:
    """
    Computes performance metrics and generates report documentation.
    """

    def __init__(
        self,
        sample_claims_csv: str,
        predictions_csv: str,
        evaluation_dir: str,
    ):
        self.sample_claims_csv = sample_claims_csv
        self.predictions_csv = predictions_csv
        self.evaluation_dir = evaluation_dir

        # Ensure evaluation directory exists
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def run_evaluation(self, model_calls_count: int, images_processed_count: int, runtime_seconds: float) -> None:
        """
        Loads ground truths and predictions, computes metrics, and writes files.
        """
        logger.info("Starting system evaluation...")

        if not os.path.exists(self.sample_claims_csv):
            logger.error(f"Sample claims file not found: {self.sample_claims_csv}")
            return
        if not os.path.exists(self.predictions_csv):
            logger.error(f"Predictions file not found: {self.predictions_csv}")
            return

        # Load dataframes
        df_gt = pd.read_csv(self.sample_claims_csv)
        df_pred = pd.read_csv(self.predictions_csv)

        # Normalize IDs for alignment
        df_gt["user_id"] = df_gt["user_id"].astype(str).str.strip().str.lower()
        df_pred["user_id"] = df_pred["user_id"].astype(str).str.strip().str.lower()

        # Merge datasets on user_id
        merged = pd.merge(
            df_gt,
            df_pred,
            on="user_id",
            suffixes=("_gt", "_pred"),
        )

        if len(merged) == 0:
            logger.error("No overlapping user_id found between ground truth and predictions. Evaluation aborted.")
            return

        logger.info(f"Aligned {len(merged)} claims for evaluation.")

        # Compute Accuracies
        claim_status_acc = (merged["claim_status_gt"] == merged["claim_status_pred"]).mean()
        issue_type_acc = (merged["issue_type_gt"] == merged["issue_type_pred"]).mean()
        object_part_acc = (merged["object_part_gt"] == merged["object_part_pred"]).mean()
        severity_acc = (merged["severity_gt"] == merged["severity_pred"]).mean()
        # Cast boolean flags cleanly
        merged["evidence_standard_met_gt"] = merged["evidence_standard_met_gt"].astype(str).str.strip().str.lower() == "true"
        merged["evidence_standard_met_pred"] = merged["evidence_standard_met_pred"].astype(str).str.strip().str.lower() == "true"
        evidence_standard_acc = (merged["evidence_standard_met_gt"] == merged["evidence_standard_met_pred"]).mean()


        metrics = {
            "total_samples": len(merged),
            "claim_status_accuracy": float(claim_status_acc),
            "issue_type_accuracy": float(issue_type_acc),
            "object_part_accuracy": float(object_part_acc),
            "severity_accuracy": float(severity_acc),
            "evidence_standard_accuracy": float(evidence_standard_acc),
        }

        # Save metrics.json
        metrics_path = os.path.join(self.evaluation_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {metrics_path}")

        # Compute Confusion Matrix for claim_status
        categories = ["supported", "contradicted", "not_enough_information"]
        cm_data = {cat: {c: 0 for c in categories} for cat in categories}

        for _, row in merged.iterrows():
            gt = row["claim_status_gt"]
            pred = row["claim_status_pred"]
            if gt in categories and pred in categories:
                cm_data[gt][pred] += 1

        cm_rows = []
        for gt in categories:
            cm_rows.append({
                "Actual/Predicted": gt,
                "supported": cm_data[gt]["supported"],
                "contradicted": cm_data[gt]["contradicted"],
                "not_enough_information": cm_data[gt]["not_enough_information"]
            })
        
        df_cm = pd.DataFrame(cm_rows)
        cm_path = os.path.join(self.evaluation_dir, "confusion_matrix.csv")
        df_cm.to_csv(cm_path, index=False)
        logger.info(f"Saved confusion matrix to {cm_path}")

        # Save Sample Predictions Side-by-Side Comparison
        df_samples = merged[[
            "user_id",
            "claim_object_gt",
            "claim_status_gt",
            "claim_status_pred",
            "issue_type_gt",
            "issue_type_pred",
            "object_part_gt",
            "object_part_pred",
            "severity_gt",
            "severity_pred",
            "evidence_standard_met_gt",
            "evidence_standard_met_pred",
        ]].copy()
        
        # Rename columns for clarity
        df_samples.columns = [
            "user_id",
            "claim_object",
            "ground_truth_status",
            "predicted_status",
            "ground_truth_issue",
            "predicted_issue",
            "ground_truth_part",
            "predicted_part",
            "ground_truth_severity",
            "predicted_severity",
            "ground_truth_evidence",
            "predicted_evidence",
        ]
        
        samples_path = os.path.join(self.evaluation_dir, "sample_predictions.csv")
        df_samples.to_csv(samples_path, index=False)
        logger.info(f"Saved sample predictions to {samples_path}")

        # Generate Evaluation Report (evaluation_report.md)
        self._write_report(
            metrics=metrics,
            model_calls=model_calls_count,
            images_processed=images_processed_count,
            runtime_seconds=runtime_seconds,
            merged_df=merged,
        )

    def _write_report(
        self,
        metrics: Dict[str, Any],
        model_calls: int,
        images_processed: int,
        runtime_seconds: float,
        merged_df: pd.DataFrame,
    ) -> None:
        """
        Compiles the evaluation_report.md artifact with cost and token estimation.
        """
        # Token and cost calculations
        # Assumptions for Gemini 2.5 Flash (Standard rate: $0.075/1M Input tokens, $0.30/1M Output tokens, Image ~258 tokens)
        # Assumptions for GPT-4o (Standard rate: $2.50/1M Input tokens, $10.00/1M Output tokens, Image ~765 tokens)
        
        # Let's say per claim parser run: 300 input tokens, 100 output tokens
        # Per image analyzer run: 850 input tokens (including vision tiles), 150 output tokens
        parser_calls = len(merged_df)
        image_calls = images_processed

        gemini_input_tokens = (parser_calls * 300) + (image_calls * (300 + 258))
        gemini_output_tokens = (parser_calls * 100) + (image_calls * 150)
        gemini_cost = (gemini_input_tokens * 0.075 / 1e6) + (gemini_output_tokens * 0.30 / 1e6)

        gpt_input_tokens = (parser_calls * 300) + (image_calls * (300 + 765))
        gpt_output_tokens = (parser_calls * 100) + (image_calls * 150)
        gpt_cost = (gpt_input_tokens * 2.50 / 1e6) + (gpt_output_tokens * 10.00 / 1e6)

        report_path = os.path.join(self.evaluation_dir, "evaluation_report.md")

        content = f"""# System Evaluation Report

This report summarizes the metric evaluation and operational analysis of the Multi-Modal Damage Claim Verification System.

## Evaluation Summary

> [!NOTE]
> Ground truth data compiled from [{os.path.basename(self.sample_claims_csv)}](file:///{self.sample_claims_csv.replace('\\', '/')}).
> Predictions loaded from [{os.path.basename(self.predictions_csv)}](file:///{self.predictions_csv.replace('\\', '/')}).

### Core Model Accuracies

| Metric | Score | Matches / Total |
| :--- | :---: | :---: |
| **Claim Status Accuracy** | {metrics['claim_status_accuracy']:.2%} | {int(metrics['claim_status_accuracy'] * metrics['total_samples'])} / {metrics['total_samples']} |
| **Issue Type Accuracy** | {metrics['issue_type_accuracy']:.2%} | {int(metrics['issue_type_accuracy'] * metrics['total_samples'])} / {metrics['total_samples']} |
| **Object Part Accuracy** | {metrics['object_part_accuracy']:.2%} | {int(metrics['object_part_accuracy'] * metrics['total_samples'])} / {metrics['total_samples']} |
| **Severity Accuracy** | {metrics['severity_accuracy']:.2%} | {int(metrics['severity_accuracy'] * metrics['total_samples'])} / {metrics['total_samples']} |
| **Evidence Standard Accuracy** | {metrics['evidence_standard_accuracy']:.2%} | {int(metrics['evidence_standard_accuracy'] * metrics['total_samples'])} / {metrics['total_samples']} |

---

## Operational Analysis

This section analyzes API performance, costs, rates, and latency limits for production deployment.

### System Usage Metrics
- **Total Claims Processed**: {metrics['total_samples']}
- **Total Images Analyzed**: {image_calls}
- **API Model Calls Made**: {model_calls}
- **Total Local Execution Time**: {runtime_seconds:.3f} seconds (Average {runtime_seconds/max(1, metrics['total_samples']):.3f}s per claim)

### Token and Cost Estimation

Below is an operational budget estimate comparing **Gemini 2.5 Flash** and **GPT-4o** APIs for running the evaluation batch.

| Model Provider | Est. Input Tokens | Est. Output Tokens | Estimated API Cost | Cost per Claim |
| :--- | :---: | :---: | :---: | :---: |
| **Google Gemini 2.5 Flash** | {gemini_input_tokens:,} | {gemini_output_tokens:,} | **${gemini_cost:.6f}** | ${gemini_cost/max(1, metrics['total_samples']):.6f} |
| **OpenAI GPT-4o** | {gpt_input_tokens:,} | {gpt_output_tokens:,} | **${gpt_cost:.6f}** | ${gpt_cost/max(1, metrics['total_samples']):.6f} |
| **Mock Mode** | 0 | 0 | **$0.000000** | $0.000000 |

*Token calculation assumptions:*
- Text-only parsing: 300 input tokens, 100 output tokens.
- Multimodal image processing: 300 text input + image resolution tokens (GPT-4o high-res tiles: 765 tokens, Gemini: 258 tokens per image), 150 output tokens.

---

### Production Deployment & Scalability Guidelines

> [!TIP]
> **RPM / TPM Rate Limit Considerations**
> Most providers impose limits (e.g. 15 RPM for Gemini free tier, 500 RPM for Tier 1 GPT-4o).
> With 1 text call and $N$ image calls per claim, a batch of 100 claims requires up to 300 calls.
> To prevent rate-limit errors, implement exponential backoff retry wrappers.

#### 1. Batching Strategy
- **Asynchronous Execution**: Submit claims concurrently using `asyncio` or Thread Pools rather than blocking loops.
- **Provider Batch APIs**: For non-realtime claims processing, utilize OpenAI Batch API to cut costs by 50% and bypass rate limits (24-hour SLA).

#### 2. Retry Strategy
- Use decorators like `tenacity` or custom wrappers with exponential backoff and jitter.
- Max retries: 3-5 times.
- Catch specific status codes (429 Rate Limit, 503 Overloaded) rather than catching all exceptions.

#### 3. Caching Strategy
- **Context Caching**: Gemini supports Context Caching for large inputs (e.g., standard instructions or video frames). Use it if instructions exceed 32k tokens.
- **Image Hash Caching**: Calculate SHA-256 hashes of submitted images. If a user submits duplicate images within multiple claims, return cached analyzer results to avoid redundant model fees.
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved evaluation report markdown to {report_path}")
