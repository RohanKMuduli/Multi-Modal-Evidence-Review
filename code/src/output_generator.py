"""
Output generation module. Formats system results and writes them to a final CSV.
"""

import logging
import os
from typing import Any, Dict, List
import pandas as pd

logger = logging.getLogger("damage_claim_system.output_generator")


class OutputGenerator:
    """
    Consolidates run predictions and saves them to the final output CSV.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []

    def add_record(
        self,
        user_id: str,
        image_paths: str,
        user_claim: str,
        claim_object: str,
        evidence_standard_met: bool,
        evidence_standard_met_reason: str,
        risk_flags: List[str],
        issue_type: str,
        object_part: str,
        claim_status: str,
        claim_status_justification: str,
        supporting_image_ids: List[str],
        valid_image: bool,
        severity: str,
    ) -> None:
        """
        Appends a processed claim result row to the local batch.
        """
        # Convert lists to formatted strings
        risk_flags_str = ";".join(risk_flags) if risk_flags else "none"
        supporting_images_str = ";".join(supporting_image_ids) if supporting_image_ids else "none"


        record = {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": evidence_standard_met_reason,
            "risk_flags": risk_flags_str,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_status_justification,
            "supporting_image_ids": supporting_images_str,
            "valid_image": valid_image,
            "severity": severity,
        }
        self.records.append(record)

    def write_csv(self) -> None:
        """
        Saves all added records to output.csv.
        """
        if not self.records:
            logger.warning("No records to write to CSV.")
            return

        try:
            parent_dir = os.path.dirname(self.output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            df = pd.DataFrame(self.records)
            # Ensure column order matches requirements exactly
            columns_order = [
                "user_id",
                "image_paths",
                "user_claim",
                "claim_object",
                "evidence_standard_met",
                "evidence_standard_met_reason",
                "risk_flags",
                "issue_type",
                "object_part",
                "claim_status",
                "claim_status_justification",
                "supporting_image_ids",
                "valid_image",
                "severity",
            ]
            
            # Select only the required columns, ordering them correctly
            df = df[columns_order]
            df.to_csv(self.output_path, index=False)
            logger.info(f"Successfully generated output CSV at: {self.output_path} (Processed {len(df)} rows)")
        except Exception as e:
            logger.error(f"Failed to generate output CSV: {e}")
            raise e
