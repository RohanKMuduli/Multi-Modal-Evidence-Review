"""
Risk assessment module. Analyzes customer claim history to identify risk signals.
"""

import logging
import os
from typing import Dict, List, Optional
import pandas as pd

from src.models import RiskAssessment
from src.utils import normalize_string

logger = logging.getLogger("damage_claim_system.risk_assessor")


class RiskAssessor:
    """
    Evaluates claims history and highlights flags for fraud detection or manual oversight.
    """

    def __init__(self, history_csv_path: str):
        self.history_path = history_csv_path
        self.user_records: Dict[str, dict] = {}
        self._load_history()

    def _load_history(self) -> None:
        """
        Loads the history database.
        """
        if not os.path.exists(self.history_path):
            logger.error(f"User history CSV file not found: {self.history_path}")
            return

        try:
            df = pd.read_csv(self.history_path)
            for _, row in df.iterrows():
                uid = normalize_string(str(row["user_id"]))
                self.user_records[uid] = {
                    "past_claim_count": int(row["past_claim_count"]),
                    "accept_claim": int(row["accept_claim"]),
                    "manual_review_claim": int(row["manual_review_claim"]),
                    "rejected_claim": int(row["rejected_claim"]),
                    "last_90_days_claim_count": int(row["last_90_days_claim_count"]),
                    "history_flags": str(row["history_flags"]),
                    "history_summary": str(row["history_summary"]),
                }
            logger.info(f"Loaded {len(self.user_records)} user history profiles.")
        except Exception as e:
            logger.error(f"Failed to load user history CSV: {e}")

    def assess_user_risk(self, user_id: str) -> RiskAssessment:
        """
        Assess risk for a user based on historical claim statistics.
        Critical Rule: History adds context only and must never override visual evidence.
        """
        uid_key = normalize_string(user_id)
        record = self.user_records.get(uid_key)

        if not record:
            logger.info(f"No history records found for user '{user_id}'. Registering as new user.")
            return RiskAssessment(
                user_history_risk=False,
                manual_review_required=False,
                history_summary="New user, no prior claim records found.",
                risk_flags=["none"],
            )

        past_claims = record["past_claim_count"]
        accepted = record["accept_claim"]
        manual_reviews = record["manual_review_claim"]
        rejected = record["rejected_claim"]
        last_90_days = record["last_90_days_claim_count"]
        history_summary = record["history_summary"]

        # Parse flags
        flags_str = record["history_flags"]
        raw_flags = [f.strip() for f in flags_str.split(";") if f.strip() and f.strip().lower() != "none"]

        # Heuristic Risk Flags
        user_history_risk = "user_history_risk" in raw_flags
        manual_review_required = "manual_review_required" in raw_flags
        derived_flags = list(raw_flags)

        # Flag 1: High claim frequency recently
        if last_90_days >= 3:
            user_history_risk = True
            if "user_history_risk" not in derived_flags:
                derived_flags.append("user_history_risk")

        # Flag 2: High rejection rate
        rejection_rate = 0.0
        if past_claims > 0:
            rejection_rate = rejected / past_claims
        if past_claims >= 3 and rejection_rate >= 0.40:
            user_history_risk = True
            manual_review_required = True
            if "user_history_risk" not in derived_flags:
                derived_flags.append("user_history_risk")
            if "manual_review_required" not in derived_flags:
                derived_flags.append("manual_review_required")

        logger.info(
            f"Risk assessment for {user_id}: risk={user_history_risk}, manual_review={manual_review_required}, flags={derived_flags}"
        )

        return RiskAssessment(
            user_history_risk=user_history_risk,
            manual_review_required=manual_review_required,
            history_summary=history_summary,
            risk_flags=derived_flags if derived_flags else ["none"],
        )
