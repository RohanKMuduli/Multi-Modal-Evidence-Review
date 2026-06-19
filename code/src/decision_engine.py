"""
Decision engine module. Implements deterministic business rules to evaluate claims.
"""

import logging
from typing import List, Tuple
from src.config import SUPPORTED_SEVERITIES
from src.models import (
    ClaimContext,
    EvidenceRequirementStatus,
    ImageAnalysisResult,
    VerificationDecision,
)
from src.utils import normalize_string

logger = logging.getLogger("damage_claim_system.decision_engine")


class DecisionEngine:
    """
    Applies deterministic claims rules to combine image predictions,
    evidence standards, and claim summaries into a final claim status.
    """

    def evaluate_claim(
        self,
        claim_context: ClaimContext,
        evidence_status: EvidenceRequirementStatus,
        image_analyses: List[ImageAnalysisResult],
    ) -> VerificationDecision:
        """
        Processes claims parameters and returns a VerificationDecision.
        """
        claim_part = normalize_string(claim_context.object_part)
        claim_issue = normalize_string(claim_context.issue_type)

        logger.info(f"Evaluating claim. Claimed part: '{claim_part}', Claimed issue: '{claim_issue}'")

        # 1. Filter valid images
        valid_images = [img for img in image_analyses if img.valid_image]
        any_valid_image = len(valid_images) > 0

        if not any_valid_image:
            justification = (
                "No valid or clear images were submitted. All provided images "
                "were flagged with severe quality issues (e.g. blurry, wrong angle)."
            )
            return VerificationDecision(
                claim_status="not_enough_information",
                claim_status_justification=justification,
                supporting_image_ids=[],
                valid_image=False,
                severity="unknown",
            )

        # 2. Check for matching images (where detected part and issue type match the claim context)
        supporting_images: List[ImageAnalysisResult] = []
        part_visible_images: List[ImageAnalysisResult] = []

        for img in valid_images:
            det_part = normalize_string(img.object_part)
            det_issue = normalize_string(img.issue_type)

            # Check if this image displays the claimed part
            if det_part == claim_part or det_part == "body":
                part_visible_images.append(img)
                # Check if it has visible damage matching the claimed issue
                if img.visible_damage and (det_issue == claim_issue or claim_issue == "unknown"):
                    supporting_images.append(img)

        # 3. Apply Decision Rules

        # Scenario A: Supporting evidence found
        if len(supporting_images) > 0:
            # Check if evidence requirements (like minimum images or parts coverage) are satisfied
            if evidence_status.evidence_standard_met:
                # Find maximum severity among supporting images
                severity = self._aggregate_severity(supporting_images)
                supporting_ids = [img.image_id for img in supporting_images]
                justification = (
                    f"Claim is supported by {len(supporting_images)} image(s) showing "
                    f"visible '{claim_issue}' damage on the '{claim_part}'. "
                    f"Evidence standards are fully met."
                )
                return VerificationDecision(
                    claim_status="supported",
                    claim_status_justification=justification,
                    supporting_image_ids=supporting_ids,
                    valid_image=True,
                    severity=severity,
                )
            else:
                # If supporting images exist but they fail minimum counts/requirements
                justification = (
                    f"Damage matching the claim was detected, but the evidence standards were not fully met. "
                    f"Detail: {evidence_status.reason}"
                )
                return VerificationDecision(
                    claim_status="not_enough_information",
                    claim_status_justification=justification,
                    supporting_image_ids=[img.image_id for img in supporting_images],
                    valid_image=True,
                    severity=self._aggregate_severity(supporting_images),
                )

        # Scenario B: Claimed part is visible, but no matching damage is found
        if len(part_visible_images) > 0:
            # Check if they are clean (no damage at all, or wrong damage)
            # If the part is clearly visible in a clear image and shows no damage: Contradicted
            undamaged_images = [img for img in part_visible_images if not img.visible_damage or img.issue_type == "none"]
            if len(undamaged_images) > 0:
                justification = (
                    f"Contradicted: The claimed part '{claim_part}' is clearly visible in the submitted "
                    f"images, but no damage is present (or the part is clean)."
                )
                return VerificationDecision(
                    claim_status="contradicted",
                    claim_status_justification=justification,
                    supporting_image_ids=[],
                    valid_image=True,
                    severity="none",
                )
            else:
                # Part is visible, damage is visible, but it does NOT match the claim (e.g. scratch instead of dent)
                justification = (
                    f"Not Enough Information: Damage was detected on '{claim_part}', but the detected type "
                    f"does not match the claimed issue of '{claim_issue}'."
                )
                return VerificationDecision(
                    claim_status="not_enough_information",
                    claim_status_justification=justification,
                    supporting_image_ids=[],
                    valid_image=True,
                    severity=self._aggregate_severity(part_visible_images),
                )

        # Scenario C: The claimed part is not visible in any valid image
        justification = (
            f"Not Enough Information: The claimed part '{claim_part}' was not visible or could not be "
            f"identified in any of the valid submitted images. Detections found: "
            f"{[img.object_part for img in valid_images]}."
        )
        return VerificationDecision(
            claim_status="not_enough_information",
            claim_status_justification=justification,
            supporting_image_ids=[],
            valid_image=True,
            severity="unknown",
        )

    def _aggregate_severity(self, images: List[ImageAnalysisResult]) -> str:
        """
        Determines the overall severity by choosing the maximum severity among the analyzed images.
        Hierarchy: high > medium > low > none/unknown.
        """
        severity_ranks = {"high": 3, "medium": 2, "low": 1, "unknown": 0, "none": 0}
        max_rank = -1
        max_severity = "unknown"

        for img in images:
            sev = img.severity.lower()
            rank = severity_ranks.get(sev, 0)
            if rank > max_rank:
                max_rank = rank
                max_severity = sev

        return max_severity
