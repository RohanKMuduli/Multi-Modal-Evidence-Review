"""
Claim parsing module. Parses unstructured claim conversation texts into structured contexts.
"""

import logging
from src.config import SUPPORTED_ISSUE_TYPES, SUPPORTED_PARTS, ISSUE_TYPE_TO_FAMILY
from src.models import BaseModelAdapter, ClaimContext
from src.utils import normalize_string

logger = logging.getLogger("damage_claim_system.claim_parser")


class ClaimParser:
    """
    Parses claim conversation logs into structured parameters.
    Includes deterministic rule-based fallbacks in case model calls fail or fail validation.
    """

    def __init__(self, model_adapter: BaseModelAdapter):
        self.model_adapter = model_adapter

    def parse(self, user_claim: str, claim_object: str) -> ClaimContext:
        """
        Parses a claim string using the configured model adapter,
        with a deterministic fallback on failure.
        """
        claim_object = normalize_string(claim_object)
        logger.info(f"Parsing claim for object type '{claim_object}'...")

        try:
            context = self.model_adapter.parse_claim(user_claim, claim_object)
            # Validate output types are compliant with allowed vocabulary
            if context.issue_type not in SUPPORTED_ISSUE_TYPES:
                logger.warning(
                    f"Parsed issue_type '{context.issue_type}' not in vocabulary. Forcing fallback validation."
                )
                context.issue_type = self._deterministic_fallback(user_claim, claim_object).issue_type

            valid_parts = SUPPORTED_PARTS.get(claim_object, set())
            if context.object_part not in valid_parts:
                parsed_part_text = str(context.object_part).lower().replace("-", "_")
                matched_part = next(
                    (
                        part
                        for part in valid_parts
                        if part != "unknown"
                        and (
                            part in parsed_part_text
                            or part.replace("_", " ") in parsed_part_text
                        )
                    ),
                    None,
                )
                if matched_part:
                    context.object_part = matched_part
                else:
                    logger.warning(
                        f"Parsed object_part '{context.object_part}' not valid for {claim_object}. Forcing fallback validation."
                    )
                    context.object_part = self._deterministic_fallback(user_claim, claim_object).object_part

            context.issue_family = ISSUE_TYPE_TO_FAMILY.get(context.issue_type, "unknown")
            return context

        except Exception as e:
            logger.error(f"Model parser failed with error: {e}. Executing deterministic fallback parser.")
            return self._deterministic_fallback(user_claim, claim_object)

    def _deterministic_fallback(self, user_claim: str, claim_object: str) -> ClaimContext:
        """
        Fallback parser that extracts claims using simple pattern matching.
        """
        text_lower = user_claim.lower()

        # Extract issue type
        issue_type = "unknown"
        if "dent" in text_lower:
            issue_type = "dent"
        elif "scratch" in text_lower:
            issue_type = "scratch"
        elif "shatter" in text_lower or "smash" in text_lower:
            issue_type = "glass_shatter"
        elif "crack" in text_lower or "line" in text_lower:
            issue_type = "crack"
        elif "broken" in text_lower or "snap" in text_lower or "tear" in text_lower:
            if claim_object == "package":
                issue_type = "torn_packaging"
            else:
                issue_type = "broken_part"
        elif "missing" in text_lower or "lost" in text_lower:
            issue_type = "missing_part"
        elif "torn" in text_lower or "ripped" in text_lower:
            issue_type = "torn_packaging"
        elif "crushed" in text_lower or "bashed" in text_lower or "squished" in text_lower:
            issue_type = "crushed_packaging"
        elif "water" in text_lower or "wet" in text_lower or "liquid" in text_lower:
            issue_type = "water_damage"
        elif "stain" in text_lower or "spill" in text_lower or "coffee" in text_lower:
            issue_type = "stain"
        elif "no damage" in text_lower or "clean" in text_lower:
            issue_type = "none"

        # Extract object part
        object_part = "unknown"
        valid_parts = SUPPORTED_PARTS.get(claim_object, set())
        for part in valid_parts:
            part_spaced = part.replace("_", " ")
            if part_spaced in text_lower or part in text_lower:
                object_part = part
                break

        # Fallback default parts if none identified
        if object_part == "unknown":
            if claim_object == "car":
                object_part = "body"
            elif claim_object == "laptop":
                object_part = "body"
            elif claim_object == "package":
                object_part = "box"

        issue_family = ISSUE_TYPE_TO_FAMILY.get(issue_type, "unknown")

        logger.info(f"Fallback matched: issue_type={issue_type}, object_part={object_part}")

        return ClaimContext(
            claim_summary=f"Fallback claim parser extraction for {claim_object}.",
            issue_type=issue_type,
            object_part=object_part,
            issue_family=issue_family,
        )
