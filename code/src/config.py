"""
Configuration module for the damage claim verification system.
Contains supported vocabulary, path mappings, and default settings.
"""

from typing import Dict, List, Set

# Supported high-level object types
SUPPORTED_OBJECT_TYPES: Set[str] = {"car", "laptop", "package"}

# Supported parts per object type
SUPPORTED_PARTS: Dict[str, Set[str]] = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

# Supported issue types
SUPPORTED_ISSUE_TYPES: Set[str] = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

# Mapping from issue types to families for evidence matching
ISSUE_TYPE_TO_FAMILY: Dict[str, str] = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass_shatter": "glass_shatter",
    "broken_part": "broken_part",
    "missing_part": "missing_part",
    "torn_packaging": "torn_packaging",
    "crushed_packaging": "crushed_packaging",
    "water_damage": "water_damage",
    "stain": "stain",
    "none": "none",
    "unknown": "unknown",
}

# Supported image quality and risk flags
ALLOWED_RISK_FLAGS: List[str] = [
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required"
]

IMAGE_QUALITY_FLAGS: Set[str] = {
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "possible_manipulation",
    "non_original_image",
}


# Severity categories
SUPPORTED_SEVERITIES: Set[str] = {"none", "low", "medium", "high", "unknown"}

# Client / Adapter configuration
DEFAULT_RETRY_ATTEMPTS: int = 3
DEFAULT_RETRY_BACKOFF_FACTOR: float = 2.0
DEFAULT_TIMEOUT: int = 30

# Image heuristics settings
BLUR_THRESHOLD: float = 100.0  # Laplacian variance threshold. Lower means blurrier.
BRIGHTNESS_LOW_THRESHOLD: float = 40.0  # Average pixel intensity below this is low light.
BRIGHTNESS_HIGH_THRESHOLD: float = 220.0  # Average pixel intensity above this is glare/overexposed.
CONTRAST_THRESHOLD: float = 30.0  # Standard deviation of pixel intensities. Lower means low contrast.
