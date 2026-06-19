"""
Image analysis module. Handles loading, preprocessing, running computer vision heuristics,
and querying the multimodal model.
"""

import logging
import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

from src.config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_HIGH_THRESHOLD,
    BRIGHTNESS_LOW_THRESHOLD,
    CONTRAST_THRESHOLD,
)
from src.models import BaseModelAdapter, ClaimContext, ImageAnalysisResult

logger = logging.getLogger("damage_claim_system.image_analyzer")


class ImageAnalyzer:
    """
    Performs preprocessing, local image quality checks using OpenCV/Pillow,
    and sends queries to the multimodal adapter.
    """

    def __init__(self, model_adapter: BaseModelAdapter):
        self.model_adapter = model_adapter

    def analyze(self, image_path: str, claim_context: ClaimContext, claim_object: str) -> ImageAnalysisResult:
        """
        Runs quality heuristics locally, then delegates semantic analysis to the model adapter.
        """
        logger.info(f"Analyzing image: {image_path}")

        # Check if the file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return ImageAnalysisResult(
                image_id=image_path,
                valid_image=False,
                object_detected="unknown",
                object_part="unknown",
                visible_damage=False,
                issue_type="unknown",
                severity="unknown",
                quality_flags=["cropped_or_obstructed"],
            )

        # Run local CV heuristics for image quality
        quality_heuristics_flags = self._run_quality_heuristics(image_path)

        try:
            # Query the multi-modal client
            analysis_result = self.model_adapter.analyze_image(
                image_path=image_path,
                claim_context=claim_context,
                claim_object=claim_object,
                quality_heuristics_flags=quality_heuristics_flags,
            )
            return analysis_result

        except Exception as e:
            logger.error(f"Failed model vision analysis for {image_path}: {e}. Falling back to heuristics-only validation.")
            # Deterministic fallback: image is marked invalid if heuristics flag severe issues
            is_valid = not any(
                f in quality_heuristics_flags for f in ["blurry_image"]
            )
            return ImageAnalysisResult(
                image_id=image_path,
                valid_image=is_valid,
                object_detected=claim_object,
                object_part=claim_context.object_part,
                visible_damage=False,
                issue_type="unknown",
                severity="unknown",
                quality_flags=quality_heuristics_flags,
            )

    def _run_quality_heuristics(self, image_path: str) -> List[str]:
        """
        Executes OpenCV heuristics to assess image quality: blur, low light, glare, low contrast.
        """
        flags = []
        try:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"OpenCV could not read image: {image_path}. Flagging obstructed.")
                return ["cropped_or_obstructed"]

            # Convert to grayscale for basic statistics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Blur Detection (Laplacian Variance)
            # Low variance means edges are less sharp, suggesting blur
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            if variance < BLUR_THRESHOLD:
                logger.info(f"Blur detected on {image_path} (Variance: {variance:.2f} < {BLUR_THRESHOLD})")
                flags.append("blurry_image")

            # 2. Brightness Check (Low Light & Glare)
            mean_brightness = np.mean(gray)
            if mean_brightness < BRIGHTNESS_LOW_THRESHOLD:
                logger.info(f"Low light detected on {image_path} (Brightness: {mean_brightness:.2f} < {BRIGHTNESS_LOW_THRESHOLD})")
                flags.append("low_light_or_glare")
            elif mean_brightness > BRIGHTNESS_HIGH_THRESHOLD:
                logger.info(f"Glare/Overexposure detected on {image_path} (Brightness: {mean_brightness:.2f} > {BRIGHTNESS_HIGH_THRESHOLD})")
                flags.append("low_light_or_glare")

            # 3. Contrast Check
            std_contrast = np.std(gray)
            if std_contrast < CONTRAST_THRESHOLD:
                logger.info(f"Low contrast detected on {image_path} (Contrast std: {std_contrast:.2f} < {CONTRAST_THRESHOLD})")
                if "low_light_or_glare" not in flags:
                    flags.append("low_light_or_glare")

        except Exception as e:
            logger.error(f"Error executing quality heuristics for {image_path}: {e}")
            flags.append("cropped_or_obstructed")

        return flags
