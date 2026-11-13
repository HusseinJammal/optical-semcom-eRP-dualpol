"""Evaluation block: metrics, evaluation on the true (SSFM) channel, eRP validity, link SNR."""
from .erp_validity import PASS_MARGIN_DB, TRAINING_RULE_MARGIN_DB, erp_validity, verdict
from .evaluate import evaluate_ssfm, evaluate_training_channel, per_image_path, save_per_image
from .metrics import MetricAccumulator, accuracy, papr_db, psnr_per_image, ssim_per_image
from .snr_probe import link_snr

__all__ = ["PASS_MARGIN_DB", "TRAINING_RULE_MARGIN_DB", "erp_validity", "verdict", "evaluate_ssfm",
           "evaluate_training_channel", "per_image_path", "save_per_image", "MetricAccumulator", "accuracy", "papr_db",
           "psnr_per_image", "ssim_per_image", "link_snr"]
