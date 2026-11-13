"""Classical baseline block: JPEG + IEEE 802.11n LDPC + Gray 64-QAM over the same link, and its task classifiers."""
from .chain import run_jpeg_ldpc
from .classifier import build_cifar_resnet18, classify_u8, jpeg_round_trip_u8, load_task_classifier, to_u8
from .jpeg import choose_jpeg_quality, jpeg_decode_rgb, jpeg_encode_rgb, split_jpeg
from .ldpc import RATES, FrameCodec, LDPC80211n

__all__ = ["run_jpeg_ldpc", "build_cifar_resnet18", "classify_u8", "jpeg_round_trip_u8", "load_task_classifier",
           "to_u8", "choose_jpeg_quality", "jpeg_decode_rgb", "jpeg_encode_rgb", "split_jpeg", "RATES", "FrameCodec",
           "LDPC80211n"]
