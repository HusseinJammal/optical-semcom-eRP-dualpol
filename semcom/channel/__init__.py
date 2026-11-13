"""Fibre channel block: linear operator, eRP/RP (training), batched Manakov SSFM (evaluation), EDFA ASE."""
from .ase import add_ase, ase_floor_watt, ase_floor_watt_total, ase_snr_db, ase_snr_db_total
from .erp import rp_propagate_link
from .linear import D_op, dispersion_mask, omega_axis, propagate_link, span_gain
from .params import FiberConfig, LinkConfig, PropagationConfig, SimConfig
from .propagate import DIFFERENTIABLE_MODELS, MODELS, propagate

__all__ = ["add_ase", "ase_floor_watt", "ase_floor_watt_total", "ase_snr_db", "ase_snr_db_total",
           "rp_propagate_link", "D_op", "dispersion_mask", "omega_axis", "propagate_link", "span_gain",
           "FiberConfig", "LinkConfig", "PropagationConfig", "SimConfig", "DIFFERENTIABLE_MODELS", "MODELS",
           "propagate"]
