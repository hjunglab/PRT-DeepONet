#!/usr/bin/env python3
"""Geometry inputs of the adsorption models for the 3000 base geometries.

    wg   pore mask of the adsorption geometry (irreversible_adsorption/mask/m<base>.npz, value > 0.5)
    GDF  normalized geodesic distance from the inlet (common.inlet_distance_norm)
Output: $PRT_WORK/irreversible_adsorption/geometry_features.npz
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import inlet_distance_norm
from irreversible_adsorption.core import adsorption_mask

bases = np.arange(1, common.N_BASE + 1)
mask = np.stack([(adsorption_mask(b) > 0.5).astype(np.int8) for b in bases])
gdf = np.stack([inlet_distance_norm(m.astype(int)) for m in mask])
out = os.path.join(common.work_dir("irreversible_adsorption"), "geometry_features.npz")
np.savez_compressed(out, bases=bases, mask=mask, gdf=gdf)
print("[save]", out)
