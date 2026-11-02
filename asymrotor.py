"""
Root-level compatibility module for Dr. Mario Ulises Delgado-Jaime's `asymrotor.py`.

Allows direct execution of:
    from asymrotor import asym_rotor, asym_rotor_lines, rotor_levels, planar_top_c, ray_kappa
"""

from specwatson.asymrotor import (
    asym_rotor,
    asym_rotor_lines,
    rotor_levels,
    planar_top_c,
    ray_kappa,
)

__all__ = [
    "asym_rotor",
    "asym_rotor_lines",
    "rotor_levels",
    "planar_top_c",
    "ray_kappa",
]
