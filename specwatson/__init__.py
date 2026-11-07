"""
SpecWatson: High-Resolution Electronic Rovibronic Spectroscopy Suite
===================================================================
A modern Python/GPU-ready library for simulating and fitting rotationally
resolved electronic spectra using Rigid Rotor and Watson Centrifugal
Distortion Hamiltonians.

Designed for high-resolution laser spectroscopy (LIF, REMPI) in supersonic
molecular beams.
"""

from .hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from .transitions import RovibronicTransitionCalculator
from .spectrum import SpectrumSimulator
from .fitting import ContourFitter
from .io_parsers import (
    parse_spectrometer_file,
    auto_assign_spectrometer_lines,
    ExperimentalDataset,
    AssignedQuantumLine,
    AutoAssignmentResult,
    DetectedExperimentalPeak,
    render_jb95_workstation_html,
)
from .asymrotor import (
    asym_rotor,
    asym_rotor_lines,
    rotor_levels,
    planar_top_c,
    ray_kappa,
    convergent_grid_fit,
)

__version__ = "0.3.0"
__author__ = "Miguel Diaz · Dr. Leonardo Alvarez-Valtierra · Dr. Mario Ulises Delgado-Jaime"
