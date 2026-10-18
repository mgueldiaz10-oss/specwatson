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

__version__ = "0.1.0"
__author__ = "Miguel Diaz / Molecular Spectroscopy Lab (UGto)"
