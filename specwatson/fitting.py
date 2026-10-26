"""
SpecWatson: Automated Contour and Parameter Fitting Engine.
Uses global optimization (Differential Evolution / Genetic Algorithm) and local
gradient-free refinement (Nelder-Mead / Powell) to match experimental band contours.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy.optimize import differential_evolution, minimize
from .hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from .transitions import RovibronicTransitionCalculator
from .spectrum import SpectrumSimulator


class ContourFitter:
    """
    Automated contour fitting engine for rotationally resolved electronic spectra.
    Adjusts molecular parameters (A', B', C', origin, temperature, etc.)
    to maximize correlation or minimize RMS residual against experimental data.
    """

    def __init__(
        self,
        exp_freq_mhz: np.ndarray,
        exp_intensity: np.ndarray,
        ground_state: MoleculeState,
        initial_excited_state: MoleculeState,
        J_max: int = 15,
        fwhm_mhz: float = 30.0,
        temperature_K: float = 5.0,
        dipole_fractions: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        selection_criterion: Optional[str] = None
    ):
        # Normalize experimental data
        self.exp_freq = exp_freq_mhz
        max_exp = np.max(exp_intensity) if np.max(exp_intensity) > 0 else 1.0
        self.exp_int = exp_intensity / max_exp
        
        self.ground_state = ground_state
        self.initial_excited = initial_excited_state
        self.J_max = J_max
        self.fwhm_mhz = fwhm_mhz
        self.temperature_K = temperature_K
        self.dipole_fractions = dipole_fractions
        self.selection_criterion = selection_criterion

        # Precompute ground state manifold (fixed during excited state fit)
        g_ham = AsymmetricRotorHamiltonian(self.ground_state)
        self.ground_manifold = g_ham.compute_manifold(self.J_max)

    def _forward_model(self, A_p: float, B_p: float, C_p: float, origin_shift_mhz: float) -> np.ndarray:
        """Runs the forward quantum simulation for candidate excited state parameters."""
        exc_state = MoleculeState(
            name="S1_fit",
            A=A_p,
            B=B_p,
            C=C_p,
            Delta_J=self.initial_excited.Delta_J,
            Delta_JK=self.initial_excited.Delta_JK,
            Delta_K=self.initial_excited.Delta_K,
            delta_J=self.initial_excited.delta_J,
            delta_K=self.initial_excited.delta_K,
            origin=self.initial_excited.origin
        )
        e_ham = AsymmetricRotorHamiltonian(exc_state)
        exc_manifold = e_ham.compute_manifold(self.J_max)

        calc = RovibronicTransitionCalculator(
            ground_manifold=self.ground_manifold,
            excited_manifold=exc_manifold,
            band_origin_cm1=self.initial_excited.origin,
            temperature_K=self.temperature_K,
            dipole_fractions=self.dipole_fractions,
            selection_criterion=self.selection_criterion
        )
        lines = calc.calculate_transitions(self.J_max, self.J_max)

        if not lines:
            return np.zeros_like(self.exp_freq)

        for line in lines:
            line.frequency_mhz += origin_shift_mhz

        sim = SpectrumSimulator(lines, fwhm_mhz=self.fwhm_mhz, profile="gaussian")
        freq_range = (float(np.min(self.exp_freq)), float(np.max(self.exp_freq)))
        _, sim_y = sim.simulate(grid_points=len(self.exp_freq), freq_range_mhz=freq_range)

        return sim_y

    def objective_rms(self, params: np.ndarray) -> float:
        """RMS residual loss function between experimental and simulated contours."""
        A_p, B_p, C_p, origin_shift = params
        
        if not (A_p >= B_p >= C_p > 0):
            return 1e6

        sim_y = self._forward_model(A_p, B_p, C_p, origin_shift)
        
        if len(sim_y) != len(self.exp_int):
            return 1e6

        diff = self.exp_int - sim_y
        rms = np.sqrt(np.mean(diff ** 2))
        return float(rms)

    def objective_cross_correlation(self, params: np.ndarray) -> float:
        """
        Negative normalized cross-correlation (maximizes overlap between spectra).
        Robust against baseline shifts and noise.
        """
        A_p, B_p, C_p, origin_shift = params
        if not (A_p >= B_p >= C_p > 0):
            return 1.0

        sim_y = self._forward_model(A_p, B_p, C_p, origin_shift)
        dot_prod = np.dot(self.exp_int, sim_y)
        norm_exp = np.linalg.norm(self.exp_int)
        norm_sim = np.linalg.norm(sim_y)

        if norm_sim < 1e-9:
            return 1.0

        corr = dot_prod / (norm_exp * norm_sim)
        return float(-corr)

    def fit_global(
        self,
        bounds: List[Tuple[float, float]],
        maxiter: int = 50,
        popsize: int = 15,
        workers: int = 1
    ) -> Dict:
        """
        Performs global genetic/differential evolution optimization across parameter bounds.
        bounds = [(A_min, A_max), (B_min, B_max), (C_min, C_max), (shift_min, shift_max)]
        """
        result = differential_evolution(
            func=self.objective_cross_correlation,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            disp=False,
            workers=workers,
            seed=42
        )
        return {
            "success": result.success,
            "best_params": result.x,
            "best_correlation": -result.fun,
            "message": result.message
        }

    def refine_local(self, initial_guess: np.ndarray, method: str = "Nelder-Mead") -> Dict:
        """Refines parameter estimates using Nelder-Mead simplex algorithm."""
        result = minimize(
            fun=self.objective_rms,
            x0=initial_guess,
            method=method,
            options={"maxiter": 200, "disp": False}
        )
        return {
            "success": result.success,
            "refined_params": result.x,
            "final_rms": result.fun,
            "message": result.message
        }
