"""
SpecWatson: Spectrum Simulation and Line Shape Convolution Engine.
Convolves stick rovibronic transitions with Gaussian, Lorentzian, or Voigt line profiles
to produce high-resolution simulated spectra matching experimental laser bandwidths.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.special import voigt_profile
from .transitions import RovibronicLine


class SpectrumSimulator:
    """
    Synthesizes continuous rotationally resolved spectra from discrete lines
    using instrumental line profiles (Gaussian, Lorentzian, Voigt).
    """

    def __init__(
        self,
        lines: List[RovibronicLine],
        fwhm_mhz: float = 30.0,
        profile: str = "voigt",     # "gaussian", "lorentzian", "voigt"
        lorentz_fraction: float = 0.3
    ):
        self.lines = lines
        self.fwhm_mhz = max(1e-3, float(fwhm_mhz))
        self.profile = profile.lower()
        self.lorentz_fraction = float(np.clip(lorentz_fraction, 0.0, 1.0))

    def _convolve_subset(
        self,
        x: np.ndarray,
        freqs: np.ndarray,
        intensities: np.ndarray
    ) -> np.ndarray:
        """Fast local-window convolution of stick frequencies and intensities onto grid x."""
        y = np.zeros_like(x, dtype=np.float64)
        if len(freqs) == 0 or len(x) == 0:
            return y

        f_min, f_max = float(x[0]), float(x[-1])
        half_win = 10.0 * self.fwhm_mhz

        active_mask = (freqs >= f_min - half_win) & (freqs <= f_max + half_win) & (intensities > 0)
        active_freqs = freqs[active_mask]
        active_intensities = intensities[active_mask]
        if len(active_freqs) == 0:
            return y

        sigma = self.fwhm_mhz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = self.fwhm_mhz / 2.0
        n_pts = len(x)
        step = (f_max - f_min) / max(1, n_pts - 1) if n_pts > 1 else 1.0
        win_pts = max(15, int(np.ceil(half_win / max(1e-6, step))))

        if self.profile == "gaussian":
            norm_g = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
            inv_2sig2 = -0.5 / (sigma ** 2)
            indices = np.searchsorted(x, active_freqs)
            for idx_c, f0, I0 in zip(indices, active_freqs, active_intensities):
                i0 = max(0, idx_c - win_pts)
                i1 = min(n_pts, idx_c + win_pts)
                if i0 < i1:
                    dx = x[i0:i1] - f0
                    y[i0:i1] += I0 * norm_g * np.exp(inv_2sig2 * (dx ** 2))

        elif self.profile == "lorentzian":
            norm_l = gamma / np.pi
            gam2 = gamma ** 2
            indices = np.searchsorted(x, active_freqs)
            for idx_c, f0, I0 in zip(indices, active_freqs, active_intensities):
                i0 = max(0, idx_c - win_pts * 2)
                i1 = min(n_pts, idx_c + win_pts * 2)
                if i0 < i1:
                    dx = x[i0:i1] - f0
                    y[i0:i1] += I0 * norm_l / (dx ** 2 + gam2)

        elif self.profile == "voigt":
            sigma_g = max(1e-4, sigma * (1.0 - self.lorentz_fraction))
            gamma_l = max(1e-4, gamma * self.lorentz_fraction)
            indices = np.searchsorted(x, active_freqs)
            for idx_c, f0, I0 in zip(indices, active_freqs, active_intensities):
                i0 = max(0, idx_c - win_pts)
                i1 = min(n_pts, idx_c + win_pts)
                if i0 < i1:
                    dx = x[i0:i1] - f0
                    y[i0:i1] += I0 * voigt_profile(dx, sigma_g, gamma_l)
        else:
            raise ValueError(f"Unknown line profile: {self.profile}")

        return y

    def simulate(
        self,
        grid_points: int = 5000,
        margin_mhz: float = 200.0,
        freq_range_mhz: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the continuous spectrum (frequency_axis_mhz, intensity_axis).
        """
        if not self.lines:
            return np.array([]), np.array([])

        all_freqs = np.array([line.frequency_mhz for line in self.lines], dtype=np.float64)
        all_intensities = np.array([line.intensity for line in self.lines], dtype=np.float64)

        if freq_range_mhz is not None:
            f_min, f_max = float(freq_range_mhz[0]), float(freq_range_mhz[1])
        else:
            f_min = float(np.min(all_freqs)) - margin_mhz
            f_max = float(np.max(all_freqs)) + margin_mhz

        x = np.linspace(f_min, f_max, grid_points)
        y = self._convolve_subset(x, all_freqs, all_intensities)

        max_y = float(np.max(y)) if len(y) > 0 and np.max(y) > 0 else 1.0
        y_norm = y / max_y

        return x, y_norm

    def simulate_subbands(
        self,
        grid_points: int = 2500,
        margin_mhz: float = 200.0,
        freq_range_mhz: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Simulates the total spectrum AND the separate Type-a, Type-b, and Type-c sub-spectra,
        all normalized consistently to the peak of the total spectrum.
        """
        if not self.lines:
            empty = np.array([])
            return empty, empty, {"a": empty, "b": empty, "c": empty}

        all_freqs = np.array([line.frequency_mhz for line in self.lines], dtype=np.float64)
        all_intensities = np.array([line.intensity for line in self.lines], dtype=np.float64)
        all_types = np.array([line.band_type.lower() for line in self.lines])

        if freq_range_mhz is not None:
            f_min, f_max = float(freq_range_mhz[0]), float(freq_range_mhz[1])
        else:
            f_min = float(np.min(all_freqs)) - margin_mhz
            f_max = float(np.max(all_freqs)) + margin_mhz

        x = np.linspace(f_min, f_max, grid_points)
        sub_raw = {}
        y_tot = np.zeros_like(x, dtype=np.float64)

        for btype in ("a", "b", "c"):
            mask = all_types == btype
            if np.any(mask):
                y_sub = self._convolve_subset(x, all_freqs[mask], all_intensities[mask])
            else:
                y_sub = np.zeros_like(x, dtype=np.float64)
            sub_raw[btype] = y_sub
            y_tot += y_sub

        max_y = float(np.max(y_tot)) if len(y_tot) > 0 and np.max(y_tot) > 0 else 1.0
        y_norm = y_tot / max_y
        sub_norm = {k: v / max_y for k, v in sub_raw.items()}
        return x, y_norm, sub_norm

    def export_csv(self, filename: str, grid_points: int = 5000, in_wavenumbers: bool = False):
        """Exports the simulated spectrum to a standard 2-column CSV file."""
        x, y = self.simulate(grid_points=grid_points)
        if in_wavenumbers:
            x = x / 29979.2458
            header = "Wavenumber_cm1,Normalized_Intensity"
        else:
            header = "Frequency_MHz,Normalized_Intensity"
        
        data = np.column_stack((x, y))
        np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt="%.6f")
