"""
SpecWatson: Spectrum Simulation and Line Shape Convolution Engine.
Convolves stick rovibronic transitions with Gaussian, Lorentzian, or Voigt line profiles
to produce high-resolution simulated spectra matching experimental laser bandwidths.
"""

import numpy as np
from typing import List, Tuple, Optional
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
        self.fwhm_mhz = fwhm_mhz
        self.profile = profile.lower()
        self.lorentz_fraction = np.clip(lorentz_fraction, 0.0, 1.0)

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

        all_freqs = np.array([line.frequency_mhz for line in self.lines])
        all_intensities = np.array([line.intensity for line in self.lines])

        if freq_range_mhz is not None:
            f_min, f_max = freq_range_mhz
        else:
            f_min = np.min(all_freqs) - margin_mhz
            f_max = np.max(all_freqs) + margin_mhz

        x = np.linspace(f_min, f_max, grid_points)
        y = np.zeros_like(x)

        # Standard deviation for Gaussian profile: FWHM = 2 * sqrt(2*ln(2)) * sigma
        sigma = self.fwhm_mhz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # Half-width at half-maximum for Lorentzian: gamma = FWHM / 2
        gamma = self.fwhm_mhz / 2.0

        # Filter lines within range to speed up computation
        active_mask = (all_freqs >= f_min - 5.0 * self.fwhm_mhz) & (all_freqs <= f_max + 5.0 * self.fwhm_mhz)
        active_freqs = all_freqs[active_mask]
        active_intensities = all_intensities[active_mask]

        if self.profile == "gaussian":
            # Vectorized Gaussian convolution
            for f0, I0 in zip(active_freqs, active_intensities):
                # Only compute in local window +/- 4*sigma
                dx = x - f0
                y += I0 * (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dx / sigma) ** 2)

        elif self.profile == "lorentzian":
            for f0, I0 in zip(active_freqs, active_intensities):
                dx = x - f0
                y += I0 * (1.0 / np.pi) * (gamma / (dx ** 2 + gamma ** 2))

        elif self.profile == "voigt":
            # True Voigt profile using Faddeeva function (scipy.special.voigt_profile)
            sigma_g = sigma * (1.0 - self.lorentz_fraction)
            gamma_l = gamma * self.lorentz_fraction
            for f0, I0 in zip(active_freqs, active_intensities):
                dx = x - f0
                y += I0 * voigt_profile(dx, max(1e-4, sigma_g), max(1e-4, gamma_l))
        else:
            raise ValueError(f"Unknown line profile: {self.profile}")

        # Normalize peak intensity to 1.0 for convenient comparison
        max_y = np.max(y) if len(y) > 0 and np.max(y) > 0 else 1.0
        y_norm = y / max_y

        return x, y_norm

    def export_csv(self, filename: str, grid_points: int = 5000, in_wavenumbers: bool = False):
        """Exports the simulated spectrum to a standard 2-column CSV file."""
        x, y = self.simulate(grid_points=grid_points)
        if in_wavenumbers:
            x = x / 29979.2458
            header = "Wavenumber_cm1,Normalized_Intensity"
        else:
            header = "Frequency_MHz,Normalized_Intensity"

        data = np.column_stack((x, y))
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
