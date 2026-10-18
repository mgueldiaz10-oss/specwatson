"""
SpecWatson: Transitions and Selection Rules for Asymmetric Rotor Electronic Spectra.
Computes allowed rovibronic transitions, Hönl-London / asymmetric line strengths,
Boltzmann populations, and transition dipole projections (hybrid a/b/c bands).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from scipy.special import factorial


# Conversion factor: 1 cm^-1 = 29979.2458 MHz
MHZ_TO_WAVENUMBER = 1.0 / 29979.2458
WAVENUMBER_TO_MHZ = 29979.2458
KB_MHZ = 20836.61912  # Boltzmann constant in MHz / K


@dataclass
class RovibronicLine:
    """Represents a single rotationally resolved electronic transition."""
    J_upper: int
    Ka_upper: int
    Kc_upper: int
    J_lower: int
    Ka_lower: int
    Kc_lower: int
    
    branch: str       # 'P', 'Q', or 'R'
    band_type: str    # 'a', 'b', or 'c'
    
    frequency_cm1: float      # Frequency in cm^-1
    frequency_mhz: float      # Frequency in MHz
    
    line_strength: float      # Quantum mechanical dipole matrix element squared
    population: float         # Thermal Boltzmann population of lower state
    intensity: float          # population * line_strength * dipole_fraction


@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """
    Computes Wigner 3-j symbol using the Racah formula.
    Needed for direction cosine matrix elements between |J, K> states.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    if t_min > t_max:
        return 0.0

    sum_val = 0.0
    for t in range(int(t_min), int(t_max) + 1):
        denom = (factorial(t) *
                 factorial(j1 + j2 - j3 - t) *
                 factorial(j1 - m1 - t) *
                 factorial(j2 + m2 - t) *
                 factorial(j3 - j2 + m1 + t) *
                 factorial(j3 - j1 - m2 + t))
        term = ((-1.0) ** t) / denom
        sum_val += term

    prefactor = ((-1.0) ** (j1 - j2 - m3)) * np.sqrt(
        (factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) * factorial(-j1 + j2 + j3)) /
        factorial(j1 + j2 + j3 + 1)
    ) * np.sqrt(
        factorial(j1 + m1) * factorial(j1 - m1) *
        factorial(j2 + m2) * factorial(j2 - m2) *
        factorial(j3 + m3) * factorial(j3 - m3)
    )

    return prefactor * sum_val


class RovibronicTransitionCalculator:
    """
    Calculates transition frequencies and intensities between two electronic states
    (e.g., S0 -> S1 or X -> A) for an asymmetric top molecule.
    """

    def __init__(
        self,
        ground_manifold: Dict[int, Dict],
        excited_manifold: Dict[int, Dict],
        band_origin_cm1: float = 30000.0,
        temperature_K: float = 5.0,
        dipole_fractions: Tuple[float, float, float] = (1.0, 0.0, 0.0) # (mu_a^2, mu_b^2, mu_c^2)
    ):
        self.ground = ground_manifold
        self.excited = excited_manifold
        self.band_origin_cm1 = band_origin_cm1
        self.temperature_K = max(1e-4, temperature_K)
        
        # Normalize dipole components
        total_dip = sum(dipole_fractions)
        if total_dip > 0:
            self.mu_a2 = dipole_fractions[0] / total_dip
            self.mu_b2 = dipole_fractions[1] / total_dip
            self.mu_c2 = dipole_fractions[2] / total_dip
        else:
            self.mu_a2, self.mu_b2, self.mu_c2 = 1.0, 0.0, 0.0

    def compute_partition_function(self) -> float:
        """Calculates rotational partition function Q_rot for the ground state."""
        Q = 0.0
        # Reference energy E0 = lowest ground state energy
        E0 = self.ground[0]["energies"][0]
        
        for J, data in self.ground.items():
            energies = data["energies"]
            # Multiplicity (2J + 1)
            degeneracy = 2 * J + 1
            boltzmann = np.exp(-(energies - E0) / (KB_MHZ * self.temperature_K))
            Q += degeneracy * np.sum(boltzmann)
        return float(Q)

    def calculate_transitions(self, J_max_ground: int, J_max_excited: int) -> List[RovibronicLine]:
        """
        Computes all dipole-allowed transitions up to J_max.
        Selection rules:
            Delta J = 0, +/- 1 (P, Q, R branches; J=0 -> J'=0 forbidden)
            Type a: Ka: even, Kc: odd
            Type b: Ka: odd,  Kc: odd
            Type c: Ka: odd,  Kc: even
        """
        Q_rot = self.compute_partition_function()
        E0 = self.ground[0]["energies"][0]
        lines = []

        band_origin_mhz = self.band_origin_cm1 * WAVENUMBER_TO_MHZ

        for J_low in range(min(J_max_ground + 1, len(self.ground))):
            g_data = self.ground[J_low]
            E_low_arr = g_data["energies"]
            vec_low = g_data["vectors"]
            labels_low = g_data["labels"]
            K_low = g_data["K_vals"]

            for J_up in [J_low - 1, J_low, J_low + 1]:
                if J_up < 0 or J_up > J_max_excited or J_up not in self.excited:
                    continue
                if J_low == 0 and J_up == 0:
                    continue

                e_data = self.excited[J_up]
                E_up_arr = e_data["energies"]
                vec_up = e_data["vectors"]
                labels_up = e_data["labels"]
                K_up = e_data["K_vals"]

                # Branch designation
                if J_up - J_low == -1:
                    branch = "P"
                elif J_up - J_low == 0:
                    branch = "Q"
                else:
                    branch = "R"

                # Precompute 3-j symbols for direction cosines in symmetric top basis
                # Delta J = J_up - J_low
                # We need:
                # - Type a (q = 0): K_up - K_low = 0
                # - Type b, c (q = +/- 1): K_up - K_low = +/- 1
                
                # Check each upper and lower asymmetric level
                for i_low, (Ka_l, Kc_l) in enumerate(labels_low):
                    E_low = E_low_arr[i_low]
                    # Boltzmann population
                    boltz_pop = (2 * J_low + 1) * np.exp(-(E_low - E0) / (KB_MHZ * self.temperature_K)) / Q_rot
                    
                    if boltz_pop < 1e-12:
                        continue

                    v_l = vec_low[:, i_low] # eigenvector in symmetric basis

                    for i_up, (Ka_u, Kc_u) in enumerate(labels_up):
                        E_up = E_up_arr[i_up]
                        v_u = vec_up[:, i_up]

                        d_Ka = abs(Ka_u - Ka_l)
                        d_Kc = abs(Kc_u - Kc_l)

                        # Determine band types allowed by parity
                        # Type a: Ka even, Kc odd
                        # Type b: Ka odd,  Kc odd
                        # Type c: Ka odd,  Kc even
                        is_type_a = (d_Ka % 2 == 0) and (d_Kc % 2 != 0) and (self.mu_a2 > 0)
                        is_type_b = (d_Ka % 2 != 0) and (d_Kc % 2 != 0) and (self.mu_b2 > 0)
                        is_type_c = (d_Ka % 2 != 0) and (d_Kc % 2 == 0) and (self.mu_c2 > 0)

                        if not (is_type_a or is_type_b or is_type_c):
                            continue

                        # Compute line strength S(J', tau' <- J'', tau'')
                        # For Type a (mu along a-axis = z in I^r): q = 0, K_up = K_low
                        S_a = 0.0
                        if is_type_a:
                            amp_a = 0.0
                            for k_idx_l, k_l in enumerate(K_low):
                                # find matching K_up
                                if k_l in K_up:
                                    k_idx_u = np.where(K_up == k_l)[0][0]
                                    # 3j factor: (2J_up+1)(2J_low+1) * [ (J_up 1 J_low / -k_l 0 k_l) ]^2
                                    w = wigner_3j(J_up, 1, J_low, -k_l, 0, k_l)
                                    amp_a += v_u[k_idx_u] * v_l[k_idx_l] * w
                            S_a = (2 * J_up + 1) * (2 * J_low + 1) * (amp_a ** 2)

                        # For Type b/c (mu along x/y-axis in I^r): q = +/- 1, K_up - K_low = +/- 1
                        S_bc = 0.0
                        if is_type_b or is_type_c:
                            amp_plus = 0.0
                            amp_minus = 0.0
                            for k_idx_l, k_l in enumerate(K_low):
                                # K_up = k_l + 1
                                if (k_l + 1) in K_up:
                                    k_idx_u = np.where(K_up == (k_l + 1))[0][0]
                                    w = wigner_3j(J_up, 1, J_low, -(k_l + 1), 1, k_l)
                                    amp_plus += v_u[k_idx_u] * v_l[k_idx_l] * w
                                # K_up = k_l - 1
                                if (k_l - 1) in K_up:
                                    k_idx_u = np.where(K_up == (k_l - 1))[0][0]
                                    w = wigner_3j(J_up, 1, J_low, -(k_l - 1), -1, k_l)
                                    amp_minus += v_u[k_idx_u] * v_l[k_idx_l] * w

                            # Line strength for perpendicular component
                            # For type b (x-axis) vs type c (y-axis), linear combination
                            S_bc = (2 * J_up + 1) * (2 * J_low + 1) * (0.5 * (amp_plus - amp_minus)**2 + 0.5 * (amp_plus + amp_minus)**2)

                        total_line_strength = self.mu_a2 * S_a + (self.mu_b2 + self.mu_c2) * S_bc
                        total_intensity = boltz_pop * total_line_strength

                        if total_intensity < 1e-12:
                            continue

                        # Transition frequency
                        # Delta E in MHz = (band_origin_mhz + E_up) - E_low
                        freq_mhz = band_origin_mhz + (E_up - E_low)
                        freq_cm1 = freq_mhz * MHZ_TO_WAVENUMBER

                        primary_band = "a" if is_type_a else ("b" if is_type_b else "c")

                        lines.append(RovibronicLine(
                            J_upper=J_up, Ka_upper=Ka_u, Kc_upper=Kc_u,
                            J_lower=J_low, Ka_lower=Ka_l, Kc_lower=Kc_l,
                            branch=branch,
                            band_type=primary_band,
                            frequency_cm1=freq_cm1,
                            frequency_mhz=freq_mhz,
                            line_strength=float(total_line_strength),
                            population=float(boltz_pop),
                            intensity=float(total_intensity)
                        ))

        # Sort lines by frequency
        lines.sort(key=lambda line: line.frequency_cm1)
        return lines
