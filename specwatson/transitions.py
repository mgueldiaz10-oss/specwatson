"""
SpecWatson: Transitions and Selection Rules for Asymmetric Rotor Electronic Spectra.
Computes allowed rovibronic transitions, Hönl-London / asymmetric line strengths,
Boltzmann populations, and transition dipole projections (pure Type a, Type b, Type c, or hybrid bands).
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache


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
    
    line_strength: float      # Quantum mechanical dipole matrix element squared S(J'tau' <- J''tau'')
    population: float         # Thermal Boltzmann factor of lower state exp(-E'' / kT) / Q_rot
    intensity: float          # population * line_strength * dipole_fraction


@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """
    Computes Wigner 3-j symbol using exact integer factorials (Racah formula).
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

    fact = math.factorial
    sum_val = 0.0
    for t in range(int(t_min), int(t_max) + 1):
        denom = (
            fact(t)
            * fact(j1 + j2 - j3 - t)
            * fact(j1 - m1 - t)
            * fact(j2 + m2 - t)
            * fact(j3 - j2 + m1 + t)
            * fact(j3 - j1 - m2 + t)
        )
        sum_val += ((-1.0) ** t) / denom

    prefactor = ((-1.0) ** (j1 - j2 - m3)) * math.sqrt(
        (fact(j1 + j2 - j3) * fact(j1 - j2 + j3) * fact(-j1 + j2 + j3))
        / fact(j1 + j2 + j3 + 1)
    ) * math.sqrt(
        fact(j1 + m1) * fact(j1 - m1)
        * fact(j2 + m2) * fact(j2 - m2)
        * fact(j3 + m3) * fact(j3 - m3)
    )

    return float(prefactor * sum_val)


class RovibronicTransitionCalculator:
    """
    Calculates transition frequencies and intensities between two electronic states
    (e.g., S0 -> S1 or X -> A) for an asymmetric top molecule in the I^r representation.

    Selection rules by Band Type (Four-Group D2 / Wang Symmetry):
        - Type a (dipole along principal axis a): ΔKa = even (0, ±2...), ΔKc = odd (±1, ±3...)
        - Type b (dipole along principal axis b): ΔKa = odd (±1, ±3...), ΔKc = odd (±1, ±3...)
        - Type c (dipole along principal axis c): ΔKa = odd (±1, ±3...), ΔKc = even (0, ±2...)
    """

    def __init__(
        self,
        ground_manifold: Dict[int, Dict],
        excited_manifold: Dict[int, Dict],
        band_origin_cm1: float = 30000.0,
        temperature_K: float = 5.0,
        dipole_fractions: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # (mu_a^2, mu_b^2, mu_c^2)
        selection_criterion: Optional[str] = None,  # 'a', 'b', 'c', or 'hybrid'/None
    ):
        self.ground = ground_manifold
        self.excited = excited_manifold
        self.band_origin_cm1 = band_origin_cm1
        self.temperature_K = max(1e-4, temperature_K)

        crit = (selection_criterion or "").strip().lower()
        if crit == "a":
            self.mu_a2, self.mu_b2, self.mu_c2 = 1.0, 0.0, 0.0
            self.selection_criterion = "a"
        elif crit == "b":
            self.mu_a2, self.mu_b2, self.mu_c2 = 0.0, 1.0, 0.0
            self.selection_criterion = "b"
        elif crit == "c":
            self.mu_a2, self.mu_b2, self.mu_c2 = 0.0, 0.0, 1.0
            self.selection_criterion = "c"
        else:
            total_dip = float(dipole_fractions[0]) + float(dipole_fractions[1]) + float(dipole_fractions[2])
            if total_dip > 1e-9:
                self.mu_a2 = float(dipole_fractions[0]) / total_dip
                self.mu_b2 = float(dipole_fractions[1]) / total_dip
                self.mu_c2 = float(dipole_fractions[2]) / total_dip
            else:
                self.mu_a2, self.mu_b2, self.mu_c2 = 1.0, 0.0, 0.0
            self.selection_criterion = "hybrid"

    def compute_partition_function(self) -> float:
        """Calculates rotational partition function Q_rot for the ground state."""
        Q = 0.0
        E0 = self.ground[0]["energies"][0]
        for J, data in self.ground.items():
            energies = data["energies"]
            degeneracy = 2 * J + 1
            boltzmann = np.exp(-(energies - E0) / (KB_MHZ * self.temperature_K))
            Q += degeneracy * float(np.sum(boltzmann))
        return max(1e-12, float(Q))

    def calculate_transitions(
        self,
        J_max_ground: int,
        J_max_excited: int,
        min_intensity_threshold: float = 1e-9
    ) -> List[RovibronicLine]:
        """
        Computes all dipole-allowed transitions up to J_max using vectorized
        direction-cosine matrix transformations in the I^r representation:
            - S_a = (2J'+1)(2J''+1) |<J'tau' | Phi_za | J''tau''>|^2   (ΔKa even, ΔKc odd)
            - S_b = (2J'+1)(2J''+1) |<J'tau' | Phi_xb | J''tau''>|^2   (ΔKa odd,  ΔKc odd)
            - S_c = (2J'+1)(2J''+1) |<J'tau' | Phi_yc | J''tau''>|^2   (ΔKa odd,  ΔKc even)
        Note: S(J'tau' <- J''tau'') already sums over the (2J''+1) M-sublevels, so the
        thermal weight per M-sublevel is exp(-E'' / k_B T) / Q_rot (matching JB95 & PGOPHER).
        """
        Q_rot = self.compute_partition_function()
        E0 = self.ground[0]["energies"][0]
        lines: List[RovibronicLine] = []

        band_origin_mhz = self.band_origin_cm1 * WAVENUMBER_TO_MHZ
        do_a = self.mu_a2 > 1e-9
        do_b = self.mu_b2 > 1e-9
        do_c = self.mu_c2 > 1e-9

        for J_low in range(min(J_max_ground + 1, len(self.ground))):
            g_data = self.ground[J_low]
            E_low_arr = g_data["energies"]
            vec_low = g_data["vectors"]
            labels_low = g_data["labels"]
            K_low = g_data["K_vals"]

            # Thermal Boltzmann factor per M-state (S already includes (2*J_low + 1) sum over M'')
            boltz_arr = np.exp(-(E_low_arr - E0) / (KB_MHZ * self.temperature_K)) / Q_rot
            if np.max(boltz_arr) < 1e-12:
                continue

            Ka_low_arr = np.array([lbl[0] for lbl in labels_low], dtype=np.int32)
            Kc_low_arr = np.array([lbl[1] for lbl in labels_low], dtype=np.int32)

            for J_up in (J_low - 1, J_low, J_low + 1):
                if J_up < 0 or J_up > J_max_excited or J_up not in self.excited:
                    continue
                if J_low == 0 and J_up == 0:
                    continue

                e_data = self.excited[J_up]
                E_up_arr = e_data["energies"]
                vec_up = e_data["vectors"]
                labels_up = e_data["labels"]

                if J_up - J_low == -1:
                    branch = "P"
                elif J_up - J_low == 0:
                    branch = "Q"
                else:
                    branch = "R"

                dim_u = 2 * J_up + 1
                dim_l = 2 * J_low + 1
                prefactor = float(dim_u * dim_l)

                Ka_up_arr = np.array([lbl[0] for lbl in labels_up], dtype=np.int32)
                Kc_up_arr = np.array([lbl[1] for lbl in labels_up], dtype=np.int32)

                # Parity masks of shape (dim_u, dim_l)
                d_Ka_odd = ((Ka_up_arr[:, None] - Ka_low_arr[None, :]) % 2) != 0
                d_Kc_odd = ((Kc_up_arr[:, None] - Kc_low_arr[None, :]) % 2) != 0

                # 1. Type-a transitions (z = a axis, q = 0, K_up = K_low)
                if do_a:
                    w_0 = np.zeros((dim_u, dim_l), dtype=np.float64)
                    for idx_l, k_val in enumerate(K_low):
                        if -J_up <= k_val <= J_up:
                            w_0[J_up + int(k_val), idx_l] = wigner_3j(J_up, 1, J_low, -int(k_val), 0, int(k_val))
                    amp_0 = vec_up.T @ (w_0 @ vec_low)
                    S_a_mat = prefactor * (amp_0 ** 2)
                    mask_a = (~d_Ka_odd) & d_Kc_odd
                    S_a_mat = np.where(mask_a, S_a_mat, 0.0)
                else:
                    S_a_mat = None

                # 2. Type-b (x = b axis) and Type-c (y = c axis) transitions (q = +/- 1, K_up = K_low +/- 1)
                if do_b or do_c:
                    w_p = np.zeros((dim_u, dim_l), dtype=np.float64)
                    w_m = np.zeros((dim_u, dim_l), dtype=np.float64)
                    for idx_l, k_val in enumerate(K_low):
                        k_int = int(k_val)
                        if -J_up <= k_int + 1 <= J_up:
                            w_p[J_up + k_int + 1, idx_l] = wigner_3j(J_up, 1, J_low, -(k_int + 1), 1, k_int)
                        if -J_up <= k_int - 1 <= J_up:
                            w_m[J_up + k_int - 1, idx_l] = wigner_3j(J_up, 1, J_low, -(k_int - 1), -1, k_int)
                    amp_p = vec_up.T @ (w_p @ vec_low)
                    amp_m = vec_up.T @ (w_m @ vec_low)

                    if do_b:
                        # Pure Type-b: antisymmetric combination (amp_plus - amp_minus)/sqrt(2), ΔKa odd, ΔKc odd
                        S_b_mat = prefactor * 0.5 * ((amp_p - amp_m) ** 2)
                        mask_b = d_Ka_odd & d_Kc_odd
                        S_b_mat = np.where(mask_b, S_b_mat, 0.0)
                    else:
                        S_b_mat = None

                    if do_c:
                        # Pure Type-c: symmetric combination (amp_plus + amp_minus)/sqrt(2), ΔKa odd, ΔKc even
                        S_c_mat = prefactor * 0.5 * ((amp_p + amp_m) ** 2)
                        mask_c = d_Ka_odd & (~d_Kc_odd)
                        S_c_mat = np.where(mask_c, S_c_mat, 0.0)
                    else:
                        S_c_mat = None
                else:
                    S_b_mat = None
                    S_c_mat = None

                # Emit RovibronicLine objects for each active band type (strictly unmixed!)
                for b_type, weight_mu2, S_mat in (
                    ("a", self.mu_a2, S_a_mat),
                    ("b", self.mu_b2, S_b_mat),
                    ("c", self.mu_c2, S_c_mat),
                ):
                    if S_mat is None or weight_mu2 <= 1e-9:
                        continue
                    I_mat = weight_mu2 * S_mat * boltz_arr[None, :]
                    active_indices = np.argwhere(I_mat >= min_intensity_threshold)
                    for i_up, i_low in active_indices:
                        i_u = int(i_up)
                        i_l = int(i_low)
                        freq_mhz = float(band_origin_mhz + (E_up_arr[i_u] - E_low_arr[i_l]))
                        Ka_u, Kc_u = labels_up[i_u]
                        Ka_l, Kc_l = labels_low[i_l]
                        lines.append(
                            RovibronicLine(
                                J_upper=J_up,
                                Ka_upper=Ka_u,
                                Kc_upper=Kc_u,
                                J_lower=J_low,
                                Ka_lower=Ka_l,
                                Kc_lower=Kc_l,
                                branch=branch,
                                band_type=b_type,
                                frequency_cm1=freq_mhz * MHZ_TO_WAVENUMBER,
                                frequency_mhz=freq_mhz,
                                line_strength=float(S_mat[i_u, i_l]),
                                population=float(boltz_arr[i_l]),
                                intensity=float(I_mat[i_u, i_l]),
                            )
                        )

        lines.sort(key=lambda line: line.frequency_mhz)
        return lines
