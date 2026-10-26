"""
SpecWatson: Spectrometer & JB95 Experimental Data Parsers and Automatic Quantum Line Assigner (io_parsers.py)
=============================================================================================================
Provides:
  1. Native reading, baseline processing, and parameter extraction for:
     - Raw laser spectrometer fluorescence / absorption scans (e.g. `PmtFla`, `.dat`, `.txt`, `.csv`, 2-col `.asc`)
     - JB95 / PGOPHER rotational line-list files (`.asc` format such as `IAR.asc`:
       `BandType J' Ka' Kc' J'' Ka'' Kc'' Intensity Freq_Obs Freq_Calc [Residual]`)
  2. Automatic Quantum Line Assigner (`auto_assign_spectrometer_lines`):
     - Detects experimental peaks with sub-grid parabolic interpolation from raw scans (`PmtFla`).
     - Matches experimental peaks to quantum rovibronic transitions (`A`, `B`, `C` type `J' Ka' Kc' <- J'' Ka'' Kc''`).
     - Optionally refines rotational constants `(A'', B'', C'', A', B', C', nu0)` via quantum least-squares on assigned lines.
     - Exports complete JB95-compatible `.asc` files identical in format to `IAR.asc`.
"""

import os
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from .hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from .transitions import (
    RovibronicLine,
    RovibronicTransitionCalculator,
    MHZ_TO_WAVENUMBER,
    WAVENUMBER_TO_MHZ,
)
from .spectrum import SpectrumSimulator


@dataclass
class ExperimentalDataset:
    """Standardized container for loaded experimental or JB95 reference data."""
    filename: str
    file_format: str  # "spectrometer_xy" (e.g. PmtFla) or "jb95_asc" (e.g. IAR.asc)
    freq_mhz: np.ndarray
    intensity_raw: np.ndarray
    intensity_norm: np.ndarray
    origin_mhz: float
    rel_freq_ghz: np.ndarray
    num_points_or_lines: int
    summary: str
    # Optional metadata extracted from JB95 .asc files:
    extracted_constants: Optional[Dict[str, float]] = None
    band_type_counts: Optional[Dict[str, int]] = None
    assigned_lines_count: int = 0
    assigned_rms_mhz: float = 0.0
    jb95_lines: List[RovibronicLine] = field(default_factory=list)


@dataclass
class AssignedQuantumLine:
    """Represents a theoretical transition evaluated for JB95 .asc output (assigned or unassigned)."""
    band_type: str      # 'A', 'B', or 'C'
    J_up: int
    Ka_up: int
    Kc_up: int
    J_low: int
    Ka_low: int
    Kc_low: int
    branch: str         # 'P', 'Q', or 'R'
    intensity_jb95: float
    freq_obs_mhz: float     # 0.0000 if unassigned, else experimental peak frequency in MHz
    freq_calc_mhz: float
    residual_mhz: float     # freq_obs_mhz - freq_calc_mhz if assigned, else 0.0
    is_assigned: bool = False

    def to_jb95_asc_row(self) -> str:
        """Formats the transition into the exact column layout of JB95 .asc files (like IAR.asc)."""
        b_let = self.band_type.upper()
        if self.is_assigned and self.freq_obs_mhz > 1.0:
            return (
                f"{b_let} {self.J_up:2d} {self.Ka_up:2d} {self.Kc_up:2d}  "
                f"{self.J_low:2d} {self.Ka_low:2d} {self.Kc_low:2d} "
                f"{self.intensity_jb95:7.3f} {self.freq_obs_mhz:14.4f} {self.freq_calc_mhz:14.4f} "
                f"{self.residual_mhz:8.4f}"
            )
        return (
            f"{b_let} {self.J_up:2d} {self.Ka_up:2d} {self.Kc_up:2d}  "
            f"{self.J_low:2d} {self.Ka_low:2d} {self.Kc_low:2d} "
            f"{self.intensity_jb95:7.3f}      0.0000 {self.freq_calc_mhz:14.4f}"
        )


@dataclass
class AutoAssignmentResult:
    """Result container from automatic quantum line assignment on a raw spectrometer scan."""
    total_catalog_lines: int
    assigned_count: int
    detected_peaks_count: int
    rms_residual_mhz: float
    asc_file_path: str
    asc_preview_text: str
    assigned_table_rows: List[List[object]]
    all_jb95_entries: List[AssignedQuantumLine]
    assigned_entries: List[AssignedQuantumLine]
    peak_freqs_mhz: np.ndarray
    peak_heights_norm: np.ndarray
    refined_constants: Dict[str, float]
    report_text: str


def _extract_constants_from_jb95_lines(lines: List[RovibronicLine]) -> Optional[Dict[str, float]]:
    """
    Recovers exact rotational constants (A0, B0, C0, A1, B1, C1) and band origin (nu0_mhz)
    from low-J transitions (J <= 3) in a JB95 .asc line list using exact algebraic
    combinations followed by Levenberg-Marquardt quantum eigenvalue refinement.
    """
    low_j_lines = [ln for ln in lines if ln.J_upper <= 3 and ln.J_lower <= 3]
    if len(low_j_lines) < 7:
        return None

    tmap: Dict[Tuple[int, int, int, int, int, int], float] = {}
    for ln in low_j_lines:
        key = (ln.J_upper, ln.Ka_upper, ln.Kc_upper, ln.J_lower, ln.Ka_lower, ln.Kc_lower)
        tmap[key] = ln.frequency_mhz

    all_freqs = np.array([ln.frequency_mhz for ln in low_j_lines], dtype=np.float64)
    nu0_guess = float(np.median(all_freqs))
    A0_g, B0_g, C0_g = 2200.0, 800.0, 500.0
    A1_g, B1_g, C1_g = 2180.0, 790.0, 495.0

    if (
        (2, 1, 2, 1, 0, 1) in tmap
        and (1, 1, 0, 1, 0, 1) in tmap
        and (1, 0, 1, 1, 1, 0) in tmap
        and (1, 0, 1, 2, 1, 2) in tmap
        and (2, 2, 1, 1, 1, 0) in tmap
        and (1, 1, 0, 2, 2, 1) in tmap
        and (1, 1, 1, 0, 0, 0) in tmap
    ):
        C1_g = (tmap[(2, 1, 2, 1, 0, 1)] - tmap[(1, 1, 0, 1, 0, 1)]) / 4.0
        C0_g = (tmap[(1, 0, 1, 1, 1, 0)] - tmap[(1, 0, 1, 2, 1, 2)]) / 4.0
        A1_g = (tmap[(2, 2, 1, 1, 1, 0)] - tmap[(1, 0, 1, 1, 1, 0)]) / 4.0
        A0_g = (tmap[(1, 1, 0, 1, 0, 1)] - tmap[(1, 1, 0, 2, 2, 1)]) / 4.0
        nu0_guess = tmap[(1, 1, 1, 0, 0, 0)] - (A1_g + C1_g)
        if (2, 2, 0, 1, 1, 1) in tmap and (2, 0, 2, 1, 1, 1) in tmap and (0, 0, 0, 1, 1, 1) in tmap:
            s21 = tmap[(2, 2, 0, 1, 1, 1)] + tmap[(2, 0, 2, 1, 1, 1)]
            B1_g = (s21 - 2.0 * tmap[(0, 0, 0, 1, 1, 1)]) / 4.0 - A1_g - C1_g
        if (1, 1, 1, 2, 2, 0) in tmap and (1, 1, 1, 2, 0, 2) in tmap:
            s12 = tmap[(1, 1, 1, 2, 2, 0)] + tmap[(1, 1, 1, 2, 0, 2)]
            B0_g = (2.0 * tmap[(1, 1, 1, 0, 0, 0)] - s12) / 4.0 - A0_g - C0_g

    sample_lines = low_j_lines[:40]

    def residuals(p):
        d_nu0, A0, B0, C0, A1, B1, C1 = p
        s0 = MoleculeState("S0", A=A0, B=B0, C=C0)
        s1 = MoleculeState("S1", A=A1, B=B1, C=C1)
        m0 = AsymmetricRotorHamiltonian(s0).compute_manifold(3)
        m1 = AsymmetricRotorHamiltonian(s1).compute_manifold(3)

        e0_lookup = {}
        e1_lookup = {}
        for J in range(4):
            for idx, (ka, kc) in enumerate(m0[J]["labels"]):
                e0_lookup[(J, ka, kc)] = m0[J]["energies"][idx]
            for idx, (ka, kc) in enumerate(m1[J]["labels"]):
                e1_lookup[(J, ka, kc)] = m1[J]["energies"][idx]

        res = []
        for ln in sample_lines:
            eu = e1_lookup.get((ln.J_upper, ln.Ka_upper, ln.Kc_upper))
            el = e0_lookup.get((ln.J_lower, ln.Ka_lower, ln.Kc_lower))
            if eu is not None and el is not None:
                f_pred = (nu0_guess + d_nu0) + (eu - el)
                res.append(f_pred - ln.frequency_mhz)
        return np.array(res, dtype=np.float64)

    try:
        x0 = np.array([0.0, A0_g, B0_g, C0_g, A1_g, B1_g, C1_g], dtype=np.float64)
        opt = least_squares(residuals, x0, xtol=1e-11, ftol=1e-11, max_nfev=80)
        d_nu0, A0, B0, C0, A1, B1, C1 = [float(v) for v in opt.x]
        nu0 = nu0_guess + d_nu0
        if A0 >= B0 >= C0 > 0 and A1 >= B1 >= C1 > 0:
            return {
                "origin_mhz": nu0,
                "origin_cm1": nu0 * MHZ_TO_WAVENUMBER,
                "A0": A0,
                "B0": B0,
                "C0": C0,
                "A1": A1,
                "B1": B1,
                "C1": C1,
            }
    except Exception:
        pass

    return None


def parse_spectrometer_file(
    filepath: str,
    fwhm_mhz: float = 45.0,
    grid_points: int = 2500,
    reference_origin_mhz: Optional[float] = None,
    subtract_baseline: bool = True
) -> ExperimentalDataset:
    """
    Reads either:
      - A JB95 `.asc` stick-spectrum / line-assignment file (e.g. `IAR.asc`)
      - A 2-column continuous spectrometer scan file (e.g. `PmtFla`, `.dat`, `.txt`, `.csv`)
    and returns a ready-to-plot ExperimentalDataset.
    """
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo experimental: {filepath}")

    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        raise ValueError(f"El archivo '{fname}' está vacío.")

    first_tokens = raw_lines[0].replace(",", " ").split()
    is_jb95_asc = (
        len(first_tokens) >= 10
        and first_tokens[0].upper() in ("A", "B", "C")
        and all(tok.lstrip("-").isdigit() for tok in first_tokens[1:7])
    )

    if is_jb95_asc:
        jb95_lines: List[RovibronicLine] = []
        type_counts = {"a": 0, "b": 0, "c": 0}
        assigned_residuals = []
        j_max_seen = 0

        for row in raw_lines:
            parts = row.replace(",", " ").split()
            if len(parts) < 10:
                continue
            b_char = parts[0].lower()
            if b_char not in ("a", "b", "c"):
                continue
            try:
                j_up, ka_up, kc_up = int(parts[1]), int(parts[2]), int(parts[3])
                j_lo, ka_lo, kc_lo = int(parts[4]), int(parts[5]), int(parts[6])
                inten = float(parts[7])
                f_obs = float(parts[8])
                f_calc = float(parts[9])
            except ValueError:
                continue

            j_max_seen = max(j_max_seen, j_up, j_lo)
            type_counts[b_char] = type_counts.get(b_char, 0) + 1
            if f_obs > 1.0:
                res_val = float(parts[10]) if len(parts) >= 11 else (f_obs - f_calc)
                assigned_residuals.append(res_val)

            dj = j_up - j_lo
            branch = "P" if dj < 0 else ("Q" if dj == 0 else "R")
            jb95_lines.append(
                RovibronicLine(
                    J_upper=j_up,
                    Ka_upper=ka_up,
                    Kc_upper=kc_up,
                    J_lower=j_lo,
                    Ka_lower=ka_lo,
                    Kc_lower=kc_lo,
                    branch=branch,
                    band_type=b_char,
                    frequency_cm1=f_calc * MHZ_TO_WAVENUMBER,
                    frequency_mhz=f_calc,
                    line_strength=inten,
                    population=1.0,
                    intensity=inten,
                )
            )

        if not jb95_lines:
            raise ValueError(f"No se encontraron transiciones válidas en el archivo JB95 '{fname}'.")

        extracted = _extract_constants_from_jb95_lines(jb95_lines)
        if reference_origin_mhz is not None and reference_origin_mhz > 0:
            origin_mhz = float(reference_origin_mhz)
        elif extracted is not None:
            origin_mhz = extracted["origin_mhz"]
        else:
            all_f = np.array([ln.frequency_mhz for ln in jb95_lines])
            all_w = np.array([ln.intensity for ln in jb95_lines])
            origin_mhz = float(np.average(all_f, weights=all_w))

        sim = SpectrumSimulator(lines=jb95_lines, fwhm_mhz=fwhm_mhz, profile="voigt")
        freq_grid, y_norm = sim.simulate(grid_points=grid_points)
        rel_freq_ghz = (freq_grid - origin_mhz) / 1000.0

        n_assigned = len(assigned_residuals)
        rms_mhz = float(np.sqrt(np.mean(np.square(assigned_residuals)))) if n_assigned > 0 else 0.0
        dom_type = max(type_counts, key=type_counts.get).upper()

        const_str = ""
        if extracted is not None:
            const_str = (
                f" | Constantes deducidas (Rotor Rígido): "
                f"A''={extracted['A0']:.2f}, B''={extracted['B0']:.2f}, C''={extracted['C0']:.2f} MHz; "
                f"A'={extracted['A1']:.2f}, B'={extracted['B1']:.2f}, C'={extracted['C1']:.2f} MHz; "
                f"nu0={extracted['origin_mhz']:.2f} MHz ({extracted['origin_cm1']:.4f} cm^-1)"
            )

        summary = (
            f"Archivo JB95 '{fname}': {len(jb95_lines):,} transiciones cuánticas (Tipo {dom_type}, J_max={j_max_seen}) | "
            f"Picos experimentales asignados: {n_assigned} (RMS = {rms_mhz:.2f} MHz)"
            f"{const_str}"
        )

        return ExperimentalDataset(
            filename=fname,
            file_format="jb95_asc",
            freq_mhz=freq_grid,
            intensity_raw=y_norm,
            intensity_norm=y_norm,
            origin_mhz=origin_mhz,
            rel_freq_ghz=rel_freq_ghz,
            num_points_or_lines=len(jb95_lines),
            summary=summary,
            extracted_constants=extracted,
            band_type_counts=type_counts,
            assigned_lines_count=n_assigned,
            assigned_rms_mhz=rms_mhz,
            jb95_lines=jb95_lines,
        )

    x_list = []
    y_list = []
    for row in raw_lines:
        if row[0] in ("#", ";", "%", "/"):
            continue
        parts = row.replace(",", " ").replace(";", " ").split()
        if len(parts) < 2:
            continue
        try:
            xv = float(parts[0])
            yv = float(parts[1])
            if np.isfinite(xv) and np.isfinite(yv):
                x_list.append(xv)
                y_list.append(yv)
        except ValueError:
            continue

    if len(x_list) < 5:
        raise ValueError(
            f"El archivo '{fname}' no contiene suficientes pares numéricos (Frecuencia, Intensidad)."
        )

    x_arr = np.array(x_list, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.float64)

    if x_arr[0] > x_arr[-1]:
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

    mean_x = float(np.mean(np.abs(x_arr)))
    span_x = float(x_arr[-1] - x_arr[0])
    if mean_x > 1e7:
        freq_mhz = x_arr
        unit_desc = "MHz absolutos"
    elif 1000.0 < mean_x < 1e5 and span_x < 500.0:
        freq_mhz = x_arr * WAVENUMBER_TO_MHZ
        unit_desc = "cm^-1 (convertido a MHz)"
    elif mean_x < 500.0 and span_x < 500.0:
        freq_mhz = x_arr * 1000.0
        unit_desc = "GHz relativos"
    else:
        freq_mhz = x_arr
        unit_desc = "MHz"

    if subtract_baseline:
        baseline = float(np.percentile(y_arr, 5.0))
        y_clean = np.clip(y_arr - baseline, 0.0, None)
    else:
        y_clean = np.clip(y_arr, 0.0, None)

    max_y = float(np.max(y_clean)) if np.max(y_clean) > 0 else 1.0
    y_norm = y_clean / max_y

    if reference_origin_mhz is not None and reference_origin_mhz > 0:
        origin_mhz = float(reference_origin_mhz)
    else:
        weights = y_norm ** 2
        if np.sum(weights) > 0:
            origin_mhz = float(np.average(freq_mhz, weights=weights))
        else:
            origin_mhz = float(0.5 * (freq_mhz[0] + freq_mhz[-1]))

    rel_freq_ghz = (freq_mhz - origin_mhz) / 1000.0
    step_mhz = float(np.median(np.diff(freq_mhz))) if len(freq_mhz) > 1 else 0.0

    summary = (
        f"Espectro Experimental del Espectrómetro '{fname}': {len(freq_mhz):,} puntos ({unit_desc}) | "
        f"Rango: {(freq_mhz[0]-origin_mhz)/1000.0:+.2f} a {(freq_mhz[-1]-origin_mhz)/1000.0:+.2f} GHz "
        f"(Paso d_nu = {step_mhz:.1f} MHz, nu0 ref = {origin_mhz:.2f} MHz / {origin_mhz*MHZ_TO_WAVENUMBER:.4f} cm^-1)"
    )

    return ExperimentalDataset(
        filename=fname,
        file_format="spectrometer_xy",
        freq_mhz=freq_mhz,
        intensity_raw=y_arr,
        intensity_norm=y_norm,
        origin_mhz=origin_mhz,
        rel_freq_ghz=rel_freq_ghz,
        num_points_or_lines=len(freq_mhz),
        summary=summary,
    )


def auto_assign_spectrometer_lines(
    exp_dataset: ExperimentalDataset,
    ground_state: MoleculeState,
    excited_state: MoleculeState,
    origin_mhz: float = 923604613.0295,
    J_max: int = 26,
    temperature_K: float = 2.623,
    dipole_fractions: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    selection_criterion: str = "b",
    tolerance_mhz: float = 5.5,
    min_peak_height: float = 0.05,
    min_jb95_intensity: float = 0.010,
    refine_constants: bool = False,
    output_filename: str = "Asignacion_SpecWatson_JB95.asc"
) -> AutoAssignmentResult:
    """
    Automatically assigns rotational quantum transitions (A / B / C  J' Ka' Kc' <- J'' Ka'' Kc'')
    to experimental peaks detected in a raw spectrometer scan (such as `PmtFla`),
    optionally refines (A', B', C', nu0) via quantum least squares, and writes a JB95 `.asc` file
    identical in structure to `IAR.asc`.
    """
    f_exp = exp_dataset.freq_mhz
    y_norm = exp_dataset.intensity_norm

    # 1. Sub-grid parabolic peak detection on the raw experimental spectrum
    peaks_idx, _ = find_peaks(
        y_norm,
        height=max(0.01, float(min_peak_height)),
        prominence=max(0.01, float(min_peak_height) * 0.35),
        distance=2
    )
    peak_freqs = []
    peak_heights = []
    for idx in peaks_idx:
        if 0 < idx < len(f_exp) - 1:
            y0, y1, y2 = float(y_norm[idx - 1]), float(y_norm[idx]), float(y_norm[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            delta = float(np.clip(delta, -0.75, 0.75))
            f_sub = float(f_exp[idx]) + delta * float(f_exp[idx + 1] - f_exp[idx])
            peak_freqs.append(f_sub)
            peak_heights.append(y1)
        else:
            peak_freqs.append(float(f_exp[idx]))
            peak_heights.append(float(y_norm[idx]))

    peak_freqs_arr = np.array(peak_freqs, dtype=np.float64)
    peak_heights_arr = np.array(peak_heights, dtype=np.float64)

    # 2. Compute quantum rotational manifolds and allowed transitions
    origin_cm1 = float(origin_mhz) * MHZ_TO_WAVENUMBER
    s0 = MoleculeState(
        name=ground_state.name,
        A=ground_state.A, B=ground_state.B, C=ground_state.C,
        Delta_J=ground_state.Delta_J, Delta_JK=ground_state.Delta_JK, Delta_K=ground_state.Delta_K,
        delta_J=ground_state.delta_J, delta_K=ground_state.delta_K
    )
    s1 = MoleculeState(
        name=excited_state.name,
        A=excited_state.A, B=excited_state.B, C=excited_state.C,
        Delta_J=excited_state.Delta_J, Delta_JK=excited_state.Delta_JK, Delta_K=excited_state.Delta_K,
        delta_J=excited_state.delta_J, delta_K=excited_state.delta_K,
        origin=origin_cm1
    )

    m0 = AsymmetricRotorHamiltonian(s0).compute_manifold(J_max)
    m1 = AsymmetricRotorHamiltonian(s1).compute_manifold(J_max)

    calc = RovibronicTransitionCalculator(
        ground_manifold=m0,
        excited_manifold=m1,
        band_origin_cm1=origin_cm1,
        temperature_K=temperature_K,
        dipole_fractions=dipole_fractions,
        selection_criterion=selection_criterion
    )
    lines = calc.calculate_transitions(J_max, J_max, min_intensity_threshold=1e-8)

    if not lines:
        raise ValueError("No se generaron transiciones cuánticas con los parámetros actuales.")

    # Scale intensities to JB95 standard units (where 1(1,1)<-0(0,0) has intensity 1.100, or max ~ 3.115)
    ref_line_int = next(
        (
            ln.intensity for ln in lines
            if (ln.J_upper, ln.Ka_upper, ln.Kc_upper, ln.J_lower, ln.Ka_lower, ln.Kc_lower) == (1, 1, 1, 0, 0, 0)
        ),
        max(ln.intensity for ln in lines) / 2.8318
    )
    scale_jb95 = 1.100 / max(1e-12, float(ref_line_int))

    # Filter catalog to transitions with JB95 intensity >= min_jb95_intensity (0.010 in IAR.asc)
    catalog_lines = [ln for ln in lines if ln.intensity * scale_jb95 >= min_jb95_intensity]
    # Sort descending by frequency (exact JB95 .asc convention)
    catalog_lines.sort(key=lambda x: x.frequency_mhz, reverse=True)

    # 3. Match experimental peaks to theoretical transitions within tolerance_mhz
    tol = max(0.5, float(tolerance_mhz))
    assigned_map: Dict[int, float] = {}  # index in catalog_lines -> assigned experimental freq_obs_mhz

    cat_freqs = np.array([ln.frequency_mhz for ln in catalog_lines], dtype=np.float64)
    cat_ints_jb95 = np.array([ln.intensity * scale_jb95 for ln in catalog_lines], dtype=np.float64)

    for f_pk in peak_freqs_arr:
        diffs = np.abs(cat_freqs - f_pk)
        cand_idxs = np.where((diffs <= tol) & (cat_ints_jb95 >= 0.08))[0]
        if len(cand_idxs) == 0:
            continue
        # Strongest transition in the window of this experimental peak
        best_idx = cand_idxs[int(np.argmax(cat_ints_jb95[cand_idxs]))]
        max_i = cat_ints_jb95[best_idx]
        best_f = cat_freqs[best_idx]

        # Assign the dominant transition (and its unresolved asymmetry doublet partner if within 2.0 MHz)
        for c_idx in cand_idxs:
            if cat_ints_jb95[c_idx] >= 0.75 * max_i and abs(cat_freqs[c_idx] - best_f) <= 2.0:
                # Keep closest experimental peak if already matched
                if c_idx not in assigned_map or abs(f_pk - cat_freqs[c_idx]) < abs(assigned_map[c_idx] - cat_freqs[c_idx]):
                    assigned_map[int(c_idx)] = float(f_pk)

    # 4. Optional least-squares refinement of (A', B', C', nu0) on the assigned quantum lines
    refined_info = {
        "A0": s0.A, "B0": s0.B, "C0": s0.C,
        "A1": s1.A, "B1": s1.B, "C1": s1.C,
        "origin_mhz": float(origin_mhz)
    }

    if refine_constants and len(assigned_map) >= 6:
        assigned_Subset = [(catalog_lines[idx], f_obs) for idx, f_obs in assigned_map.items() if catalog_lines[idx].J_upper <= 18]
        if len(assigned_Subset) >= 6:
            j_ref_max = max(max(ln.J_upper, ln.J_lower) for ln, _ in assigned_Subset)
            m0_ref = AsymmetricRotorHamiltonian(s0).compute_manifold(j_ref_max)
            e0_lookup = {}
            for J in range(j_ref_max + 1):
                for i_l, (ka, kc) in enumerate(m0_ref[J]["labels"]):
                    e0_lookup[(J, ka, kc)] = m0_ref[J]["energies"][i_l]

            def fit_res_func(p):
                d_nu, A1_try, B1_try, C1_try = p
                if not (A1_try >= B1_try >= C1_try > 0):
                    return np.full(len(assigned_Subset), 1e4)
                s1_try = MoleculeState("S1_try", A=A1_try, B=B1_try, C=C1_try)
                m1_try = AsymmetricRotorHamiltonian(s1_try).compute_manifold(j_ref_max)
                e1_lookup = {}
                for J in range(j_ref_max + 1):
                    for i_u, (ka, kc) in enumerate(m1_try[J]["labels"]):
                        e1_lookup[(J, ka, kc)] = m1_try[J]["energies"][i_u]

                res_vec = []
                for ln, f_obs in assigned_Subset:
                    eu = e1_lookup.get((ln.J_upper, ln.Ka_upper, ln.Kc_upper), 0.0)
                    el = e0_lookup.get((ln.J_lower, ln.Ka_lower, ln.Kc_lower), 0.0)
                    f_calc_new = (float(origin_mhz) + d_nu) + (eu - el)
                    res_vec.append(f_obs - f_calc_new)
                return np.array(res_vec, dtype=np.float64)

            try:
                p0 = np.array([0.0, s1.A, s1.B, s1.C], dtype=np.float64)
                opt = least_squares(fit_res_func, p0, xtol=1e-6, ftol=1e-6, max_nfev=25)
                d_nu_opt, A1_opt, B1_opt, C1_opt = [float(v) for v in opt.x]
                refined_info["A1"] = A1_opt
                refined_info["B1"] = B1_opt
                refined_info["C1"] = C1_opt
                refined_info["origin_mhz"] = float(origin_mhz) + d_nu_opt
            except Exception:
                pass

    # 5. Build complete list of AssignedQuantumLine entries & JB95 .asc text
    all_entries: List[AssignedQuantumLine] = []
    assigned_entries: List[AssignedQuantumLine] = []
    table_rows: List[List[object]] = []
    asc_lines_str: List[str] = []

    for idx, ln in enumerate(catalog_lines):
        i_jb95 = float(cat_ints_jb95[idx])
        is_asg = idx in assigned_map
        f_obs = assigned_map[idx] if is_asg else 0.0
        f_calc = float(ln.frequency_mhz)
        res_mhz = (f_obs - f_calc) if is_asg else 0.0

        entry = AssignedQuantumLine(
            band_type=ln.band_type.upper(),
            J_up=ln.J_upper,
            Ka_up=ln.Ka_upper,
            Kc_up=ln.Kc_upper,
            J_low=ln.J_lower,
            Ka_low=ln.Ka_lower,
            Kc_low=ln.Kc_lower,
            branch=ln.branch,
            intensity_jb95=i_jb95,
            freq_obs_mhz=f_obs,
            freq_calc_mhz=f_calc,
            residual_mhz=res_mhz,
            is_assigned=is_asg
        )
        all_entries.append(entry)
        asc_lines_str.append(entry.to_jb95_asc_row())

        if is_asg:
            assigned_entries.append(entry)
            table_rows.append([
                entry.band_type,
                f"{entry.J_up} ({entry.Ka_up}, {entry.Kc_up})",
                f"{entry.J_low} ({entry.Ka_low}, {entry.Kc_low})",
                entry.branch,
                round(entry.intensity_jb95, 3),
                round(entry.freq_obs_mhz, 4),
                round(entry.freq_calc_mhz, 4),
                round(entry.residual_mhz, 4),
            ])

    rms_val = (
        float(np.sqrt(np.mean([e.residual_mhz ** 2 for e in assigned_entries])))
        if assigned_entries else 0.0
    )

    asc_full_text = "\n".join(asc_lines_str) + "\n"
    out_path = os.path.join(tempfile.gettempdir(), output_filename)
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(asc_full_text)

    # Preview showing top assigned lines in JB95 .asc format
    assigned_preview_lines = [e.to_jb95_asc_row() for e in assigned_entries[:35]]
    preview_header = (
        "# Formato Exacto JB95 (.asc): Tipo  J' Ka' Kc'  J'' Ka'' Kc''  Int_Calc     Freq_Exp_MHz   Freq_Calc_MHz  Residual\n"
    )
    asc_preview = preview_header + "\n".join(assigned_preview_lines)

    report = (
        f"✅ Asignación Cuántica Automática Completada (Formato JB95 .asc)\n"
        f"• Archivo Experimental Analizado: '{exp_dataset.filename}' ({len(f_exp):,} puntos) → {len(peak_freqs_arr)} máximos experimentales detectados\n"
        f"• Catálogo Cuántico Generado: {len(all_entries):,} transiciones totales (Tipo {selection_criterion.upper()}, I_JB95 ≥ {min_jb95_intensity:.3f})\n"
        f"• Transiciones Cuánticas Asignadas Automáticamente (|ν_exp - ν_calc| ≤ {tol:.1f} MHz): {len(assigned_entries)} líneas\n"
        f"• Error Residual Global de Asignación (RMS): {rms_val:.2f} MHz\n"
        f"• Constantes Empleadas / Refinadas: A″={refined_info['A0']:.2f}, B″={refined_info['B0']:.2f}, C″={refined_info['C0']:.2f} MHz | "
        f"A′={refined_info['A1']:.2f}, B′={refined_info['B1']:.2f}, C′={refined_info['C1']:.2f} MHz | ν₀={refined_info['origin_mhz']:.2f} MHz\n"
        f"• Archivo .asc listo para descargar: '{output_filename}' ({len(all_entries):,} filas en formato idéntico a IAR.asc)"
    )

    return AutoAssignmentResult(
        total_catalog_lines=len(all_entries),
        assigned_count=len(assigned_entries),
        detected_peaks_count=len(peak_freqs_arr),
        rms_residual_mhz=rms_val,
        asc_file_path=out_path,
        asc_preview_text=asc_preview,
        assigned_table_rows=table_rows,
        all_jb95_entries=all_entries,
        assigned_entries=assigned_entries,
        peak_freqs_mhz=peak_freqs_arr,
        peak_heights_norm=peak_heights_arr,
        refined_constants=refined_info,
        report_text=report,
    )
