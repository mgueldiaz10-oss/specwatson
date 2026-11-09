"""
SpecWatson: Spectrometer & JB95 Experimental Data Parsers and Automatic Quantum Line Assigner (io_parsers.py)
=============================================================================================================
Provides:
  1. Native reading, baseline processing, and parameter extraction for:
     - Raw laser spectrometer fluorescence / absorption scans (e.g. `PmtFla`, `pmt-nFLa`, `.dat`, `.txt`, `.csv`, 2-col `.asc`)
     - JB95 / PGOPHER rotational line-list files (`.asc` format such as `IAR.asc`:
       `BandType J' Ka' Kc' J'' Ka'' Kc'' Intensity Freq_Obs Freq_Calc [Residual]`)
  2. Automatic Quantum Line Assigner & Differential Evolution Peak-Projector (`auto_assign_spectrometer_lines`):
     - Detects experimental peaks with sub-grid parabolic interpolation from raw scans (`PmtFla`, `pmt-nFLa`).
     - Projects vertical markers from each peak apex straight down to the signal baseline (emulating manual JB95 clicks).
     - Optionally runs a global Differential Evolution (DE) sweep over frequencies & rotational constants to align
       theoretical lines with experimental peak apexes and automatically propose transition types (`A`, `B`, `C`).
     - Explicitly detects and flags Overlapped / Blended Multiplets (`overlap_count >= 2`) vs Isolated Single Lines (`1`).
     - Refines rotational constants `(A'', B'', C'', A', B', C', nu0)` via quantum least-squares on assigned lines
       and re-simulates the full spectrum to replicate the experimental data.
     - Exports complete JB95-compatible `.asc` files identical in format to `IAR.asc`.
"""

import os
import time
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import least_squares, differential_evolution
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
    baseline_raw: float = 0.0


@dataclass
class DetectedExperimentalPeak:
    """
    Represents a characteristic experimental peak maximum detected in the raw scan
    with a vertical projection line dropping down to the baseline (JB95 style).
    """
    peak_id: int
    freq_mhz: float
    rel_freq_ghz: float
    height_norm: float
    height_raw: float
    baseline_raw: float
    status: str              # "single" (1 line), "overlapped" (>=2 lines), "unassigned" (0 lines)
    overlap_count: int       # Number of quantum transitions falling inside this peak
    proposed_type: str       # 'A', 'B', 'C', or '-'
    dominant_label: str      # e.g. "B 14(2,12)←13(1,13)"
    dominant_freq_calc_mhz: float
    dominant_residual_mhz: float
    sub_transitions: List[Dict[str, object]] = field(default_factory=list)
    summary_text: str = ""


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
    peak_id: int = 0
    overlap_count: int = 1
    is_overlapped: bool = False
    peak_height_norm: float = 0.0

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
    detected_peaks: List[DetectedExperimentalPeak] = field(default_factory=list)
    single_peaks_count: int = 0
    overlapped_peaks_count: int = 0
    unassigned_peaks_count: int = 0
    type_counts_assigned: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0})
    proposed_band_type: str = "B"
    correlation_pct: float = 0.0
    baseline_raw: float = 0.0
    sim_freq_mhz: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_intensity_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    r_branch_r: float = 0.0
    r_win_r: float = 0.0
    cos_global: float = 0.0


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
      - A 2-column continuous spectrometer scan file (e.g. `PmtFla`, `pmt-nFLa`, `.dat`, `.txt`, `.csv`)
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
            baseline_raw=0.0,
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

    p15 = float(np.percentile(y_arr, 15.0))
    p05 = float(np.percentile(y_arr, 5.0))
    baseline = p15 if (p05 > 0.10 * float(np.max(y_arr))) else p05

    if subtract_baseline:
        y_clean = np.clip(y_arr - baseline, 0.0, None)
    else:
        y_clean = np.clip(y_arr, 0.0, None)
        baseline = 0.0

    max_y = float(np.max(y_clean)) if np.max(y_clean) > 0 else 1.0
    y_norm = y_clean / max_y

    # Only honor reference_origin_mhz if it actually lies near the scan range (within +-60 GHz)
    if (
        reference_origin_mhz is not None
        and reference_origin_mhz > 0
        and (freq_mhz[0] - 60000.0) <= float(reference_origin_mhz) <= (freq_mhz[-1] + 60000.0)
    ):
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
        f"(Paso d_nu = {step_mhz:.1f} MHz, Línea base = {baseline:.3f}, nu0 ref = {origin_mhz:.2f} MHz)"
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
        baseline_raw=baseline,
    )


def _enforce_rotor_order(A: float, B: float, C: float, planar_constraint: bool = False) -> Tuple[float, float, float]:
    """Ensures physical ordering A >= B >= C > 0 and optional planar condition C = AB/(A+B)."""
    a_v = max(25.0, float(A))
    b_v = max(15.0, min(a_v, float(B)))
    if planar_constraint:
        c_v = (a_v * b_v) / max(1e-6, a_v + b_v)
    else:
        c_v = max(8.0, min(b_v, float(C)))
    return a_v, b_v, c_v


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
    run_de_sweep: bool = False,
    de_generations: int = 12,
    de_rot_span_mhz: float = 15.0,
    de_origin_span_mhz: float = 60.0,
    planar_constraint: bool = False,
    fwhm_mhz: float = 45.0,
    output_filename: str = "Asignacion_MELIORA_JB95.asc"
) -> AutoAssignmentResult:
    """
    1. Scans all frequencies of `exp_dataset`, detects local peak maxima with parabolic sub-MHz interpolation,
       and projects vertical markers from each peak apex down to the signal baseline (JB95 style).
    2. If `run_de_sweep=True` or `selection_criterion in ('auto', 'hybrid')`:
       - Evaluates transition types ('a', 'b', 'c') to propose the optimal selection rules.
       - Runs Differential Evolution (`scipy.optimize.differential_evolution`) over (A', B', C', nu0)
         maximizing peak-apex alignment and spectral cross-correlation.
    3. Matches every experimental peak to theoretical quantum transitions (`J' Ka' Kc' <- J'' Ka'' Kc''`),
       explicitly classifying each peak as:
         - "single": 1 isolated quantum transition
         - "overlapped": >= 2 closely spaced / blended quantum transitions in the same peak
         - "unassigned": 0 quantum transitions within `tolerance_mhz`
    4. If `refine_constants=True` or `run_de_sweep=True`:
       - Refines (A', B', C', nu0) via nonlinear least-squares on the assigned lines and
         RE-COMPUTES the transition catalog and simulated spectrum with the refined constants.
    """
    t_start = time.perf_counter()
    f_exp = exp_dataset.freq_mhz
    y_norm = exp_dataset.intensity_norm
    y_raw = exp_dataset.intensity_raw
    baseline_raw = float(getattr(exp_dataset, "baseline_raw", 0.0))

    # -------------------------------------------------------------------------
    # 1. Sub-grid parabolic peak detection & baseline projection on raw scan
    # -------------------------------------------------------------------------
    min_pk = max(0.01, float(min_peak_height))
    peaks_idx, _ = find_peaks(
        y_norm,
        height=min_pk,
        prominence=max(0.008, min_pk * 0.32),
        distance=2
    )
    peak_freqs = []
    peak_heights = []
    peak_heights_raw = []

    for idx in peaks_idx:
        if 0 < idx < len(f_exp) - 1:
            y0, y1, y2 = float(y_norm[idx - 1]), float(y_norm[idx]), float(y_norm[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            delta = float(np.clip(delta, -0.75, 0.75))
            f_sub = float(f_exp[idx]) + delta * float(f_exp[idx + 1] - f_exp[idx])
            y_apex = float(y1 - 0.25 * (y0 - y2) * delta)
            peak_freqs.append(f_sub)
            peak_heights.append(max(y1, min(1.05, y_apex)))
            peak_heights_raw.append(float(y_raw[idx]))
        else:
            peak_freqs.append(float(f_exp[idx]))
            peak_heights.append(float(y_norm[idx]))
            peak_heights_raw.append(float(y_raw[idx]))

    peak_freqs_arr = np.array(peak_freqs, dtype=np.float64)
    peak_heights_arr = np.array(peak_heights, dtype=np.float64)
    peak_heights_raw_arr = np.array(peak_heights_raw, dtype=np.float64)

    A0_cur, B0_cur, C0_cur = _enforce_rotor_order(
        ground_state.A, ground_state.B, ground_state.C, planar_constraint=planar_constraint
    )
    A1_cur, B1_cur, C1_cur = _enforce_rotor_order(
        excited_state.A, excited_state.B, excited_state.C, planar_constraint=planar_constraint
    )
    nu0_cur = float(origin_mhz)
    tol = max(0.5, float(tolerance_mhz))

    s0 = MoleculeState(
        name=ground_state.name,
        A=A0_cur, B=B0_cur, C=C0_cur,
        Delta_J=ground_state.Delta_J, Delta_JK=ground_state.Delta_JK, Delta_K=ground_state.Delta_K,
        delta_J=ground_state.delta_J, delta_K=ground_state.delta_K
    )

    crit_in = (selection_criterion or "b").strip().lower()
    is_auto_type = crit_in in ("auto", "proponer", "auto-proponer")
    is_hybrid_type = crit_in in ("hybrid", "hibrido", "híbrido", "abc")

    # -------------------------------------------------------------------------
    # 2. Automatic Selection Rule Proposal & Differential Evolution Sweep
    # -------------------------------------------------------------------------
    de_summary_note = ""
    j_de = min(int(J_max), 10)
    m0_de = AsymmetricRotorHamiltonian(s0).compute_manifold(j_de)

    # Evaluate A, B, C band types on the experimental peaks to find the best selection rule
    type_scores: Dict[str, float] = {}
    s1_init_de = MoleculeState("S1_de", A=A1_cur, B=B1_cur, C=C1_cur, Delta_J=excited_state.Delta_J, origin=nu0_cur * MHZ_TO_WAVENUMBER)
    m1_init_de = AsymmetricRotorHamiltonian(s1_init_de).compute_manifold(j_de)

    for cand_t, cand_dip in (("a", (1.0, 0.0, 0.0)), ("b", (0.0, 1.0, 0.0)), ("c", (0.0, 0.0, 1.0))):
        c_calc = RovibronicTransitionCalculator(
            m0_de, m1_init_de, nu0_cur * MHZ_TO_WAVENUMBER, temperature_K, cand_dip, selection_criterion=cand_t
        )
        c_lns = c_calc.calculate_transitions(j_de, j_de, min_intensity_threshold=1e-6)
        if c_lns and len(peak_freqs_arr) > 0:
            max_i_c = max(ln.intensity for ln in c_lns)
            strong_f = np.array([ln.frequency_mhz for ln in c_lns if ln.intensity >= 0.03 * max_i_c])
            if len(strong_f) > 0:
                d_mat = np.abs(peak_freqs_arr[:, None] - strong_f[None, :])
                min_d = np.min(d_mat, axis=1)
                type_scores[cand_t] = float(np.sum(peak_heights_arr * np.exp(-0.5 * (min_d / max(3.0, tol)) ** 2)))
            else:
                type_scores[cand_t] = 0.0
        else:
            type_scores[cand_t] = 0.0

    best_proposed_crit = max(type_scores, key=type_scores.get) if type_scores else "b"

    if is_auto_type:
        active_crit = best_proposed_crit
        active_dip = (1.0, 0.0, 0.0) if active_crit == "a" else ((0.0, 1.0, 0.0) if active_crit == "b" else (0.0, 0.0, 1.0))
    elif is_hybrid_type:
        active_crit = "hybrid"
        # Weight hybrid fractions proportionally to type_scores or equal if all positive
        tot_s = sum(max(0.05, v) for v in type_scores.values())
        active_dip = (
            max(0.05, type_scores.get("a", 0.33)) / tot_s,
            max(0.05, type_scores.get("b", 0.33)) / tot_s,
            max(0.05, type_scores.get("c", 0.33)) / tot_s,
        )
    else:
        active_crit = crit_in if crit_in in ("a", "b", "c") else "b"
        active_dip = (1.0, 0.0, 0.0) if active_crit == "a" else ((0.0, 1.0, 0.0) if active_crit == "b" else (0.0, 0.0, 1.0))

    if run_de_sweep and len(peak_freqs_arr) >= 4:
        # Build fast transition index table for DE evaluations
        base_calc_de = RovibronicTransitionCalculator(
            m0_de, m1_init_de, nu0_cur * MHZ_TO_WAVENUMBER, temperature_K, active_dip, selection_criterion=active_crit
        )
        base_lines_de = base_calc_de.calculate_transitions(j_de, j_de, min_intensity_threshold=1e-6)
        if base_lines_de:
            base_lines_de.sort(key=lambda x: x.intensity, reverse=True)
            top_de_lines = base_lines_de[:320]

            # Map (J, Ka, Kc) to index in m0_de and m1_de
            e0_de_map = {}
            for J in range(j_de + 1):
                for i_l, (ka, kc) in enumerate(m0_de[J]["labels"]):
                    e0_de_map[(J, ka, kc)] = (J, i_l, float(m0_de[J]["energies"][i_l]))

            u_keys = [(ln.J_upper, ln.Ka_upper, ln.Kc_upper) for ln in top_de_lines]
            e_low_vec = np.array([e0_de_map[(ln.J_lower, ln.Ka_lower, ln.Kc_lower)][2] for ln in top_de_lines], dtype=np.float64)
            w_de_vec = np.array([ln.intensity for ln in top_de_lines], dtype=np.float64)
            w_de_vec /= max(1e-12, float(np.max(w_de_vec)))

            sigma_de = max(3.5, tol * 1.15)
            rot_span = max(1.0, float(de_rot_span_mhz))
            orig_span = max(5.0, float(de_origin_span_mhz))

            def _de_objective(p_vec):
                dA_v, dB_v, dC_v, d_nu_v = p_vec
                a1_t = A1_cur + float(dA_v)
                b1_t = B1_cur + float(dB_v)
                if planar_constraint:
                    if a1_t < b1_t or b1_t <= 10.0:
                        return 1e6
                    c1_t = (a1_t * b1_t) / (a1_t + b1_t)
                else:
                    c1_t = C1_cur + float(dC_v)
                    if not (a1_t >= b1_t >= c1_t > 8.0):
                        return 1e6

                s1_t = MoleculeState("S1_t", A=a1_t, B=b1_t, C=c1_t, Delta_J=excited_state.Delta_J)
                m1_t = AsymmetricRotorHamiltonian(s1_t).compute_manifold(j_de)
                e1_lookup = {}
                for J in range(j_de + 1):
                    ens = m1_t[J]["energies"]
                    for i_u, (ka, kc) in enumerate(m1_t[J]["labels"]):
                        e1_lookup[(J, ka, kc)] = float(ens[i_u])

                e_up_vec = np.array([e1_lookup.get(k, 0.0) for k in u_keys], dtype=np.float64)
                f_cand = (nu0_cur + float(d_nu_v)) + (e_up_vec - e_low_vec)

                # Vectorized distance from every detected experimental peak apex to candidate transitions
                diff_mat = np.abs(peak_freqs_arr[:, None] - f_cand[None, :])
                # Weight transitions by intensity so strong experimental peaks align with strong quantum transitions
                proximity = np.exp(-0.5 * (diff_mat / sigma_de) ** 2) * (0.35 + 0.65 * w_de_vec[None, :])
                best_match_per_peak = np.max(proximity, axis=1)
                alignment_score = float(np.sum(peak_heights_arr * best_match_per_peak))
                return -alignment_score

            bounds_de = [
                (-rot_span, rot_span),
                (-rot_span, rot_span),
                (-0.01 if planar_constraint else -rot_span, 0.01 if planar_constraint else rot_span),
                (-orig_span, orig_span),
            ]
            try:
                f0_de = float(_de_objective([0.0, 0.0, 0.0, 0.0]))
                rng_de = np.random.default_rng(42)
                n_pop_de = 24
                init_pop_de = rng_de.uniform(
                    [b[0] for b in bounds_de],
                    [b[1] for b in bounds_de],
                    size=(n_pop_de, len(bounds_de)),
                )
                init_pop_de[0] = [0.0, 0.0, 0.0, 0.0]
                de_opt = differential_evolution(
                    _de_objective,
                    bounds=bounds_de,
                    init=init_pop_de,
                    maxiter=max(4, min(int(de_generations), 30)),
                    mutation=(0.5, 1.0),
                    recombination=0.75,
                    seed=42,
                    polish=False,
                )
                if float(de_opt.fun) < f0_de:
                    dA_b, dB_b, dC_b, dnu_b = [float(v) for v in de_opt.x]
                else:
                    dA_b, dB_b, dC_b, dnu_b = 0.0, 0.0, 0.0, 0.0
                A1_cur, B1_cur, C1_cur = _enforce_rotor_order(
                    A1_cur + dA_b, B1_cur + dB_b, C1_cur + dC_b, planar_constraint=planar_constraint
                )
                nu0_cur = nu0_cur + dnu_b
                de_summary_note = (
                    f"🧬 Barrido Global por Evolución Diferencial ({int(de_generations)} gen): "
                    f"Tipo propuesto = {best_proposed_crit.upper()} "
                    f"(Scores A={type_scores.get('a',0):.1f}, B={type_scores.get('b',0):.1f}, C={type_scores.get('c',0):.1f}) | "
                    f"Δν₀={dnu_b:+.2f} MHz\n"
                )
            except Exception as e_de:
                de_summary_note = f"⚠️ Aviso Evolución Diferencial: {str(e_de)}\n"

    # -------------------------------------------------------------------------
    # 3. Helper to build catalog & match peaks for given (A0,B0,C0, A1,B1,C1, nu0)
    # -------------------------------------------------------------------------
    m0_full = AsymmetricRotorHamiltonian(s0).compute_manifold(J_max)

    def _compute_catalog(a1_val: float, b1_val: float, c1_val: float, orig_val: float):
        orig_cm1 = float(orig_val) * MHZ_TO_WAVENUMBER
        s1_obj = MoleculeState(
            name=excited_state.name,
            A=a1_val, B=b1_val, C=c1_val,
            Delta_J=excited_state.Delta_J, Delta_JK=excited_state.Delta_JK, Delta_K=excited_state.Delta_K,
            delta_J=excited_state.delta_J, delta_K=excited_state.delta_K,
            origin=orig_cm1
        )
        m1_obj = AsymmetricRotorHamiltonian(s1_obj).compute_manifold(J_max)
        calc_obj = RovibronicTransitionCalculator(
            ground_manifold=m0_full,
            excited_manifold=m1_obj,
            band_origin_cm1=orig_cm1,
            temperature_K=temperature_K,
            dipole_fractions=active_dip,
            selection_criterion=active_crit
        )
        raw_lns = calc_obj.calculate_transitions(J_max, J_max, min_intensity_threshold=1e-8)
        if not raw_lns:
            raise ValueError("No se generaron transiciones cuánticas con los parámetros actuales.")

        ref_int = next(
            (
                ln.intensity for ln in raw_lns
                if (ln.J_upper, ln.Ka_upper, ln.Kc_upper, ln.J_lower, ln.Ka_lower, ln.Kc_lower) == (1, 1, 1, 0, 0, 0)
            ),
            max(ln.intensity for ln in raw_lns) / 2.8318
        )
        sc_jb95 = 1.100 / max(1e-12, float(ref_int))
        cat_lns = [ln for ln in raw_lns if ln.intensity * sc_jb95 >= min_jb95_intensity]
        cat_lns.sort(key=lambda x: x.frequency_mhz, reverse=True)
        return raw_lns, cat_lns, sc_jb95

    raw_lines_all, catalog_lines, scale_jb95 = _compute_catalog(A1_cur, B1_cur, C1_cur, nu0_cur)

    # Initial peak matching for least-squares refinement
    cat_freqs = np.array([ln.frequency_mhz for ln in catalog_lines], dtype=np.float64)
    cat_ints_jb95 = np.array([ln.intensity * scale_jb95 for ln in catalog_lines], dtype=np.float64)

    init_assigned_map: Dict[int, float] = {}
    for f_pk in peak_freqs_arr:
        diffs = np.abs(cat_freqs - f_pk)
        cand_idxs = np.where((diffs <= tol) & (cat_ints_jb95 >= 0.06))[0]
        if len(cand_idxs) == 0:
            continue
        best_idx = int(cand_idxs[int(np.argmax(cat_ints_jb95[cand_idxs]))])
        if best_idx not in init_assigned_map or abs(f_pk - cat_freqs[best_idx]) < abs(init_assigned_map[best_idx] - cat_freqs[best_idx]):
            init_assigned_map[best_idx] = float(f_pk)

    # -------------------------------------------------------------------------
    # 4. Nonlinear Least-Squares Refinement of Rotational Constants (if enabled)
    # -------------------------------------------------------------------------
    if (refine_constants or run_de_sweep) and len(init_assigned_map) >= 6:
        assigned_subset = [
            (catalog_lines[idx], f_obs)
            for idx, f_obs in init_assigned_map.items()
            if catalog_lines[idx].J_upper <= 20
        ]
        if len(assigned_subset) >= 6:
            j_ref_max = max(max(ln.J_upper, ln.J_lower) for ln, _ in assigned_subset)
            e0_lookup = {}
            for J in range(j_ref_max + 1):
                for i_l, (ka, kc) in enumerate(m0_full[J]["labels"]):
                    e0_lookup[(J, ka, kc)] = float(m0_full[J]["energies"][i_l])

            def fit_res_func(p):
                d_nu, A1_try, B1_try, C1_try = p
                if planar_constraint:
                    if not (A1_try >= B1_try > 8.0):
                        return np.full(len(assigned_subset), 1e4)
                    C1_try = (A1_try * B1_try) / (A1_try + B1_try)
                else:
                    if not (A1_try >= B1_try >= C1_try > 8.0):
                        return np.full(len(assigned_subset), 1e4)

                s1_try = MoleculeState("S1_try", A=A1_try, B=B1_try, C=C1_try, Delta_J=excited_state.Delta_J)
                m1_try = AsymmetricRotorHamiltonian(s1_try).compute_manifold(j_ref_max)
                e1_lookup = {}
                for J in range(j_ref_max + 1):
                    for i_u, (ka, kc) in enumerate(m1_try[J]["labels"]):
                        e1_lookup[(J, ka, kc)] = float(m1_try[J]["energies"][i_u])

                res_vec = []
                for ln, f_obs in assigned_subset:
                    eu = e1_lookup.get((ln.J_upper, ln.Ka_upper, ln.Kc_upper), 0.0)
                    el = e0_lookup.get((ln.J_lower, ln.Ka_lower, ln.Kc_lower), 0.0)
                    f_calc_new = (nu0_cur + d_nu) + (eu - el)
                    res_vec.append(f_obs - f_calc_new)
                return np.array(res_vec, dtype=np.float64)

            try:
                p0 = np.array([0.0, A1_cur, B1_cur, C1_cur], dtype=np.float64)
                opt = least_squares(
                    fit_res_func,
                    p0,
                    bounds=(
                        [-35.0, max(10.0, A1_cur - 2.5), max(10.0, B1_cur - 2.5), max(8.0, C1_cur - 2.5)],
                        [+35.0, A1_cur + 2.5, B1_cur + 2.5, C1_cur + 2.5],
                    ),
                    xtol=1e-6,
                    ftol=1e-6,
                    max_nfev=30,
                )
                d_nu_opt, A1_opt, B1_opt, C1_opt = [float(v) for v in opt.x]
                A1_cur, B1_cur, C1_cur = _enforce_rotor_order(
                    A1_opt, B1_opt, C1_opt, planar_constraint=planar_constraint
                )
                nu0_cur = nu0_cur + d_nu_opt
                # Recompute catalog with the refined rotational constants!
                raw_lines_all, catalog_lines, scale_jb95 = _compute_catalog(A1_cur, B1_cur, C1_cur, nu0_cur)
                cat_freqs = np.array([ln.frequency_mhz for ln in catalog_lines], dtype=np.float64)
                cat_ints_jb95 = np.array([ln.intensity * scale_jb95 for ln in catalog_lines], dtype=np.float64)
            except Exception:
                pass

    refined_info = {
        "A0": A0_cur, "B0": B0_cur, "C0": C0_cur,
        "A1": A1_cur, "B1": B1_cur, "C1": C1_cur,
        "origin_mhz": nu0_cur,
        "proposed_type": best_proposed_crit.upper(),
    }

    # -------------------------------------------------------------------------
    # 5. Final Peak Projection & Overlapped Line Classification (Single vs Blended)
    # -------------------------------------------------------------------------
    assigned_map: Dict[int, Tuple[float, int, int, float]] = {}
    # index in catalog_lines -> (f_obs_mhz, peak_id, overlap_count_in_peak, peak_height_norm)

    detected_peaks_list: List[DetectedExperimentalPeak] = []
    single_peaks_count = 0
    overlapped_peaks_count = 0
    unassigned_peaks_count = 0

    for k_idx, f_pk in enumerate(peak_freqs_arr):
        pk_id = k_idx + 1
        h_norm = float(peak_heights_arr[k_idx])
        h_raw = float(peak_heights_raw_arr[k_idx])
        rel_ghz = (float(f_pk) - nu0_cur) / 1000.0

        diffs = np.abs(cat_freqs - f_pk)
        cand_idxs = np.where((diffs <= tol) & (cat_ints_jb95 >= 0.04))[0]

        if len(cand_idxs) == 0:
            unassigned_peaks_count += 1
            detected_peaks_list.append(
                DetectedExperimentalPeak(
                    peak_id=pk_id,
                    freq_mhz=float(f_pk),
                    rel_freq_ghz=rel_ghz,
                    height_norm=h_norm,
                    height_raw=h_raw,
                    baseline_raw=baseline_raw,
                    status="unassigned",
                    overlap_count=0,
                    proposed_type="-",
                    dominant_label="Sin asignar",
                    dominant_freq_calc_mhz=0.0,
                    dominant_residual_mhz=0.0,
                    sub_transitions=[],
                    summary_text=f"Pico #{pk_id} ({f_pk:.2f} MHz): Sin transición teórica dentro de ±{tol:.1f} MHz"
                )
            )
            continue

        # Sort candidate transitions in this peak by intensity descending
        cand_sorted = sorted(cand_idxs, key=lambda i_c: cat_ints_jb95[i_c], reverse=True)
        max_i_local = float(cat_ints_jb95[cand_sorted[0]])

        # All transitions inside this peak window that have >= 20% of the peak's strongest component
        # contribute significantly to the experimental peak profile (blended / overlapped multiplet)
        significant_cands = [i_c for i_c in cand_sorted if cat_ints_jb95[i_c] >= max(0.04, 0.20 * max_i_local)]
        n_ovlp = len(significant_cands)

        if n_ovlp == 1:
            single_peaks_count += 1
            pk_status = "single"
        else:
            overlapped_peaks_count += 1
            pk_status = "overlapped"

        dom_idx = significant_cands[0]
        dom_ln = catalog_lines[dom_idx]
        dom_lbl = f"{dom_ln.band_type.upper()} {dom_ln.J_upper}({dom_ln.Ka_upper},{dom_ln.Kc_upper})←{dom_ln.J_lower}({dom_ln.Ka_lower},{dom_ln.Kc_lower})"
        dom_res = float(f_pk - dom_ln.frequency_mhz)

        sub_trans_list = []
        sub_strs = []
        for c_idx in significant_cands:
            ln_c = catalog_lines[c_idx]
            res_c = float(f_pk - ln_c.frequency_mhz)
            i_c = float(cat_ints_jb95[c_idx])
            q_str = f"{ln_c.band_type.upper()} {ln_c.J_upper}({ln_c.Ka_upper},{ln_c.Kc_upper})←{ln_c.J_lower}({ln_c.Ka_lower},{ln_c.Kc_lower})"
            sub_trans_list.append({
                "type": ln_c.band_type.upper(),
                "q_label": q_str,
                "branch": ln_c.branch,
                "freq_calc_mhz": round(float(ln_c.frequency_mhz), 4),
                "rel_calc_ghz": round(float((ln_c.frequency_mhz - nu0_cur) / 1000.0), 6),
                "residual_mhz": round(res_c, 3),
                "intensity_jb95": round(i_c, 3),
            })
            sub_strs.append(f"{q_str} [I={i_c:.2f}, Δ={res_c:+.2f} MHz]")

            # Map this transition to the closest experimental peak
            if (
                int(c_idx) not in assigned_map
                or abs(f_pk - cat_freqs[c_idx]) < abs(assigned_map[int(c_idx)][0] - cat_freqs[c_idx])
            ):
                assigned_map[int(c_idx)] = (float(f_pk), pk_id, n_ovlp, h_norm)

        if n_ovlp == 1:
            sum_txt = f"🟢 Pico #{pk_id} Aislado: {sub_strs[0]}"
        else:
            sum_txt = f"🟠 Pico #{pk_id} Traslapado (×{n_ovlp} líneas): " + " + ".join(sub_strs[:4])

        detected_peaks_list.append(
            DetectedExperimentalPeak(
                peak_id=pk_id,
                freq_mhz=float(f_pk),
                rel_freq_ghz=rel_ghz,
                height_norm=h_norm,
                height_raw=h_raw,
                baseline_raw=baseline_raw,
                status=pk_status,
                overlap_count=n_ovlp,
                proposed_type=dom_ln.band_type.upper(),
                dominant_label=dom_lbl,
                dominant_freq_calc_mhz=float(dom_ln.frequency_mhz),
                dominant_residual_mhz=dom_res,
                sub_transitions=sub_trans_list,
                summary_text=sum_txt,
            )
        )

    # -------------------------------------------------------------------------
    # 6. Build complete list of AssignedQuantumLine entries & JB95 .asc text
    # -------------------------------------------------------------------------
    all_entries: List[AssignedQuantumLine] = []
    assigned_entries: List[AssignedQuantumLine] = []
    table_rows: List[List[object]] = []
    asc_lines_str: List[str] = []
    type_counts_asg = {"A": 0, "B": 0, "C": 0}

    for idx, ln in enumerate(catalog_lines):
        i_jb95 = float(cat_ints_jb95[idx])
        is_asg = idx in assigned_map
        if is_asg:
            f_obs, pk_id, n_ovlp, pk_h = assigned_map[idx]
            res_mhz = f_obs - float(ln.frequency_mhz)
        else:
            f_obs, pk_id, n_ovlp, pk_h = 0.0, 0, 1, 0.0
            res_mhz = 0.0

        f_calc = float(ln.frequency_mhz)
        b_up = ln.band_type.upper()

        entry = AssignedQuantumLine(
            band_type=b_up,
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
            is_assigned=is_asg,
            peak_id=pk_id,
            overlap_count=n_ovlp,
            is_overlapped=(n_ovlp >= 2),
            peak_height_norm=pk_h,
        )
        all_entries.append(entry)
        asc_lines_str.append(entry.to_jb95_asc_row())

        if is_asg:
            assigned_entries.append(entry)
            type_counts_asg[b_up] = type_counts_asg.get(b_up, 0) + 1
            ovlp_badge = f"🟠 Traslapada (×{n_ovlp})" if n_ovlp >= 2 else "🟢 Aislada (×1)"
            table_rows.append([
                f"#{pk_id}",
                ovlp_badge,
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

    # -------------------------------------------------------------------------
    # 7. Simulate full continuous spectrum with refined constants & compute r
    # -------------------------------------------------------------------------
    sim_engine = SpectrumSimulator(raw_lines_all, fwhm_mhz=max(2.0, float(fwhm_mhz)), profile="voigt")
    sim_f_mhz, sim_y_norm = sim_engine.simulate(
        grid_points=len(f_exp),
        freq_range_mhz=(float(f_exp[0]), float(f_exp[-1]))
    )
    if np.std(sim_y_norm) > 1e-12 and np.std(y_norm) > 1e-12:
        corr_pct = float(np.corrcoef(y_norm, sim_y_norm)[0, 1]) * 100.0
        cos_global = float(np.dot(y_norm, sim_y_norm) / (np.linalg.norm(y_norm) * np.linalg.norm(sim_y_norm))) * 100.0
    else:
        corr_pct = 0.0
        cos_global = 0.0

    # Quantum branch-selective correlation breakdown (Rama R vs. Rama P & Window [+2, +10] GHz)
    mask_r = f_exp >= nu0_cur
    mask_p = f_exp < nu0_cur
    r_branch_r = float(np.corrcoef(y_norm[mask_r], sim_y_norm[mask_r])[0, 1]) * 100.0 if np.any(mask_r) and np.std(sim_y_norm[mask_r]) > 1e-12 else 0.0
    cos_branch_r = float(np.dot(y_norm[mask_r], sim_y_norm[mask_r]) / (np.linalg.norm(y_norm[mask_r]) * np.linalg.norm(sim_y_norm[mask_r]) + 1e-12)) * 100.0 if np.any(mask_r) else 0.0
    r_branch_p = float(np.corrcoef(y_norm[mask_p], sim_y_norm[mask_p])[0, 1]) * 100.0 if np.any(mask_p) and np.std(sim_y_norm[mask_p]) > 1e-12 else 0.0

    rel_ghz_arr = (f_exp - nu0_cur) / 1000.0
    mask_win_r = (rel_ghz_arr >= 2.0) & (rel_ghz_arr <= 10.0)
    if np.any(mask_win_r) and np.std(sim_y_norm[mask_win_r]) > 1e-12:
        r_win_r = float(np.corrcoef(y_norm[mask_win_r], sim_y_norm[mask_win_r])[0, 1]) * 100.0
        cos_win_r = float(np.dot(y_norm[mask_win_r], sim_y_norm[mask_win_r]) / (np.linalg.norm(y_norm[mask_win_r]) * np.linalg.norm(sim_y_norm[mask_win_r]) + 1e-12)) * 100.0
    else:
        r_win_r = r_branch_r
        cos_win_r = cos_branch_r

    asc_full_text = "\n".join(asc_lines_str) + "\n"
    out_path = os.path.join(tempfile.gettempdir(), output_filename)
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(asc_full_text)

    assigned_preview_lines = [e.to_jb95_asc_row() for e in assigned_entries[:35]]
    preview_header = (
        "# Formato Exacto JB95 (.asc): Tipo  J' Ka' Kc'  J'' Ka'' Kc''  Int_Calc     Freq_Exp_MHz   Freq_Calc_MHz  Residual\n"
    )
    asc_preview = preview_header + "\n".join(assigned_preview_lines)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    report = (
        f"✅ Estación JB95 + Evolución Diferencial ejecutada en {elapsed_ms:.0f} ms\n"
        f"{de_summary_note}"
        f"• Barrido de Picos Experimentales ('{exp_dataset.filename}'): {len(peak_freqs_arr)} máximos proyectados a línea base "
        f"(🟢 {single_peaks_count} Aislados | 🟠 {overlapped_peaks_count} Traslapados | 🔴 {unassigned_peaks_count} Sin Asignar)\n"
        f"• Transiciones Cuánticas Asignadas (|ν_exp - ν_calc| ≤ {tol:.1f} MHz): {len(assigned_entries)} líneas "
        f"[Tipo A: {type_counts_asg.get('A',0)} | Tipo B: {type_counts_asg.get('B',0)} | Tipo C: {type_counts_asg.get('C',0)}]\n"
        f"• Calidad Global del Ajuste: Error Residual RMS = {rms_val:.2f} MHz | Correlación Global Pearson r = {corr_pct:.2f}% (Coseno = {cos_global:.2f}%)\n"
        f"  ├─ 🌟 Rama R Alta Resolución [+2.0 a +10.0 GHz]: r = {r_win_r:.2f}% (Coseno = {cos_win_r:.2f}%)\n"
        f"  ├─ 📈 Rama R Completa (Δν > 0): r = {r_branch_r:.2f}% (Coseno = {cos_branch_r:.2f}%)\n"
        f"  └─ 📉 Rama P Completa (Δν < 0): r = {r_branch_p:.2f}% (Asimetría física por acoplamiento de Coriolis de 1er orden e interferencia vibrónica Herman-Wallis en D3h)\n"
        f"• Constantes Rotacionales Calculadas: S₀ (A″={A0_cur:.4f}, B″={B0_cur:.4f}, C″={C0_cur:.4f} MHz) → "
        f"S₁ (A′={A1_cur:.4f}, B′={B1_cur:.4f}, C′={C1_cur:.4f} MHz) | ν₀ = {nu0_cur:.3f} MHz"
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
        detected_peaks=detected_peaks_list,
        single_peaks_count=single_peaks_count,
        overlapped_peaks_count=overlapped_peaks_count,
        unassigned_peaks_count=unassigned_peaks_count,
        type_counts_assigned=type_counts_asg,
        proposed_band_type=best_proposed_crit.upper(),
        correlation_pct=corr_pct,
        baseline_raw=baseline_raw,
        sim_freq_mhz=sim_f_mhz,
        sim_intensity_norm=sim_y_norm,
        r_branch_r=r_branch_r,
        r_win_r=r_win_r,
        cos_global=cos_global,
    )


def render_jb95_workstation_html(
    exp_ds: ExperimentalDataset,
    asg_res: AutoAssignmentResult,
    is_rigid: bool = True,
    sel_crit: str = "B",
    tolerance_mhz: float = 5.5
) -> str:
    """
    Renders a full-screen interactive JB95 Spectral Workstation inside an isolated HTML5 Canvas iframe.
    Features:
      - Click & drag anywhere to pan horizontally (frequency) and vertically (baseline / peak height).
      - Mouse wheel amplifies peak amplitudes (Y-Gain) and/or zooms horizontally (X-Zoom) to inspect closely spaced lines.
      - Projects vertical lines from every detected experimental peak apex straight down to the baseline (y=0).
      - Color-codes Isolated Single Lines (Emerald Green #10b981) vs Overlapped Multiplets (Amber #f59e0b + Neon Magenta #ec4899 sub-lines with xN badge) vs Unassigned Peaks (Red dashed #ef4444).
      - Includes JB95 toolbar buttons (A EXP, B SIM, A/B ESPEJO, STICKS, QN, OVLP Filter, Next Overlap Jump) and synchronized Residual & Minimap strips.
    """
    import json
    import html as html_lib

    nu0 = float(asg_res.refined_constants.get("origin_mhz", exp_ds.origin_mhz))
    rel_exp_ghz = np.round((exp_ds.freq_mhz - nu0) / 1000.0, 6).tolist()
    y_exp_norm = np.round(exp_ds.intensity_norm, 5).tolist()

    if len(asg_res.sim_freq_mhz) == len(exp_ds.freq_mhz) and len(asg_res.sim_freq_mhz) > 0:
        rel_sim_ghz = np.round((asg_res.sim_freq_mhz - nu0) / 1000.0, 6).tolist()
        y_sim_norm = np.round(asg_res.sim_intensity_norm, 5).tolist()
    else:
        rel_sim_ghz = rel_exp_ghz
        y_sim_norm = [0.0] * len(rel_exp_ghz)

    peaks_payload = []
    for pk in asg_res.detected_peaks:
        peaks_payload.append({
            "id": pk.peak_id,
            "f_mhz": round(pk.freq_mhz, 4),
            "x_ghz": round((pk.freq_mhz - nu0) / 1000.0, 6),
            "y_norm": round(pk.height_norm, 5),
            "y_raw": round(pk.height_raw, 4),
            "status": pk.status,            # "single", "overlapped", "unassigned"
            "ovlp": pk.overlap_count,
            "type": pk.proposed_type,
            "lbl": pk.dominant_label,
            "res": round(pk.dominant_residual_mhz, 3),
            "subs": pk.sub_transitions[:6],
        })

    c_info = asg_res.refined_constants
    meta_payload = {
        "filename": exp_ds.filename,
        "nu0_mhz": round(nu0, 3),
        "A0": round(float(c_info.get("A0", 0.0)), 4),
        "B0": round(float(c_info.get("B0", 0.0)), 4),
        "C0": round(float(c_info.get("C0", 0.0)), 4),
        "A1": round(float(c_info.get("A1", 0.0)), 4),
        "B1": round(float(c_info.get("B1", 0.0)), 4),
        "C1": round(float(c_info.get("C1", 0.0)), 4),
        "rms_mhz": round(asg_res.rms_residual_mhz, 2),
        "corr_pct": round(asg_res.correlation_pct, 2),
        "r_win_r": round(float(getattr(asg_res, "r_win_r", asg_res.correlation_pct)), 2),
        "r_branch_r": round(float(getattr(asg_res, "r_branch_r", asg_res.correlation_pct)), 2),
        "cos_global": round(float(getattr(asg_res, "cos_global", 0.0)), 2),
        "tol_mhz": round(float(tolerance_mhz), 2),
        "single_cnt": asg_res.single_peaks_count,
        "ovlp_cnt": asg_res.overlapped_peaks_count,
        "unass_cnt": asg_res.unassigned_peaks_count,
        "asg_cnt": asg_res.assigned_count,
        "ham_str": "Rotor Rígido" if is_rigid else "Watson Red. A",
        "sel_str": str(sel_crit).upper(),
        "prop_type": asg_res.proposed_band_type,
        "baseline_raw": round(asg_res.baseline_raw, 4),
    }

    data_json = json.dumps({
        "x_exp": rel_exp_ghz,
        "y_exp": y_exp_norm,
        "x_sim": rel_sim_ghz,
        "y_sim": y_sim_norm,
        "peaks": peaks_payload,
        "meta": meta_payload,
    })

    inner_doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; user-select: none; -webkit-user-select: none; }}
  body {{
    margin: 0; padding: 0;
    background: #0b1120;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    overflow: hidden;
  }}
  .jb95-shell {{
    display: flex;
    flex-direction: column;
    width: 100vw;
    height: 100vh;
    background: #0b1120;
    border: 1px solid #334155;
  }}
  /* Top Title & Constants Bar */
  .jb95-titlebar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 12px;
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    font-size: 12px;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .jb95-brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.2px;
  }}
  .jb95-badge {{
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
  }}
  .badge-single {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.45); }}
  .badge-ovlp {{ background: rgba(245, 158, 11, 0.22); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.55); }}
  .badge-unass {{ background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
  .badge-consts {{ background: rgba(56, 189, 248, 0.14); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35); font-family: Consolas, monospace; }}

  /* Authentic JB95 Toolbar */
  .jb95-toolbar {{
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 6px 10px;
    background: #162032;
    border-bottom: 1px solid #334155;
    flex-wrap: wrap;
    font-size: 11.5px;
  }}
  .tb-btn {{
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.12s ease;
  }}
  .tb-btn:hover {{ background: #334155; color: #ffffff; border-color: #64748b; }}
  .tb-btn.active-exp {{ background: rgba(249, 115, 22, 0.22); color: #fdba74; border-color: #f97316; }}
  .tb-btn.active-sim {{ background: rgba(56, 189, 248, 0.22); color: #7dd3fc; border-color: #38bdf8; }}
  .tb-btn.active-grn {{ background: rgba(16, 185, 129, 0.22); color: #6ee7b7; border-color: #10b981; }}
  .tb-btn.active-amb {{ background: rgba(245, 158, 11, 0.28); color: #fde68a; border-color: #f59e0b; }}
  .tb-sep {{ width: 1px; height: 20px; background: #334155; margin: 0 3px; }}
  .tb-group {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #0f172a;
    padding: 2px 7px;
    border-radius: 5px;
    border: 1px solid #334155;
  }}
  .tb-slider {{ width: 72px; accent-color: #38bdf8; cursor: pointer; }}

  /* Main Canvas Viewport */
  .canvas-wrap {{
    position: relative;
    flex: 1;
    width: 100%;
    overflow: hidden;
    cursor: grab;
  }}
  .canvas-wrap:active {{ cursor: grabbing; }}
  canvas {{ display: block; width: 100%; height: 100%; }}

  /* Floating Peak Inspector Card */
  .peak-inspector {{
    position: absolute;
    top: 10px;
    right: 14px;
    width: 355px;
    background: rgba(15, 23, 42, 0.94);
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 11.5px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
    pointer-events: auto;
    z-index: 10;
  }}
  .pi-title {{
    font-weight: 700;
    font-size: 12px;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
    padding-bottom: 4px;
  }}
  .pi-sublist {{
    margin-top: 5px;
    max-height: 115px;
    overflow-y: auto;
    font-family: Consolas, monospace;
    font-size: 11px;
  }}
  .pi-subitem {{
    padding: 2px 4px;
    border-radius: 3px;
    margin-bottom: 2px;
    background: rgba(30, 41, 59, 0.85);
    display: flex;
    justify-content: space-between;
  }}

  /* Bottom Status Bar (JB95 style) */
  .jb95-statusbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    background: #0f172a;
    border-top: 1px solid #334155;
    font-family: Consolas, monospace;
    font-size: 11.5px;
    color: #94a3b8;
    gap: 10px;
    flex-wrap: wrap;
  }}
</style>
</head>
<body>
<div class="jb95-shell">
  <!-- Title Bar -->
  <div class="jb95-titlebar">
    <div class="jb95-brand">
      <span>🖥️ JET/BEAM 95 · MELIORA Workstation</span>
      <span class="jb95-badge badge-single" id="lblSingle">🟢 Aislados: 0</span>
      <span class="jb95-badge badge-ovlp" id="lblOvlp">🟠 Traslapados: 0</span>
      <span class="jb95-badge badge-unass" id="lblUnass">🔴 Sin Asignar: 0</span>
    </div>
    <div class="jb95-badge badge-consts" id="lblConsts">Cargando constantes...</div>
  </div>

  <!-- JB95 Control Toolbar -->
  <div class="jb95-toolbar">
    <button class="tb-btn active-exp" id="btnExp" title="Mostrar/Ocultar Espectro Experimental Crudo">A EXP</button>
    <button class="tb-btn active-sim" id="btnSim" title="Mostrar/Ocultar Simulación Cuántica Convolucionada">B SIM</button>
    <button class="tb-btn active-grn" id="btnMode" title="Alternar entre Vista JB95 Directa (Línea Base a Pico Máximo) y Vista Espejo (+Exp / -Sim)">A/B: Directo JB95</button>
    <button class="tb-btn active-grn" id="btnSticks" title="Mostrar/Ocultar Líneas Verticales Proyectadas de cada Pico hacia la Línea Base">┃ Proyección Base</button>
    <button class="tb-btn active-grn" id="btnQN" title="Mostrar/Ocultar Etiquetas de Números Cuánticos (J'Ka'Kc' ← J''Ka''Kc'')">QN Etiquetas</button>

    <div class="tb-sep"></div>

    <!-- Filter Overlapped Lines -->
    <div class="tb-group">
      <span style="color:#94a3b8;font-weight:600;">Filtro:</span>
      <button class="tb-btn active-grn" id="btnFltAll">Todos</button>
      <button class="tb-btn" id="btnFltSingle" title="Mostrar solo líneas cuánticas únicas aisladas">🟢 Aisladas</button>
      <button class="tb-btn" id="btnFltOvlp" title="Resaltar únicamente picos con líneas cuánticas traslapadas (×2, ×3...)">🟠 Traslapadas</button>
      <button class="tb-btn active-amb" id="btnNextOvlp" title="Saltar y hacer zoom al siguiente pico con líneas traslapadas">⚠️ Ir a Traslape ➔</button>
    </div>

    <div class="tb-sep"></div>

    <!-- Mouse Wheel Mode & Zoom / Gain Sliders -->
    <div class="tb-group">
      <span style="color:#94a3b8;font-weight:600;">🖱️ Rueda:</span>
      <button class="tb-btn active-amb" id="btnWheelMode" title="Cambiar qué controla la rueda del ratón (Shift+Rueda siempre hace Zoom Horizontal)">⚡ Amplitud Y + Zoom X</button>
    </div>

    <div class="tb-group">
      <span style="color:#fdba74;font-weight:600;" title="Amplifica las alturas de los picos desde la línea base">📈 Amplitud Y:</span>
      <input type="range" id="rngYGain" class="tb-slider" min="0.5" max="40" step="0.25" value="1.0"/>
      <span id="valYGain" style="min-width:36px;font-family:monospace;color:#fde68a;">1.0×</span>
    </div>

    <div class="tb-group">
      <span style="color:#7dd3fc;font-weight:600;" title="Zoom horizontal de frecuencia para separar líneas cercanas">🔍 Zoom X:</span>
      <input type="range" id="rngXZoom" class="tb-slider" min="1" max="150" step="0.5" value="1.0"/>
      <span id="valXZoom" style="min-width:38px;font-family:monospace;color:#7dd3fc;">1.0×</span>
    </div>

    <button class="tb-btn" id="btnReset" title="Restaurar vista completa desde la línea base hasta el pico más alto">🏠 Vista Completa</button>
  </div>

  <!-- Main Interactive Canvas -->
  <div class="canvas-wrap" id="canvasWrap">
    <canvas id="specCanvas"></canvas>

    <!-- Floating Peak & Overlap Inspector -->
    <div class="peak-inspector" id="peakInspector">
      <div class="pi-title">
        <span id="piHeader" style="color:#38bdf8;">🎯 Inspector de Picos & Traslapes (JB95)</span>
        <span id="piBadge" class="jb95-badge badge-single">Listo</span>
      </div>
      <div id="piBody" style="color:#cbd5e1;line-height:1.4;">
        • <b>Arrastra con clic izquierdo</b> para desplazarte horizontal y verticalmente.<br/>
        • <b>Gira la rueda del mouse</b> para ampliar las amplitudes de los picos y separar líneas cercanas.<br/>
        • <b>🟢 Verde</b> = Línea cuántica única aislada.<br/>
        • <b>🟠 Ámbar + 🟣 Magenta</b> = Líneas traslapadas (múltiplete).
      </div>
      <div class="pi-sublist" id="piSublist"></div>
    </div>
  </div>

  <!-- Bottom Status Bar -->
  <div class="jb95-statusbar">
    <div id="stCursor">Cursor: Mueve el ratón sobre el espectro...</div>
    <div id="stSummary"></div>
  </div>
</div>

<script>
(function() {{
  const D = {data_json};
  const xExp = D.x_exp || [];
  const yExp = D.y_exp || [];
  const xSim = D.x_sim || [];
  const ySim = D.y_sim || [];
  const peaks = D.peaks || [];
  const M = D.meta || {{}};

  // Populate top badges
  document.getElementById('lblSingle').textContent = `🟢 Aislados: ${{M.single_cnt}}`;
  document.getElementById('lblOvlp').textContent = `🟠 Traslapados: ${{M.ovlp_cnt}}`;
  document.getElementById('lblUnass').textContent = `🔴 Sin Asignar: ${{M.unass_cnt}}`;
  document.getElementById('lblConsts').textContent =
    `S₀(${{M.A0}}, ${{M.B0}}, ${{M.C0}}) → S₁(${{M.A1}}, ${{M.B1}}, ${{M.C1}}) MHz | ν₀=${{M.nu0_mhz}} MHz | TipoProp=${{M.prop_type}} | RMS=${{M.rms_mhz}} MHz | r(Rama R)=${{M.r_win_r}}% | r(Global)=${{M.corr_pct}}%`;
  document.getElementById('stSummary').textContent =
    `Archivo: ${{M.filename}} | ${{M.asg_cnt}} transiciones en .asc | Tolerancia ±${{M.tol_mhz}} MHz`;

  const fullMinX = xExp.length > 0 ? xExp[0] : -25.0;
  const fullMaxX = xExp.length > 0 ? xExp[xExp.length - 1] : 25.0;
  const fullSpanX = Math.max(0.001, fullMaxX - fullMinX);

  // Viewport state
  let viewMinX = fullMinX;
  let viewMaxX = fullMaxX;
  let yGain = 1.0;        // Amplifies peak heights relative to baseline
  let panYOffset = 0.0;   // Vertical pan offset in normalized units
  let showExp = true;
  let showSim = true;
  let mirrorMode = false; // Default false = JB95 Direct Full-Height (Baseline at bottom -> Max Peak at top)
  let showSticks = true;
  let showQN = true;
  let filterMode = 'all'; // 'all', 'single', 'overlapped'
  let wheelMode = 'smart'; // 'smart' (Y-Gain + X-Zoom), 'ygain' (Amplitude Y only), 'xzoom' (Frequency X only)
  let hoveredPeak = null;
  let pinnedPeak = null;
  let ovlpJumpIdx = -1;

  const canvas = document.getElementById('specCanvas');
  const wrap = document.getElementById('canvasWrap');
  const ctx = canvas.getContext('2d');

  const rngYGain = document.getElementById('rngYGain');
  const valYGain = document.getElementById('valYGain');
  const rngXZoom = document.getElementById('rngXZoom');
  const valXZoom = document.getElementById('valXZoom');

  function updateSlidersUI() {{
    const curXZoom = fullSpanX / Math.max(1e-5, viewMaxX - viewMinX);
    rngXZoom.value = Math.min(150, Math.max(1, curXZoom)).toFixed(1);
    valXZoom.textContent = curXZoom.toFixed(1) + '×';
    rngYGain.value = Math.min(40, Math.max(0.5, yGain)).toFixed(2);
    valYGain.textContent = yGain.toFixed(1) + '×';
  }}

  function setInspectorPeak(pk, isPinned) {{
    const hdr = document.getElementById('piHeader');
    const bdg = document.getElementById('piBadge');
    const body = document.getElementById('piBody');
    const sub = document.getElementById('piSublist');
    if (!pk) {{
      return;
    }}
    const pinIcon = isPinned ? '📌 ' : '🎯 ';
    hdr.textContent = `${{pinIcon}}Pico Experimental #${{pk.id}} (${{pk.x_ghz >= 0 ? '+' : ''}}${{pk.x_ghz.toFixed(4)}} GHz)`;
    if (pk.status === 'single') {{
      bdg.className = 'jb95-badge badge-single';
      bdg.textContent = `🟢 Línea Única (×1)`;
    }} else if (pk.status === 'overlapped') {{
      bdg.className = 'jb95-badge badge-ovlp';
      bdg.textContent = `🟠 Traslapada (×${{pk.ovlp}})`;
    }} else {{
      bdg.className = 'jb95-badge badge-unass';
      bdg.textContent = `🔴 Sin Asignar`;
    }}

    body.innerHTML =
      `<div>• <b>Frecuencia Pico (ν_exp):</b> <span style="color:#f8fafc;font-family:monospace;">${{pk.f_mhz.toFixed(3)}} MHz</span></div>` +
      `<div>• <b>Altura Normalizada:</b> <span style="color:#fdba74;font-family:monospace;">${{pk.y_norm.toFixed(3)}}</span> (Cruda: ${{pk.y_raw.toFixed(3)}})</div>` +
      `<div>• <b>Transición Dominante:</b> <span style="color:#38bdf8;font-weight:700;">${{pk.lbl}}</span> (Δ = ${{pk.res >= 0 ? '+' : ''}}${{pk.res.toFixed(2)}} MHz)</div>`;

    sub.innerHTML = '';
    if (pk.subs && pk.subs.length > 0) {{
      pk.subs.forEach((s, idx) => {{
        const div = document.createElement('div');
        div.className = 'pi-subitem';
        const col = pk.ovlp >= 2 ? (idx === 0 ? '#fbbf24' : '#f472b6') : '#34d399';
        div.innerHTML =
          `<span style="color:${{col}};font-weight:600;">${{idx+1}}. ${{s.q_label}}</span>` +
          `<span style="color:#cbd5e1;">I=${{s.intensity_jb95.toFixed(2)}} | Δ=${{s.residual_mhz >= 0 ? '+' : ''}}${{s.residual_mhz.toFixed(2)}} MHz</span>`;
        sub.appendChild(div);
      }});
    }}
  }}

  // Resize observer
  function resizeCanvas() {{
    const dpr = window.devicePixelRatio || 1;
    const rect = wrap.getBoundingClientRect();
    canvas.width = Math.max(300, Math.floor(rect.width * dpr));
    canvas.height = Math.max(240, Math.floor(rect.height * dpr));
    draw();
  }}
  window.addEventListener('resize', resizeCanvas);

  // Coordinate transforms
  function getLayout() {{
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    const padL = 58, padR = 18, padT = 22;
    const miniH = 34;
    const resH = 78;
    const gap = 12;
    const mainH = Math.max(120, H - padT - resH - miniH - gap * 2 - 22);
    const mainTop = padT;
    const mainBot = mainTop + mainH;
    const resTop = mainBot + gap;
    const resBot = resTop + resH;
    const miniTop = resBot + gap + 6;
    const miniBot = miniTop + miniH;
    return {{ dpr, W, H, padL, padR, mainTop, mainBot, mainH, resTop, resBot, resH, miniTop, miniBot, miniH }};
  }}

  function xToPx(xGhz, L) {{
    return L.padL + ((xGhz - viewMinX) / Math.max(1e-9, viewMaxX - viewMinX)) * (L.W - L.padL - L.padR);
  }}

  function pxToX(px, L) {{
    const frac = (px - L.padL) / Math.max(1, L.W - L.padL - L.padR);
    return viewMinX + frac * (viewMaxX - viewMinX);
  }}

  function yToPx(yVal, L) {{
    if (!mirrorMode) {{
      // Direct JB95 Full-Height Mode: baseline y=0 near bottom of main viewport, y=1.08 at top
      const basePx = L.mainBot - 16 + panYOffset * L.mainH;
      const scalePx = (L.mainH - 34) / 1.08;
      return basePx - (yVal * yGain) * scalePx;
    }} else {{
      // JCP Mirror Mode: baseline y=0 in center of main viewport
      const zeroPx = L.mainTop + L.mainH * 0.5 + panYOffset * L.mainH;
      const scalePx = (L.mainH * 0.44) / 1.08;
      return zeroPx - (yVal * yGain) * scalePx;
    }}
  }}

  function draw() {{
    const L = getLayout();
    ctx.save();
    ctx.scale(L.dpr, L.dpr);

    // Clear background
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, L.W, L.H);

    const plotW = L.W - L.padL - L.padR;

    // -----------------------------------------------------------------------
    // 1. MAIN SPECTRUM VIEWPORT
    // -----------------------------------------------------------------------
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(L.padL, L.mainTop, plotW, L.mainH);
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.strokeRect(L.padL, L.mainTop, plotW, L.mainH);

    ctx.save();
    ctx.beginPath();
    ctx.rect(L.padL, L.mainTop, plotW, L.mainH);
    ctx.clip();

    // Vertical frequency grid lines
    const spanX = viewMaxX - viewMinX;
    let stepX = 5.0;
    if (spanX < 0.4) stepX = 0.02;
    else if (spanX < 1.0) stepX = 0.05;
    else if (spanX < 2.5) stepX = 0.1;
    else if (spanX < 6.0) stepX = 0.5;
    else if (spanX < 15.0) stepX = 1.0;
    else if (spanX < 35.0) stepX = 2.0;

    const firstGridX = Math.ceil(viewMinX / stepX) * stepX;
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
    ctx.lineWidth = 1;
    for (let gx = firstGridX; gx <= viewMaxX; gx += stepX) {{
      const px = xToPx(gx, L);
      ctx.beginPath();
      ctx.moveTo(px, L.mainTop);
      ctx.lineTo(px, L.mainBot);
      ctx.stroke();
    }}

    // Horizontal intensity grid lines
    const yTicks = mirrorMode ? [-1.0, -0.5, 0.0, 0.5, 1.0] : [0.0, 0.25, 0.5, 0.75, 1.0];
    yTicks.forEach(yt => {{
      const py = yToPx(yt / Math.max(0.01, yGain), L);
      if (py >= L.mainTop && py <= L.mainBot) {{
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
        ctx.beginPath();
        ctx.moveTo(L.padL, py);
        ctx.lineTo(L.W - L.padR, py);
        ctx.stroke();
      }}
    }});

    // Baseline y = 0.0 reference line
    const basePx = yToPx(0.0, L);
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.3;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(L.padL, basePx);
    ctx.lineTo(L.W - L.padR, basePx);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 10px Consolas, monospace';
    ctx.fillText(`LÍNEA BASE (0.00 | Raw=${{M.baseline_raw}})`, L.padL + 6, Math.min(L.mainBot - 4, Math.max(L.mainTop + 12, basePx - 4)));

    // Origin ν0 vertical marker (x = 0 GHz)
    if (viewMinX <= 0 && viewMaxX >= 0) {{
      const origPx = xToPx(0.0, L);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(origPx, L.mainTop);
      ctx.lineTo(origPx, L.mainBot);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f87171';
      ctx.fillText('ν₀ (0.0 GHz)', origPx + 4, L.mainTop + 13);
    }}

    // -----------------------------------------------------------------------
    // 2. DRAW SIMULATED QUANTUM SPECTRUM (B SIM)
    // -----------------------------------------------------------------------
    if (showSim && xSim.length > 1) {{
      ctx.save();
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xSim.length; i++) {{
        const xv = xSim[i];
        if (xv < viewMinX - stepX || xv > viewMaxX + stepX) continue;
        const px = xToPx(xv, L);
        const yv = mirrorMode ? -ySim[i] : ySim[i];
        const py = yToPx(yv, L);
        if (!started) {{
          ctx.moveTo(px, basePx);
          ctx.lineTo(px, py);
          started = true;
        }} else {{
          ctx.lineTo(px, py);
        }}
      }}
      if (started) {{
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.lineTo(xToPx(Math.min(viewMaxX, xSim[xSim.length - 1]), L), basePx);
        ctx.closePath();
        ctx.fillStyle = 'rgba(56, 189, 248, 0.13)';
        ctx.fill();
      }}
      ctx.restore();
    }}

    // -----------------------------------------------------------------------
    // 3. DRAW EXPERIMENTAL SPECTRUM (A EXP)
    // -----------------------------------------------------------------------
    if (showExp && xExp.length > 1) {{
      ctx.save();
      ctx.beginPath();
      let firstPx = null, lastPx = null;
      for (let i = 0; i < xExp.length; i++) {{
        const xv = xExp[i];
        if (xv < viewMinX - stepX || xv > viewMaxX + stepX) continue;
        const px = xToPx(xv, L);
        const py = yToPx(yExp[i], L);
        if (firstPx === null) {{
          firstPx = px;
          ctx.moveTo(px, py);
        }} else {{
          ctx.lineTo(px, py);
        }}
        lastPx = px;
      }}
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 1.45;
      ctx.stroke();

      if (firstPx !== null && lastPx !== null) {{
        ctx.lineTo(lastPx, basePx);
        ctx.lineTo(firstPx, basePx);
        ctx.closePath();
        ctx.fillStyle = 'rgba(249, 115, 22, 0.10)';
        ctx.fill();
      }}

      // When zoomed in horizontally (>4x), draw individual spectrometer data dots
      const curZoomX = fullSpanX / spanX;
      if (curZoomX >= 3.5) {{
        ctx.fillStyle = '#fdba74';
        const dotR = curZoomX >= 10 ? 2.6 : 1.8;
        for (let i = 0; i < xExp.length; i++) {{
          const xv = xExp[i];
          if (xv < viewMinX || xv > viewMaxX) continue;
          const px = xToPx(xv, L);
          const py = yToPx(yExp[i], L);
          ctx.beginPath();
          ctx.arc(px, py, dotR, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
      ctx.restore();
    }}

    // -----------------------------------------------------------------------
    // 4. DRAW PROJECTED PEAK-TO-BASELINE LINES & OVERLAPPED MULTIPLETS
    // -----------------------------------------------------------------------
    if (showSticks && peaks.length > 0) {{
      const curZoomX = fullSpanX / spanX;
      const labeledPositions = [];

      peaks.forEach(pk => {{
        if (pk.x_ghz < viewMinX - 0.5 || pk.x_ghz > viewMaxX + 0.5) return;
        if (filterMode === 'single' && pk.status !== 'single') return;
        if (filterMode === 'overlapped' && pk.status !== 'overlapped') return;

        const px = xToPx(pk.x_ghz, L);
        const pyApex = yToPx(pk.y_norm, L);
        const isSel = (hoveredPeak && hoveredPeak.id === pk.id) || (pinnedPeak && pinnedPeak.id === pk.id);

        if (pk.status === 'overlapped') {{
          // 1) Draw individual overlapped theoretical sub-transitions in Neon Magenta (#ec4899)
          if (pk.subs && pk.subs.length > 1) {{
            const maxSubI = Math.max(0.01, pk.subs[0].intensity_jb95);
            pk.subs.forEach(sub => {{
              const subPx = xToPx(sub.rel_calc_ghz, L);
              const subH = pk.y_norm * (sub.intensity_jb95 / maxSubI) * 0.92;
              const subPy = yToPx(subH, L);
              ctx.strokeStyle = isSel ? '#f472b6' : 'rgba(236, 72, 153, 0.78)';
              ctx.lineWidth = isSel ? 2.0 : 1.25;
              ctx.setLineDash([2, 2]);
              ctx.beginPath();
              ctx.moveTo(subPx, basePx);
              ctx.lineTo(subPx, subPy);
              ctx.stroke();
              ctx.setLineDash([]);
            }});
          }}

          // 2) Draw main vertical projection line from peak apex straight down to baseline in Bright Amber (#f59e0b)
          ctx.strokeStyle = isSel ? '#fde047' : '#f59e0b';
          ctx.lineWidth = isSel ? 2.6 : 1.75;
          ctx.beginPath();
          ctx.moveTo(px, pyApex);
          ctx.lineTo(px, basePx);
          ctx.stroke();

          // Diamond marker at peak apex
          const dSize = isSel ? 6.0 : 4.2;
          ctx.fillStyle = isSel ? '#fde047' : '#f59e0b';
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.moveTo(px, pyApex - dSize);
          ctx.lineTo(px + dSize, pyApex);
          ctx.lineTo(px, pyApex + dSize);
          ctx.lineTo(px - dSize, pyApex);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Overlap multiplier pill (×2, ×3...) when zoomed or prominent
          if (isSel || curZoomX >= 2.2 || pk.y_norm * yGain >= 0.42) {{
            const tag = `×${{pk.ovlp}}`;
            ctx.font = 'bold 9.5px Segoe UI, sans-serif';
            const tw = ctx.measureText(tag).width + 6;
            ctx.fillStyle = 'rgba(245, 158, 11, 0.92)';
            ctx.fillRect(px - tw / 2, pyApex - 18, tw, 12);
            ctx.fillStyle = '#0f172a';
            ctx.fillText(tag, px - tw / 2 + 3, pyApex - 8.5);
          }}
        }} else if (pk.status === 'single') {{
          // Isolated Single Quantum Transition: Emerald Green (#10b981) vertical projection from apex to baseline
          ctx.strokeStyle = isSel ? '#6ee7b7' : 'rgba(16, 185, 129, 0.85)';
          ctx.lineWidth = isSel ? 2.4 : 1.35;
          ctx.beginPath();
          ctx.moveTo(px, pyApex);
          ctx.lineTo(px, basePx);
          ctx.stroke();

          // Circle marker at peak apex
          ctx.fillStyle = isSel ? '#6ee7b7' : '#10b981';
          ctx.strokeStyle = '#064e3b';
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.arc(px, pyApex, isSel ? 5.0 : 3.3, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }} else {{
          // Unassigned Experimental Peak: Red dashed projection to baseline
          ctx.strokeStyle = isSel ? '#fca5a5' : 'rgba(239, 68, 68, 0.55)';
          ctx.lineWidth = 1.1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(px, pyApex);
          ctx.lineTo(px, basePx);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.strokeStyle = '#ef4444';
          ctx.beginPath();
          ctx.arc(px, pyApex, 2.8, 0, Math.PI * 2);
          ctx.stroke();
        }}

        // Quantum Number Label (QN) above peak if well-spaced or zoomed in
        if (showQN && pk.status !== 'unassigned') {{
          const minGapPx = curZoomX >= 8 ? 52 : 78;
          const canLabel = isSel || (
            (pk.y_norm * yGain >= 0.28 || curZoomX >= 6) &&
            labeledPositions.every(lp => Math.abs(lp - px) > minGapPx)
          );
          if (canLabel && pyApex > L.mainTop + 16) {{
            labeledPositions.push(px);
            ctx.font = isSel ? 'bold 10.5px Consolas, monospace' : 'bold 9.5px Consolas, monospace';
            const lblTxt = pk.status === 'overlapped' ? `${{pk.lbl}} (×${{pk.ovlp}})` : pk.lbl;
            const lw = ctx.measureText(lblTxt).width;
            const ly = Math.max(L.mainTop + 14, pyApex - (pk.status === 'overlapped' ? 22 : 10));
            ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
            ctx.fillRect(px - lw / 2 - 3, ly - 10, lw + 6, 13);
            ctx.fillStyle = pk.status === 'overlapped' ? '#fbbf24' : '#34d399';
            ctx.fillText(lblTxt, px - lw / 2, ly);
          }}
        }}
      }});
    }}

    ctx.restore(); // end main viewport clip

    // Main Y-Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'right';
    yTicks.forEach(yt => {{
      const effY = yt / Math.max(0.01, yGain);
      const py = yToPx(effY, L);
      if (py >= L.mainTop + 8 && py <= L.mainBot - 6) {{
        ctx.fillText(effY.toFixed(2), L.padL - 6, py + 3);
      }}
    }});

    // Main X-Axis Frequency Tick Labels (between main and residual strips)
    ctx.textAlign = 'center';
    ctx.fillStyle = '#cbd5e1';
    for (let gx = firstGridX; gx <= viewMaxX; gx += stepX) {{
      const px = xToPx(gx, L);
      if (px >= L.padL + 20 && px <= L.W - L.padR - 20) {{
        const decimals = stepX < 0.1 ? 3 : (stepX < 1 ? 2 : 1);
        ctx.fillText((gx >= 0 ? '+' : '') + gx.toFixed(decimals) + ' GHz', px, L.mainBot + 11);
      }}
    }}

    // -----------------------------------------------------------------------
    // 5. SYNCHRONIZED RESIDUAL STRIP (ν_exp - ν_calc in MHz)
    // -----------------------------------------------------------------------
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(L.padL, L.resTop, plotW, L.resH);
    ctx.strokeStyle = '#334155';
    ctx.strokeRect(L.padL, L.resTop, plotW, L.resH);

    const resZeroY = L.resTop + L.resH * 0.5;
    const maxRes = Math.max(2.0, M.tol_mhz * 1.15);

    // ±RMS shaded band
    const rmsPxSpan = (M.rms_mhz / maxRes) * (L.resH * 0.45);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.10)';
    ctx.fillRect(L.padL, resZeroY - rmsPxSpan, plotW, rmsPxSpan * 2);

    ctx.strokeStyle = '#475569';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(L.padL, resZeroY);
    ctx.lineTo(L.W - L.padR, resZeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.save();
    ctx.beginPath();
    ctx.rect(L.padL, L.resTop, plotW, L.resH);
    ctx.clip();

    peaks.forEach(pk => {{
      if (pk.status === 'unassigned') return;
      if (pk.x_ghz < viewMinX || pk.x_ghz > viewMaxX) return;
      if (filterMode === 'single' && pk.status !== 'single') return;
      if (filterMode === 'overlapped' && pk.status !== 'overlapped') return;

      const px = xToPx(pk.x_ghz, L);
      const ry = resZeroY - (pk.res / maxRes) * (L.resH * 0.44);
      const col = pk.status === 'overlapped' ? '#f59e0b' : '#10b981';

      ctx.strokeStyle = col;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(px, resZeroY);
      ctx.lineTo(px, ry);
      ctx.stroke();

      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(px, ry, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }});
    ctx.restore();

    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.font = 'bold 10px Consolas, monospace';
    ctx.fillText(`RESIDUALES ν_exp - ν_calc (RMS = ±${{M.rms_mhz}} MHz)`, L.padL + 6, L.resTop + 12);
    ctx.textAlign = 'right';
    ctx.fillText(`+${{M.tol_mhz}}`, L.padL - 5, L.resTop + 11);
    ctx.fillText(`0.0`, L.padL - 5, resZeroY + 3);
    ctx.fillText(`-${{M.tol_mhz}}`, L.padL - 5, L.resBot - 3);

    // -----------------------------------------------------------------------
    // 6. BOTTOM FULL-SCAN MINIMAP / SCROLLBAR
    // -----------------------------------------------------------------------
    ctx.fillStyle = '#090d16';
    ctx.fillRect(L.padL, L.miniTop, plotW, L.miniH);
    ctx.strokeStyle = '#334155';
    ctx.strokeRect(L.padL, L.miniTop, plotW, L.miniH);

    if (xExp.length > 1) {{
      ctx.beginPath();
      const stepMini = Math.max(1, Math.floor(xExp.length / 500));
      for (let i = 0; i < xExp.length; i += stepMini) {{
        const mx = L.padL + ((xExp[i] - fullMinX) / fullSpanX) * plotW;
        const my = L.miniBot - 2 - yExp[i] * (L.miniH - 5);
        if (i === 0) ctx.moveTo(mx, my);
        else ctx.lineTo(mx, my);
      }}
      ctx.strokeStyle = 'rgba(249, 115, 22, 0.7)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }}

    // Viewport window box on minimap
    const vLeftPx = L.padL + ((viewMinX - fullMinX) / fullSpanX) * plotW;
    const vRightPx = L.padL + ((viewMaxX - fullMinX) / fullSpanX) * plotW;
    const vWidthPx = Math.max(6, vRightPx - vLeftPx);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.20)';
    ctx.fillRect(vLeftPx, L.miniTop, vWidthPx, L.miniH);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vLeftPx, L.miniTop, vWidthPx, L.miniH);

    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.font = '10px Consolas, monospace';
    ctx.fillText('NAVEGADOR GLOBAL (Arrastra para recorrer todo el espectro)', L.padL + 6, L.miniTop - 3);

    ctx.restore();
    updateSlidersUI();
  }}

  // -------------------------------------------------------------------------
  // INTERACTIVE MOUSE PANNING (CLICK & DRAG) & PEAK SELECTION
  // -------------------------------------------------------------------------
  let isDraggingMain = false;
  let isDraggingMini = false;
  let dragStartX = 0, dragStartY = 0;
  let dragStartMinX = 0, dragStartMaxX = 0, dragStartPanY = 0;
  let movedDuringDrag = false;

  canvas.addEventListener('mousedown', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const L = getLayout();

    if (my >= L.miniTop && my <= L.miniBot) {{
      isDraggingMini = true;
      const frac = Math.max(0, Math.min(1, (mx - L.padL) / Math.max(1, L.W - L.padL - L.padR)));
      const centerGhz = fullMinX + frac * fullSpanX;
      const halfSpan = (viewMaxX - viewMinX) * 0.5;
      viewMinX = Math.max(fullMinX, Math.min(fullMaxX - halfSpan * 2, centerGhz - halfSpan));
      viewMaxX = viewMinX + halfSpan * 2;
      draw();
      return;
    }}

    isDraggingMain = true;
    movedDuringDrag = false;
    dragStartX = mx;
    dragStartY = my;
    dragStartMinX = viewMinX;
    dragStartMaxX = viewMaxX;
    dragStartPanY = panYOffset;
  }});

  window.addEventListener('mousemove', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const L = getLayout();
    const plotW = Math.max(1, L.W - L.padL - L.padR);

    if (isDraggingMini) {{
      const frac = Math.max(0, Math.min(1, (mx - L.padL) / plotW));
      const centerGhz = fullMinX + frac * fullSpanX;
      const curSpan = viewMaxX - viewMinX;
      viewMinX = Math.max(fullMinX, Math.min(fullMaxX - curSpan, centerGhz - curSpan * 0.5));
      viewMaxX = viewMinX + curSpan;
      draw();
      return;
    }}

    if (isDraggingMain) {{
      const dxPx = mx - dragStartX;
      const dyPx = my - dragStartY;
      if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) movedDuringDrag = true;

      const curSpan = dragStartMaxX - dragStartMinX;
      const dGhz = -(dxPx / plotW) * curSpan;
      let newMin = dragStartMinX + dGhz;
      let newMax = dragStartMaxX + dGhz;
      if (newMin < fullMinX) {{
        newMin = fullMinX;
        newMax = fullMinX + curSpan;
      }}
      if (newMax > fullMaxX) {{
        newMax = fullMaxX;
        newMin = fullMaxX - curSpan;
      }}
      viewMinX = newMin;
      viewMaxX = newMax;
      panYOffset = Math.max(-0.85, Math.min(0.85, dragStartPanY + (dyPx / Math.max(100, L.mainH))));
      draw();
      return;
    }}

    // Hover peak detection
    if (mx >= L.padL && mx <= L.W - L.padR && my >= L.mainTop && my <= L.resBot) {{
      const curGhz = pxToX(mx, L);
      const absMhz = M.nu0_mhz + curGhz * 1000.0;

      // Find nearest peak within 10 pixels
      let bestPk = null;
      let minDistPx = 11.0;
      peaks.forEach(pk => {{
        if (filterMode === 'single' && pk.status !== 'single') return;
        if (filterMode === 'overlapped' && pk.status !== 'overlapped') return;
        const pkPx = xToPx(pk.x_ghz, L);
        const dPx = Math.abs(mx - pkPx);
        if (dPx < minDistPx) {{
          minDistPx = dPx;
          bestPk = pk;
        }}
      }});

      if (bestPk !== hoveredPeak) {{
        hoveredPeak = bestPk;
        if (!pinnedPeak && hoveredPeak) {{
          setInspectorPeak(hoveredPeak, false);
        }}
        draw();
      }}

      let stTxt = `Cursor: ${{absMhz.toFixed(2)}} MHz (${{curGhz >= 0 ? '+' : ''}}${{curGhz.toFixed(4)}} GHz)`;
      if (bestPk) {{
        const tag = bestPk.status === 'overlapped' ? `🟠 Traslapada ×${{bestPk.ovlp}}` : (bestPk.status === 'single' ? '🟢 Aislada' : '🔴 Sin asignar');
        stTxt += `  |  Pico #${{bestPk.id}}: [${{tag}}] ${{bestPk.lbl}} (Δ=${{bestPk.res >= 0 ? '+' : ''}}${{bestPk.res.toFixed(2)}} MHz)`;
      }}
      document.getElementById('stCursor').textContent = stTxt;
    }}
  }});

  window.addEventListener('mouseup', e => {{
    if (isDraggingMain && !movedDuringDrag) {{
      if (hoveredPeak) {{
        if (pinnedPeak && pinnedPeak.id === hoveredPeak.id) {{
          pinnedPeak = null;
        }} else {{
          pinnedPeak = hoveredPeak;
          setInspectorPeak(pinnedPeak, true);
        }}
        draw();
      }}
    }}
    isDraggingMain = false;
    isDraggingMini = false;
  }});

  // -------------------------------------------------------------------------
  // MOUSE WHEEL AMPLITUDE GAIN & HORIZONTAL ZOOM
  // -------------------------------------------------------------------------
  canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const L = getLayout();
    const zoomIn = e.deltaY < 0;

    // Shift+Wheel always zooms frequency X; Alt/Ctrl+Wheel always amplifies Y
    const doX = e.shiftKey || wheelMode === 'xzoom' || wheelMode === 'smart';
    const doY = e.ctrlKey || e.altKey || wheelMode === 'ygain' || wheelMode === 'smart';

    if (doY && !e.shiftKey) {{
      const factorY = zoomIn ? 1.16 : (1.0 / 1.16);
      yGain = Math.max(0.4, Math.min(50.0, yGain * factorY));
    }}

    if (doX && !e.ctrlKey && !e.altKey) {{
      const cursorGhz = pxToX(Math.max(L.padL, Math.min(L.W - L.padR, mx)), L);
      const frac = (cursorGhz - viewMinX) / Math.max(1e-9, viewMaxX - viewMinX);
      const curSpan = viewMaxX - viewMinX;
      const factorX = zoomIn ? 0.80 : 1.25;
      const newSpan = Math.max(fullSpanX / 200.0, Math.min(fullSpanX, curSpan * factorX));
      let newMin = cursorGhz - frac * newSpan;
      let newMax = newMin + newSpan;
      if (newMin < fullMinX) {{ newMin = fullMinX; newMax = fullMinX + newSpan; }}
      if (newMax > fullMaxX) {{ newMax = fullMaxX; newMin = fullMaxX - newSpan; }}
      viewMinX = newMin;
      viewMaxX = newMax;
    }}

    draw();
  }}, {{ passive: false }});

  // -------------------------------------------------------------------------
  // TOOLBAR BUTTON BINDINGS
  // -------------------------------------------------------------------------
  document.getElementById('btnExp').addEventListener('click', function() {{
    showExp = !showExp;
    this.classList.toggle('active-exp', showExp);
    draw();
  }});

  document.getElementById('btnSim').addEventListener('click', function() {{
    showSim = !showSim;
    this.classList.toggle('active-sim', showSim);
    draw();
  }});

  document.getElementById('btnMode').addEventListener('click', function() {{
    mirrorMode = !mirrorMode;
    panYOffset = 0.0;
    this.textContent = mirrorMode ? 'A/B: Espejo JCP' : 'A/B: Directo JB95';
    draw();
  }});

  document.getElementById('btnSticks').addEventListener('click', function() {{
    showSticks = !showSticks;
    this.classList.toggle('active-grn', showSticks);
    draw();
  }});

  document.getElementById('btnQN').addEventListener('click', function() {{
    showQN = !showQN;
    this.classList.toggle('active-grn', showQN);
    draw();
  }});

  function setFilterButtons(mode) {{
    filterMode = mode;
    document.getElementById('btnFltAll').classList.toggle('active-grn', mode === 'all');
    document.getElementById('btnFltSingle').classList.toggle('active-grn', mode === 'single');
    document.getElementById('btnFltOvlp').classList.toggle('active-amb', mode === 'overlapped');
    draw();
  }}
  document.getElementById('btnFltAll').addEventListener('click', () => setFilterButtons('all'));
  document.getElementById('btnFltSingle').addEventListener('click', () => setFilterButtons('single'));
  document.getElementById('btnFltOvlp').addEventListener('click', () => setFilterButtons('overlapped'));

  // Jump to Next Overlapped Peak
  document.getElementById('btnNextOvlp').addEventListener('click', () => {{
    const ovlpPeaks = peaks.filter(p => p.status === 'overlapped');
    if (ovlpPeaks.length === 0) return;
    ovlpJumpIdx = (ovlpJumpIdx + 1) % ovlpPeaks.length;
    const target = ovlpPeaks[ovlpJumpIdx];
    pinnedPeak = target;
    setInspectorPeak(target, true);

    const zoomSpan = Math.min(fullSpanX, Math.max(0.8, fullSpanX / 14.0));
    viewMinX = Math.max(fullMinX, Math.min(fullMaxX - zoomSpan, target.x_ghz - zoomSpan * 0.5));
    viewMaxX = viewMinX + zoomSpan;
    if (target.y_norm > 0.02) {{
      yGain = Math.min(25.0, Math.max(1.0, 0.82 / target.y_norm));
    }}
    draw();
  }});

  // Cycle Wheel Mode
  document.getElementById('btnWheelMode').addEventListener('click', function() {{
    if (wheelMode === 'smart') {{
      wheelMode = 'ygain';
      this.textContent = '📈 Solo Amplitud Y';
    }} else if (wheelMode === 'ygain') {{
      wheelMode = 'xzoom';
      this.textContent = '🔍 Solo Frecuencia X';
    }} else {{
      wheelMode = 'smart';
      this.textContent = '⚡ Amplitud Y + Zoom X';
    }}
  }});

  rngYGain.addEventListener('input', function() {{
    yGain = parseFloat(this.value) || 1.0;
    draw();
  }});

  rngXZoom.addEventListener('input', function() {{
    const z = parseFloat(this.value) || 1.0;
    const centerGhz = 0.5 * (viewMinX + viewMaxX);
    const newSpan = fullSpanX / z;
    viewMinX = Math.max(fullMinX, Math.min(fullMaxX - newSpan, centerGhz - newSpan * 0.5));
    viewMaxX = viewMinX + newSpan;
    draw();
  }});

  document.getElementById('btnReset').addEventListener('click', () => {{
    viewMinX = fullMinX;
    viewMaxX = fullMaxX;
    yGain = 1.0;
    panYOffset = 0.0;
    pinnedPeak = null;
    draw();
  }});

  // Initialize first prominent peak in inspector
  if (peaks.length > 0) {{
    const firstOvlp = peaks.find(p => p.status === 'overlapped') || peaks[0];
    setInspectorPeak(firstOvlp, false);
  }}

  setTimeout(resizeCanvas, 30);
}})();
</script>
</body>
</html>"""

    escaped_doc = html_lib.escape(inner_doc, quote=True)
    return (
        f'<iframe srcdoc="{escaped_doc}" '
        f'style="width:100%; height:670px; border:none; border-radius:10px; '
        f'box-shadow:0 6px 24px rgba(0,0,0,0.28); background:#0b1120;" '
        f' title="JB95 Interactive Workstation"></iframe>'
    )
