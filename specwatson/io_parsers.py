"""
SpecWatson: Spectrometer & JB95 Experimental Data Parsers (io_parsers.py)
========================================================================
Provides native reading, baseline processing, and parameter extraction for:
  1. Raw laser spectrometer fluorescence / absorption scans (e.g. `PmtFla`, `.dat`, `.txt`, `.csv`, 2-col `.asc`)
  2. JB95 / PGOPHER rotational line-list files (`.asc` format such as `IAR.asc`:
     `BandType J' Ka' Kc' J'' Ka'' Kc'' Intensity Freq_Obs Freq_Calc [Residual]`)
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import least_squares
from .hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from .transitions import RovibronicLine, MHZ_TO_WAVENUMBER, WAVENUMBER_TO_MHZ
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


def _extract_constants_from_jb95_lines(lines: List[RovibronicLine]) -> Optional[Dict[str, float]]:
    """
    Recovers exact rotational constants (A0, B0, C0, A1, B1, C1) and band origin (nu0_mhz)
    from low-J transitions (J <= 3) in a JB95 .asc line list using exact algebraic
    combinations followed by Levenberg-Marquardt quantum eigenvalue refinement.
    """
    low_j_lines = [ln for ln in lines if ln.J_upper <= 3 and ln.J_lower <= 3]
    if len(low_j_lines) < 7:
        return None

    # Map transition tuple (J', Ka', Kc', J'', Ka'', Kc'') -> frequency_mhz
    tmap: Dict[Tuple[int, int, int, int, int, int], float] = {}
    for ln in low_j_lines:
        key = (ln.J_upper, ln.Ka_upper, ln.Kc_upper, ln.J_lower, ln.Ka_lower, ln.Kc_lower)
        tmap[key] = ln.frequency_mhz

    all_freqs = np.array([ln.frequency_mhz for ln in low_j_lines], dtype=np.float64)
    nu0_guess = float(np.median(all_freqs))
    A0_g, B0_g, C0_g = 2200.0, 800.0, 500.0
    A1_g, B1_g, C1_g = 2180.0, 790.0, 495.0

    # Exact algebraic seed for B-type low-J lines if present
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

    # Refine via exact Hamiltonian eigenvalues on J <= 3 transitions
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

    # Detect if this is a JB95 .asc line-list file (starts with A, B, or C followed by 6 integers)
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

    # Otherwise: parse 2-column continuous spectrometer scan (e.g. PmtFla, .dat, .txt, .csv)
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

    # Ensure ascending frequency order
    if x_arr[0] > x_arr[-1]:
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

    # Detect frequency units
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

    # Baseline subtraction (5th percentile) and peak normalization
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
