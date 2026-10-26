"""
SpecWatson Interactive Web GUI (Gradio)
=======================================
Cloud / Local Interactive Interface for High-Resolution Rovibronic Spectroscopy:
  1. Direct Quantum Spectrum Simulator:
     - Selectable Hamiltonian Model: Pure Rigid Rotor (JB95 standard) vs. Watson A-Reduction Centrifugal Distortion
     - Selectable Spectral Selection Rules: Pure Type a, Pure Type b, Pure Type c, or Hybrid (a + b + c) with sub-band decomposition
     - Standard JCP / JB95 Mirror View: Experimental Spectrum on Top (+Y) and Simulated Spectrum Inverted Below (-Y)
     - Native Spectrometer File Loader: Reads raw fluorescence scans (`PmtFla`, `.dat`, `.txt`, `.csv`) and JB95 line lists (`IAR.asc`, `.asc`)
     - Interactive Visibility, Opacity, X/Y Zoom & Pan, and 300/600 DPI Publication Export
  2. Automatic Quantum Line Assigner (`.asc` / JB95 Generator) & Autonomous Contour Fitter (Differential Evolution + Simplex)
  3. 3D Quantum Chemistry & Molecular Sketcher (RDKit / TorchANI GPU / GPU4PySCF -> Inertia Tensor -> A, B, C, κ, μ)

Author: Miguel Díaz (Maestría en Ciencias - DCI, Universidad de Guanajuato)
Academic Advisors & Support: Dr. Leonardo Álvarez-Valtierra & Dr. Alejandro Gil-Villegas (SNII Nivel III)
"""

import os
import time
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator, WAVENUMBER_TO_MHZ, MHZ_TO_WAVENUMBER
from specwatson.spectrum import SpectrumSimulator
from specwatson.fitting import ContourFitter
from specwatson.io_parsers import parse_spectrometer_file, auto_assign_spectrometer_lines
from specwatson.quantum_chem import (
    MOLECULE_CATALOG,
    EXPERIMENTAL_LITERATURE_DB,
    run_quantum_3d_pipeline,
    plot_molecular_pas_figure,
    render_3dmol_pas_html,
    render_jsme_sketcher_html,
)

# Paths to real experimental resources shared from Dr. Leonardo's spectrometer & JB95
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PMTFLA_PATH = os.path.join(BASE_DIR, "recursos", "PmtFla")
DEFAULT_IAR_ASC_PATH = os.path.join(BASE_DIR, "recursos", "IAR.asc")

# Selection Rule Mode choices
BAND_TYPE_CHOICES = [
    "Tipo b Puro (ΔKa = impar, ΔKc = impar | Ej. Carbazol, Fenol, H₂O)",
    "Tipo a Puro (ΔKa = par, ΔKc = impar | Ej. Fluoreno, H₂CO, CO₂)",
    "Tipo c Puro (ΔKa = impar, ΔKc = par | Dipolo fuera del plano c)",
    "Banda Híbrida (Mezcla a + b + c controlada por pesos μₐ², μ_b², μ_c²)",
]

# Hamiltonian Model choices
HAMILTONIAN_CHOICES = [
    "Rotor Rígido Puro (3 constantes: A, B, C | ΔJ = ΔJK = ΔK = δJ = δK = 0 — Método JB95 / Carbazol)",
    "Hamiltoniano de Watson Reducción A (Rotor Rígido + Distorsión Centrífuga Cuártica)",
]

# Experimental Reference Source choices
EXP_SOURCE_CHOICES = [
    "📡 Espectrómetro Real: PmtFla (Fluorescencia LIF Carbazol - Dr. Leonardo)",
    "📄 Ajuste JB95 Real: IAR.asc (2,705 líneas Tipo B Carbazol - Dr. Leonardo)",
    "📚 Referencia de Literatura del Sistema Molecular Seleccionado",
    "📤 Archivo Propio Subido (.asc, PmtFla, .dat, .txt, .csv)",
]

# Pre-configured molecular presets for Tab 1 (Aromatics + Simple HITRAN/JPL Standards)
PRESETS = {
    "Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)": {
        "A0": 2253.2409, "B0": 594.2103, "C0": 470.3670,
        "A1": 2250.6104, "B1": 586.3482, "C1": 465.2923,
        "origin_mhz": 923604613.0295,
        "Trot": 2.6, "fwhm": 45.0,
        "mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[0],       # Tipo b Puro
        "ham_mode": HAMILTONIAN_CHOICES[0],      # Rotor Rígido Puro
        "exp_source": EXP_SOURCE_CHOICES[0],     # PmtFla real
        "Jmax": 22,
        "ref": "Álvarez-Valtierra & Pratt, J. Chem. Phys. 125, 244309 (2006) | Datos Espectrómetro PmtFla / IAR.asc"
    },
    "trans-Resveratrol (Polifenol - Confórmero I Plano)": {
        "A0": 1684.50, "B0": 218.40, "C0": 193.35,
        "A1": 1652.10, "B1": 216.85, "C1": 191.80,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.0, "fwhm": 18.0,
        "mu_a": 0.82, "mu_b": 0.18, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[3],       # Híbrida a/b
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 14,
        "ref": "Estudio DFT/Watson Confórmero I Planar (DCI-UG)"
    },
    "trans-Resveratrol (Polifenol - Confórmero II Torsional)": {
        "A0": 1592.30, "B0": 229.15, "C0": 204.80,
        "A1": 1564.80, "B1": 227.40, "C1": 203.10,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.0, "fwhm": 18.0,
        "mu_a": 0.65, "mu_b": 0.25, "mu_c": 0.10,
        "band_mode": BAND_TYPE_CHOICES[3],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 14,
        "ref": "Estudio DFT/Watson Confórmero II No-Planar (DCI-UG)"
    },
    "Indol (Cromóforo del Triptófano | Caminati & Pratt)": {
        "A0": 3877.80, "B0": 1636.00, "C0": 1150.90,
        "A1": 3743.00, "B1": 1616.00, "C1": 1129.00,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.5, "fwhm": 22.0,
        "mu_a": 0.62, "mu_b": 0.38, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[3],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 12,
        "ref": "Berden, Meerts & Pratt, Chem. Phys. Lett. (1994)"
    },
    "1-Naftol (Banda Híbrida a/b | Johnson & Pratt 1990)": {
        "A0": 1948.50, "B0": 1132.40, "C0": 716.20,
        "A1": 1912.10, "B1": 1118.80, "C1": 706.50,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 4.0, "fwhm": 22.0,
        "mu_a": 0.60, "mu_b": 0.40, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[3],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 12,
        "ref": "Johnson & Pratt, J. Chem. Phys. 93 (1990)"
    },
    "Fenol (Prototipo Aromático Tipo b | Schmitt & Pratt)": {
        "A0": 5650.52, "B0": 2619.20, "C0": 1789.84,
        "A1": 5313.40, "B1": 2581.50, "C1": 1737.60,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.0, "fwhm": 25.0,
        "mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[0],       # Tipo b Puro
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 12,
        "ref": "Schmitt, Ratzer & Pratt, J. Chem. Phys. (1998)"
    },
    "Benzonitrilo (Estándar Dipolar Tipo b | Wohlfart / JPL)": {
        "A0": 5655.26, "B0": 1546.88, "C0": 1214.40,
        "A1": 5456.00, "B1": 1516.00, "C1": 1186.00,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.0, "fwhm": 20.0,
        "mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[0],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 12,
        "ref": "Wohlfart et al., J. Mol. Spectrosc. 247 (2008)"
    },
    "Anisol (Metoxibenceno | Álvarez-Valtierra & Pratt)": {
        "A0": 5028.85, "B0": 1569.36, "C0": 1207.72,
        "A1": 4845.10, "B1": 1548.20, "C1": 1186.40,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 3.5, "fwhm": 22.0,
        "mu_a": 0.25, "mu_b": 0.75, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[3],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 12,
        "ref": "Álvarez-Valtierra & Pratt (Supersonic Jet LIF)"
    },
    "Dióxido de Azufre (SO₂ - Estándar Triatómico Tipo b | JPL / NIST)": {
        "A0": 60778.55, "B0": 10318.07, "C0": 8799.70,
        "A1": 59850.00, "B1": 10265.00, "C1": 8750.00,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 6.0, "fwhm": 45.0,
        "mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[0],
        "ham_mode": HAMILTONIAN_CHOICES[1],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 8,
        "ref": "Catálogo JPL / NIST (Lovas 1985)"
    },
    "Formaldehído (H₂CO - Cuasi-Simétrico Tipo a | HITRAN / JPL)": {
        "A0": 281970.56, "B0": 38836.05, "C0": 34002.20,
        "A1": 262600.00, "B1": 33760.00, "C1": 30450.00,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 8.0, "fwhm": 50.0,
        "mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[1],       # Tipo a Puro
        "ham_mode": HAMILTONIAN_CHOICES[1],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 7,
        "ref": "HITRAN / JPL (Bocquet et al. 1996)"
    },
    "Agua (H₂O - Rotor Asimétrico Ligero Tipo b | HITRAN / JPL)": {
        "A0": 835840.29, "B0": 435347.85, "C0": 278138.70,
        "A1": 820500.00, "B1": 430100.00, "C1": 274500.00,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 10.0, "fwhm": 50.0,
        "mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[0],
        "ham_mode": HAMILTONIAN_CHOICES[1],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 5,
        "ref": "HITRAN2020 / JPL (Matsushima et al.)"
    },
    "Dióxido de Carbono (CO₂ - Rotor Lineal D∞h Tipo a | HITRAN)": {
        "A0": 58492.35, "B0": 11698.47, "C0": 11698.47,
        "A1": 58030.50, "B1": 11606.10, "C1": 11606.10,
        "origin_mhz": 30000.0 * WAVENUMBER_TO_MHZ,
        "Trot": 8.0, "fwhm": 40.0,
        "mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0,
        "band_mode": BAND_TYPE_CHOICES[1],
        "ham_mode": HAMILTONIAN_CHOICES[0],
        "exp_source": EXP_SOURCE_CHOICES[2],
        "Jmax": 10,
        "ref": "HITRAN2020 (Gordon et al. 2022) — Ramas P y R Lineales (2B = 23.4 GHz)"
    },
}


def _resolve_selection_criterion(band_mode_str: str, mu_a: float, mu_b: float, mu_c: float):
    """Maps UI band_mode selection to ('a' | 'b' | 'c' | 'hybrid') and normalized dipole weights."""
    bm = str(band_mode_str)
    if "Tipo a Puro" in bm:
        return "a", (1.0, 0.0, 0.0), "Tipo a Puro (ΔKa = 0, ±2 [par], ΔKc = ±1, ±3 [impar] | Dipolo en Eje a)"
    if "Tipo b Puro" in bm:
        return "b", (0.0, 1.0, 0.0), "Tipo b Puro (ΔKa = ±1, ±3 [impar], ΔKc = ±1, ±3 [impar] | Dipolo en Eje b)"
    if "Tipo c Puro" in bm:
        return "c", (0.0, 0.0, 1.0), "Tipo c Puro (ΔKa = ±1, ±3 [impar], ΔKc = 0, ±2 [par] | Dipolo fuera del plano Eje c)"

    s = float(mu_a) + float(mu_b) + float(mu_c)
    if s <= 1e-6:
        dip = (0.0, 1.0, 0.0)
    else:
        dip = (float(mu_a) / s, float(mu_b) / s, float(mu_c) / s)
    return "hybrid", dip, f"Banda Híbrida (Mezcla a+b+c: μₐ²={dip[0]:.2f}, μ_b²={dip[1]:.2f}, μ_c²={dip[2]:.2f})"


def on_band_mode_change(band_mode_str):
    """Automatically sets dipole sliders when a Pure Band Type (a, b, or c) is chosen."""
    bm = str(band_mode_str)
    if "Tipo a Puro" in bm:
        return 1.0, 0.0, 0.0
    if "Tipo b Puro" in bm:
        return 0.0, 1.0, 0.0
    if "Tipo c Puro" in bm:
        return 0.0, 0.0, 1.0
    return 0.60, 0.40, 0.0


def simulate_rovibronic_spectrum(
    A0, B0, C0,
    A1, B1, C1,
    T_rot, fwhm_mhz,
    mu_a, mu_b, mu_c,
    J_max,
    preset_name="Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)",
    band_type_mode=BAND_TYPE_CHOICES[0],
    hamiltonian_model=HAMILTONIAN_CHOICES[0],
    dJ_khz=0.25, dJK_khz=-0.60, dK_khz=1.20, delJ_khz=0.04, delK_khz=0.30,
    exp_source_mode=EXP_SOURCE_CHOICES[0],
    uploaded_exp_file=None,
    show_subbands=False,
    show_sim=True,
    overlay_exp=True,
    mirror_exp=True,
    alpha_sim=0.85,
    alpha_exp=0.85,
    offset_exp=0.0,
    zoom_x_factor=1.0,
    pan_x_pct=0.0,
    zoom_y_factor=1.0,
    pan_y_offset=0.0,
    render_dpi=130
):
    """
    Callback for Tab 1: Direct Quantum Spectrum Simulation + Real Spectrometer Overlay (`PmtFla` / `IAR.asc`).
    Follows standard JCP / JB95 Mirror Convention:
      - Experimental Spectrum is plotted on TOP (+Y, positive).
      - Simulated Spectrum is plotted inverted on BOTTOM (-Y, negative) when mirror_exp is enabled.
    """
    t0 = time.perf_counter()
    plt.close("all")

    try:
        A0, B0, C0 = float(A0), float(B0), float(C0)
        A1, B1, C1 = float(A1), float(B1), float(C1)
        T_rot = max(0.2, float(T_rot))
        fwhm_mhz = max(2.0, float(fwhm_mhz))
        J_max = max(2, min(int(J_max), 28))
        alpha_sim = float(np.clip(alpha_sim, 0.05, 1.0))
        alpha_exp = float(np.clip(alpha_exp, 0.05, 1.0))
        offset_exp = float(offset_exp)
        zoom_x_factor = max(1.0, float(zoom_x_factor))
        pan_x_pct = float(np.clip(pan_x_pct, -100.0, 100.0))
        zoom_y_factor = max(0.2, float(zoom_y_factor))
        pan_y_offset = float(pan_y_offset)

        if not (A0 >= B0 >= C0 > 0) or not (A1 >= B1 >= C1 > 0):
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
            ax.text(
                0.5, 0.5,
                "⚠️ Restricción Física: Se requiere A ≥ B ≥ C > 0 MHz en ambos estados.",
                ha="center", va="center", fontsize=12, color="#dc2626", fontweight="bold"
            )
            ax.axis("off")
            return fig, "Error de validación física: verifica que A ≥ B ≥ C > 0 MHz."

        # 1. Configure Hamiltonian Model: Pure Rigid Rotor vs. Watson A-Reduction
        is_rigid = "Rotor Rígido" in str(hamiltonian_model)
        if is_rigid:
            dJ_mhz = dJK_mhz = dK_mhz = delJ_mhz = delK_mhz = 0.0
            ham_short_label = "Rotor Rígido Puro"
        else:
            dJ_mhz = float(dJ_khz) * 1e-3
            dJK_mhz = float(dJK_khz) * 1e-3
            dK_mhz = float(dK_khz) * 1e-3
            delJ_mhz = float(delJ_khz) * 1e-3
            delK_mhz = float(delK_khz) * 1e-3
            ham_short_label = "Watson Reducción A"

        p_ref = PRESETS.get(preset_name, PRESETS["Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)"])
        origin_mhz = float(p_ref.get("origin_mhz", 30000.0 * WAVENUMBER_TO_MHZ))
        origin_cm1 = origin_mhz * MHZ_TO_WAVENUMBER

        state_s0 = MoleculeState(
            name="S0", A=A0, B=B0, C=C0,
            Delta_J=dJ_mhz, Delta_JK=dJK_mhz, Delta_K=dK_mhz,
            delta_J=delJ_mhz, delta_K=delK_mhz
        )
        state_s1 = MoleculeState(
            name="S1", A=A1, B=B1, C=C1,
            Delta_J=dJ_mhz * 0.96, Delta_JK=dJK_mhz * 0.96, Delta_K=dK_mhz * 0.96,
            delta_J=delJ_mhz * 0.96, delta_K=delK_mhz * 0.96,
            origin=origin_cm1
        )

        ham_s0 = AsymmetricRotorHamiltonian(state_s0)
        ham_s1 = AsymmetricRotorHamiltonian(state_s1)

        manifold_s0 = ham_s0.compute_manifold(J_max=J_max)
        manifold_s1 = ham_s1.compute_manifold(J_max=J_max)

        # 2. Selection Rule Criterion (Pure a, Pure b, Pure c, or Hybrid)
        sel_crit, dip, sel_desc = _resolve_selection_criterion(band_type_mode, mu_a, mu_b, mu_c)

        calc = RovibronicTransitionCalculator(
            ground_manifold=manifold_s0,
            excited_manifold=manifold_s1,
            band_origin_cm1=origin_cm1,
            temperature_K=T_rot,
            dipole_fractions=dip,
            selection_criterion=sel_crit
        )
        lines = calc.calculate_transitions(J_max_ground=J_max, J_max_excited=J_max)

        # 3. Determine frequency grid and load Experimental / Reference data
        exp_dataset = None
        exp_src_str = str(exp_source_mode)
        if "PmtFla" in exp_src_str and os.path.exists(DEFAULT_PMTFLA_PATH):
            exp_dataset = parse_spectrometer_file(
                DEFAULT_PMTFLA_PATH, fwhm_mhz=fwhm_mhz, reference_origin_mhz=923604613.0295
            )
            origin_mhz = exp_dataset.origin_mhz
        elif "IAR.asc" in exp_src_str and os.path.exists(DEFAULT_IAR_ASC_PATH):
            exp_dataset = parse_spectrometer_file(
                DEFAULT_IAR_ASC_PATH, fwhm_mhz=fwhm_mhz, grid_points=2500, reference_origin_mhz=923604613.0295
            )
            origin_mhz = exp_dataset.origin_mhz
        elif "Subido" in exp_src_str and uploaded_exp_file is not None:
            up_path = uploaded_exp_file.name if hasattr(uploaded_exp_file, "name") else str(uploaded_exp_file)
            if os.path.exists(up_path):
                exp_dataset = parse_spectrometer_file(up_path, fwhm_mhz=fwhm_mhz, grid_points=2500)
                origin_mhz = exp_dataset.origin_mhz
                shift_mhz = origin_mhz - (origin_cm1 * WAVENUMBER_TO_MHZ)
                for ln in lines:
                    ln.frequency_mhz += shift_mhz

        sim = SpectrumSimulator(lines=lines, fwhm_mhz=fwhm_mhz, profile="voigt")

        if exp_dataset is not None and exp_dataset.file_format == "spectrometer_xy":
            f_range = (float(exp_dataset.freq_mhz[0]), float(exp_dataset.freq_mhz[-1]))
            freq_axis, intensity, sub_dict = sim.simulate_subbands(
                grid_points=len(exp_dataset.freq_mhz), freq_range_mhz=f_range
            )
            y_sim_interp = np.interp(exp_dataset.freq_mhz, freq_axis, intensity)
            freq_axis = exp_dataset.freq_mhz
            intensity = y_sim_interp
            for k in sub_dict:
                sub_dict[k] = np.interp(exp_dataset.freq_mhz, np.linspace(f_range[0], f_range[1], len(sub_dict[k])), sub_dict[k])
        elif exp_dataset is not None and exp_dataset.file_format == "jb95_asc":
            f_range = (float(exp_dataset.freq_mhz[0]), float(exp_dataset.freq_mhz[-1]))
            freq_axis, intensity, sub_dict = sim.simulate_subbands(grid_points=2500, freq_range_mhz=f_range)
        else:
            freq_axis, intensity, sub_dict = sim.simulate_subbands(grid_points=2500)

        rel_freq_ghz = (freq_axis - origin_mhz) / 1000.0
        fig, ax = plt.subplots(figsize=(10.6, 5.0), dpi=int(render_dpi))

        ref_note = ""
        if exp_dataset is not None:
            y_ref = exp_dataset.intensity_norm
            if len(y_ref) != len(intensity):
                y_ref = np.interp(freq_axis, exp_dataset.freq_mhz, y_ref)
            norm_sim = np.linalg.norm(intensity)
            norm_ref = np.linalg.norm(y_ref)
            corr_exp = (float(np.dot(intensity, y_ref) / (norm_sim * norm_ref)) * 100.0) if (norm_sim > 0 and norm_ref > 0) else 0.0
            ref_note = (
                f"\n• Comparación vs. Datos Experimentales Reales ({exp_dataset.filename}): "
                f"Correlación Cruzada r = {corr_exp:.2f}%\n"
                f"  └─ {exp_dataset.summary}"
            )
            exp_legend_title = f"Experimental Espectrómetro ({exp_dataset.filename})"
        else:
            s0_ref = MoleculeState("S0_ref", A=p_ref["A0"], B=p_ref["B0"], C=p_ref["C0"])
            s1_ref = MoleculeState("S1_ref", A=p_ref["A1"], B=p_ref["B1"], C=p_ref["C1"], origin=origin_cm1)
            m0_ref = AsymmetricRotorHamiltonian(s0_ref).compute_manifold(J_max=J_max)
            m1_ref = AsymmetricRotorHamiltonian(s1_ref).compute_manifold(J_max=J_max)
            dip_ref = (p_ref["mu_a"], p_ref["mu_b"], p_ref["mu_c"])
            lines_ref = RovibronicTransitionCalculator(
                m0_ref, m1_ref, origin_cm1, p_ref["Trot"], dip_ref
            ).calculate_transitions(J_max, J_max)
            sim_ref = SpectrumSimulator(lines=lines_ref, fwhm_mhz=p_ref["fwhm"], profile="voigt")
            _, y_ref = sim_ref.simulate(grid_points=len(freq_axis), freq_range_mhz=(float(freq_axis[0]), float(freq_axis[-1])))

            norm_sim = np.linalg.norm(intensity)
            norm_ref = np.linalg.norm(y_ref)
            corr_exp = (float(np.dot(intensity, y_ref) / (norm_sim * norm_ref)) * 100.0) if (norm_sim > 0 and norm_ref > 0) else 0.0
            ref_note = (
                f"\n• Comparación vs. Literatura ({p_ref.get('ref', '')}): "
                f"Correlación r = {corr_exp:.2f}% | "
                f"ΔA′={A1 - p_ref['A1']:+.2f} MHz, ΔB′={B1 - p_ref['B1']:+.2f} MHz, ΔC′={C1 - p_ref['C1']:+.2f} MHz"
            )
            exp_legend_title = f"Experimental / Literatura ({p_ref.get('ref', 'Ref')[:28]}...)"

        # STANDARD JCP / JB95 MIRROR CONVENTION:
        # Experimental is ALWAYS on TOP (+Y), and Simulated is inverted on BOTTOM (-Y) when mirror_exp=True!
        if overlay_exp:
            y_ref_plotted = y_ref + offset_exp
            exp_pos_label = "Arriba +Y" if mirror_exp else (f"Offset {offset_exp:+.2f}" if abs(offset_exp) > 1e-3 else "Superpuesto")
            ax.plot(
                rel_freq_ghz, y_ref_plotted,
                color="#ea580c", ls="-" if mirror_exp else "--", lw=1.35, alpha=alpha_exp, zorder=4,
                label=f"{exp_legend_title} [{exp_pos_label}]"
            )
            ax.fill_between(
                rel_freq_ghz, offset_exp, y_ref_plotted,
                color="#f97316", alpha=0.12 * alpha_exp, zorder=2
            )

        if show_sim:
            sim_sign = -1.0 if mirror_exp else 1.0
            y_sim_plotted = sim_sign * intensity
            sim_pos_label = "Espejo Invertido Abajo -Y" if mirror_exp else "Arriba +Y"
            ax.plot(
                rel_freq_ghz, y_sim_plotted,
                color="#1d4ed8", lw=1.45, alpha=alpha_sim, zorder=4,
                label=f"Simulado SpecWatson [{ham_short_label} · Tipo {sel_crit.upper()} · {sim_pos_label}]"
            )
            ax.fill_between(
                rel_freq_ghz, 0.0, y_sim_plotted,
                color="#3b82f6", alpha=0.18 * alpha_sim, zorder=2
            )

        # Optional decomposition of Type a, Type b, Type c sub-bands
        if show_subbands:
            sim_sign = -1.0 if mirror_exp else 1.0
            sub_styles = {
                "a": ("#dc2626", ":", "Sub-espectro Tipo a (ΔKa=par, ΔKc=impar)"),
                "b": ("#16a34a", "-.", "Sub-espectro Tipo b (ΔKa=impar, ΔKc=impar)"),
                "c": ("#9333ea", "--", "Sub-espectro Tipo c (ΔKa=impar, ΔKc=par)"),
            }
            for btype, (col, ls_st, lbl_sub) in sub_styles.items():
                if btype in sub_dict and len(sub_dict[btype]) == len(rel_freq_ghz) and np.max(sub_dict[btype]) > 1e-4:
                    ax.plot(rel_freq_ghz, sim_sign * sub_dict[btype], color=col, ls=ls_st, lw=1.25, alpha=0.9, zorder=5, label=lbl_sub)

        if not show_sim and not overlay_exp:
            ax.text(
                0.5, 0.5,
                "👁️ Ambas curvas están ocultas. Marca 'Mostrar Simulado' o 'Mostrar Experimental' arriba.",
                transform=ax.transAxes, ha="center", va="center", fontsize=10.5, color="#64748b", fontweight="bold"
            )

        if mirror_exp:
            ax.axhline(0.0, color="#334155", lw=1.0, ls="-", alpha=0.75)
        ax.axvline(0.0, color="#ef4444", ls=":", lw=0.95, alpha=0.7, label=f"Origen ν₀ ({origin_mhz*MHZ_TO_WAVENUMBER:.2f} cm⁻¹)")

        # Apply Horizontal Zoom & Pan
        f_min, f_max = float(rel_freq_ghz[0]), float(rel_freq_ghz[-1])
        f_mid = 0.5 * (f_min + f_max)
        half_span = 0.5 * (f_max - f_min)
        view_half_span = half_span / zoom_x_factor
        max_pan = max(0.0, half_span - view_half_span * 0.25)
        center_x = f_mid + (pan_x_pct / 100.0) * max_pan
        ax.set_xlim(center_x - view_half_span, center_x + view_half_span)

        # Apply Vertical Zoom & Pan (Up / Down)
        base_ymin = -1.14 if mirror_exp else min(-0.06, offset_exp - 0.06)
        base_ymax = max(1.14, 1.08 + offset_exp)
        y_mid = 0.5 * (base_ymin + base_ymax) - pan_y_offset
        y_half = 0.5 * (base_ymax - base_ymin) / zoom_y_factor
        ax.set_ylim(y_mid - y_half, y_mid + y_half)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        kappa_s0 = state_s0.kappa
        kappa_s1 = state_s1.kappa

        n_a = sum(1 for ln in lines if ln.band_type == "a")
        n_b = sum(1 for ln in lines if ln.band_type == "b")
        n_c = sum(1 for ln in lines if ln.band_type == "c")
        n_P = sum(1 for ln in lines if ln.branch == "P")
        n_Q = sum(1 for ln in lines if ln.branch == "Q")
        n_R = sum(1 for ln in lines if ln.branch == "R")

        ax.set_title(
            f"Espectro S₁ ← S₀ [{ham_short_label} | Regla: Tipo {sel_crit.upper()} | J_max={J_max}, T_rot={T_rot:.1f} K, FWHM={fwhm_mhz:.0f} MHz]",
            fontsize=11.0, fontweight="bold", pad=10
        )
        ax.set_xlabel("Desplazamiento de Frecuencia Relativa ν - ν₀ (GHz)", fontsize=10.5)
        ax.set_ylabel(
            "Intensidad Normalizada (+Exp Arriba / -Sim Abajo)" if mirror_exp else "Intensidad Normalizada (u.a.)",
            fontsize=10.0
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
        fig.tight_layout()

        stats = (
            f"✓ Simulación cuántica completada en {elapsed_ms:.1f} ms\n"
            f"• Método Hamiltoniano Activo: {state_s0.model_description}\n"
            f"• Criterio de Selección Espectral: {sel_desc}\n"
            f"• Desglose de Transiciones ({len(lines):,} totales): Tipo a = {n_a:,} | Tipo b = {n_b:,} | Tipo c = {n_c:,}  (Ramas: P={n_P:,}, Q={n_Q:,}, R={n_R:,})\n"
            f"• Factor de Asimetría de Ray (κ = (2B-A-C)/(A-C)): κ(S₀) = {kappa_s0:.4f} | κ(S₁) = {kappa_s1:.4f} → {state_s0.kappa_classification}\n"
            f"• Cambio Rotacional al Excitar: ΔA = {A1 - A0:+.2f} MHz, ΔB = {B1 - B0:+.2f} MHz, ΔC = {C1 - C0:+.2f} MHz"
            f"{ref_note}"
        )
        return fig, stats

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en simulación: {str(e)}", ha="center", va="center", color="red")
        ax.axis("off")
        return fig, f"Error detallado: {repr(e)}"


def handle_uploaded_exp_file(
    uploaded_file,
    A0, B0, C0, A1, B1, C1,
    band_type_mode, hamiltonian_model, fwhm_mhz
):
    """
    Processes a newly uploaded spectrometer file (.asc, PmtFla, .dat, .txt, .csv).
    If it is a JB95 .asc file (like IAR.asc), it automatically populates the extracted
    rotational constants (A0, B0, C0, A1, B1, C1), Pure Band Type, and Rigid Rotor mode!
    """
    if uploaded_file is None:
        return (
            A0, B0, C0, A1, B1, C1,
            band_type_mode, hamiltonian_model, EXP_SOURCE_CHOICES[0],
            "⚠️ Primero selecciona o arrastra un archivo (.asc, PmtFla, .dat, .csv) en la casilla de carga."
        )

    up_path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    try:
        ds = parse_spectrometer_file(up_path, fwhm_mhz=float(fwhm_mhz))
        new_band_mode = band_type_mode
        new_ham_mode = hamiltonian_model

        if ds.file_format == "jb95_asc":
            if ds.band_type_counts:
                dom = max(ds.band_type_counts, key=ds.band_type_counts.get)
                if dom == "a":
                    new_band_mode = BAND_TYPE_CHOICES[1]
                elif dom == "b":
                    new_band_mode = BAND_TYPE_CHOICES[0]
                elif dom == "c":
                    new_band_mode = BAND_TYPE_CHOICES[2]
            new_ham_mode = HAMILTONIAN_CHOICES[0]
            if ds.extracted_constants is not None:
                c = ds.extracted_constants
                return (
                    round(c["A0"], 4), round(c["B0"], 4), round(c["C0"], 4),
                    round(c["A1"], 4), round(c["B1"], 4), round(c["C1"], 4),
                    new_band_mode, new_ham_mode, EXP_SOURCE_CHOICES[3],
                    f"✅ {ds.summary}\n👉 Constantes (A, B, C) y Regla de Selección actualizadas automáticamente desde '{ds.filename}'."
                )

        return (
            A0, B0, C0, A1, B1, C1,
            new_band_mode, new_ham_mode, EXP_SOURCE_CHOICES[3],
            f"✅ {ds.summary}\n👉 Espectro experimental cargado y superpuesto en la gráfica."
        )
    except Exception as e:
        return (
            A0, B0, C0, A1, B1, C1,
            band_type_mode, hamiltonian_model, EXP_SOURCE_CHOICES[0],
            f"❌ Error al leer archivo '{os.path.basename(up_path)}': {str(e)}"
        )


def run_automatic_line_assignment_ui(
    exp_file_choice,
    uploaded_raw_file,
    A0, B0, C0,
    A1, B1, C1,
    origin_mhz_in,
    T_rot, fwhm_mhz, J_max,
    band_mode_str, ham_mode_str,
    tolerance_mhz, min_peak_pct, refine_chk
):
    """
    Callback for Automatic Quantum Line Assignment (`A/B/C J' Ka' Kc' <- J'' Ka'' Kc''`)
    from raw spectrometer scans (`PmtFla` or uploaded file) and `.asc` JB95 file export.
    Plots:
      - Top Panel (JCP Mirror Mode): Experimental `PmtFla` on TOP (+Y) with assigned peak markers
        and quantum transition labels, and Simulated + Stick Spectrum inverted on BOTTOM (-Y).
      - Bottom Panel: Residual errors (ν_exp - ν_calc in MHz) for all assigned lines.
    """
    plt.close("all")
    try:
        A0, B0, C0 = float(A0), float(B0), float(C0)
        A1, B1, C1 = float(A1), float(B1), float(C1)
        origin_mhz = float(origin_mhz_in)
        T_rot = max(0.2, float(T_rot))
        fwhm_mhz = max(2.0, float(fwhm_mhz))
        J_max = max(4, min(int(J_max), 30))
        tol_mhz = max(0.5, float(tolerance_mhz))
        min_peak_h = max(0.01, float(min_peak_pct) / 100.0)

        # Load experimental raw scan
        if "Subido" in str(exp_file_choice) and uploaded_raw_file is not None:
            up_path = uploaded_raw_file.name if hasattr(uploaded_raw_file, "name") else str(uploaded_raw_file)
            exp_ds = parse_spectrometer_file(up_path, fwhm_mhz=fwhm_mhz, reference_origin_mhz=origin_mhz)
        else:
            exp_ds = parse_spectrometer_file(DEFAULT_PMTFLA_PATH, fwhm_mhz=fwhm_mhz, reference_origin_mhz=origin_mhz)

        is_rigid = "Rotor Rígido" in str(ham_mode_str)
        dJ_val = 0.0 if is_rigid else 0.00025
        sel_crit, dip, _ = _resolve_selection_criterion(band_mode_str, 0.0, 1.0, 0.0)
        if sel_crit == "a":
            dip = (1.0, 0.0, 0.0)
        elif sel_crit == "c":
            dip = (0.0, 0.0, 1.0)
        else:
            dip = (0.0, 1.0, 0.0)
            sel_crit = "b"

        s0 = MoleculeState("S0", A=A0, B=B0, C=C0, Delta_J=dJ_val)
        s1 = MoleculeState("S1", A=A1, B=B1, C=C1, Delta_J=dJ_val, origin=origin_mhz * MHZ_TO_WAVENUMBER)

        asg_res = auto_assign_spectrometer_lines(
            exp_dataset=exp_ds,
            ground_state=s0,
            excited_state=s1,
            origin_mhz=origin_mhz,
            J_max=J_max,
            temperature_K=T_rot,
            dipole_fractions=dip,
            selection_criterion=sel_crit,
            tolerance_mhz=tol_mhz,
            min_peak_height=min_peak_h,
            min_jb95_intensity=0.010,
            refine_constants=bool(refine_chk),
            output_filename=f"Asignacion_{exp_ds.filename}_SpecWatson.asc"
        )

        # Build simulated continuous contour for the mirror plot
        m0 = AsymmetricRotorHamiltonian(s0).compute_manifold(J_max)
        m1 = AsymmetricRotorHamiltonian(s1).compute_manifold(J_max)
        calc = RovibronicTransitionCalculator(
            m0, m1, origin_mhz * MHZ_TO_WAVENUMBER, T_rot, dip, selection_criterion=sel_crit
        )
        sim_lines = calc.calculate_transitions(J_max, J_max)
        sim = SpectrumSimulator(sim_lines, fwhm_mhz=fwhm_mhz, profile="voigt")
        f_sim, y_sim = sim.simulate(
            grid_points=len(exp_ds.freq_mhz),
            freq_range_mhz=(float(exp_ds.freq_mhz[0]), float(exp_ds.freq_mhz[-1]))
        )

        rel_exp_ghz = (exp_ds.freq_mhz - origin_mhz) / 1000.0
        rel_sim_ghz = (f_sim - origin_mhz) / 1000.0

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11.0, 6.2), dpi=135,
            gridspec_kw={"height_ratios": [3.2, 1.2]}, sharex=True
        )

        # Top Panel: Experimental on TOP (+Y) and Simulated + Assigned Sticks on BOTTOM (-Y)
        ax1.plot(
            rel_exp_ghz, exp_ds.intensity_norm,
            color="#ea580c", lw=1.25, alpha=0.92,
            label=f"Experimental Espectrómetro '{exp_ds.filename}' (Arriba +Y)"
        )
        ax1.fill_between(rel_exp_ghz, 0.0, exp_ds.intensity_norm, color="#f97316", alpha=0.12)

        ax1.plot(
            rel_sim_ghz, -y_sim,
            color="#1d4ed8", lw=1.35, alpha=0.90,
            label=f"Simulado Cuántico [{('Rotor Rígido' if is_rigid else 'Watson')} · Tipo {sel_crit.upper()}] (Espejo Abajo -Y)"
        )
        ax1.fill_between(rel_sim_ghz, 0.0, -y_sim, color="#3b82f6", alpha=0.15)

        # Plot assigned quantum sticks & peak markers
        if asg_res.assigned_entries:
            asg_f_ghz = np.array([(e.freq_calc_mhz - origin_mhz) / 1000.0 for e in asg_res.assigned_entries])
            asg_obs_ghz = np.array([(e.freq_obs_mhz - origin_mhz) / 1000.0 for e in asg_res.assigned_entries])
            asg_i_raw = np.array([e.intensity_jb95 for e in asg_res.assigned_entries])
            max_i_asg = float(np.max(asg_i_raw)) if np.max(asg_i_raw) > 0 else 1.0
            asg_i_norm = (asg_i_raw / max_i_asg) * 0.88

            # Quantum stick lines pointing downwards in the simulated half (-Y)
            ax1.vlines(
                asg_f_ghz, 0.0, -asg_i_norm,
                color="#059669", lw=0.85, alpha=0.75,
                label=f"{len(asg_res.assigned_entries)} Líneas Cuánticas Asignadas ({sel_crit.upper()} J'Ka'Kc' ← J''Ka''Kc'')"
            )

            # Experimental peak markers on top (+Y)
            obs_heights = np.interp(asg_obs_ghz, rel_exp_ghz, exp_ds.intensity_norm)
            ax1.scatter(
                asg_obs_ghz, obs_heights,
                s=16, color="#10b981", edgecolors="#064e3b", linewidths=0.6, zorder=6,
                label="Picos Experimentales Emparejados"
            )

            # Annotate up to 6 prominent well-separated assigned peaks with their quantum numbers
            sorted_by_int = sorted(asg_res.assigned_entries, key=lambda x: x.intensity_jb95, reverse=True)
            labeled_x = []
            for entry in sorted_by_int:
                x_ghz = (entry.freq_obs_mhz - origin_mhz) / 1000.0
                if all(abs(x_ghz - lx) > 3.5 for lx in labeled_x) and len(labeled_x) < 6:
                    labeled_x.append(x_ghz)
                    y_val = float(np.interp(x_ghz, rel_exp_ghz, exp_ds.intensity_norm))
                    q_lbl = f"{entry.band_type} {entry.J_up}({entry.Ka_up},{entry.Kc_up})←{entry.J_low}({entry.Ka_low},{entry.Kc_low})"
                    ax1.annotate(
                        q_lbl,
                        xy=(x_ghz, y_val),
                        xytext=(x_ghz, min(1.16, y_val + 0.14)),
                        fontsize=7.2, fontweight="bold", color="#065f46", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#059669", lw=0.7)
                    )

        ax1.axhline(0.0, color="#1e293b", lw=1.0)
        ax1.axvline(0.0, color="#ef4444", ls=":", lw=0.95, alpha=0.75, label="Origen de Banda ν₀")
        ax1.set_ylim(-1.12, 1.24)
        ax1.set_ylabel("Intensidad (+Exp Arriba / -Sim Abajo)", fontsize=9.8)
        ax1.set_title(
            f"Asignación Automática de Líneas Cuánticas (Formato JB95 .asc): {asg_res.assigned_count} Transiciones Asignadas (RMS = {asg_res.rms_residual_mhz:.2f} MHz)",
            fontsize=10.8, fontweight="bold"
        )
        ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
        ax1.grid(True, ls="--", alpha=0.35)

        # Bottom Panel: Residuals (ν_exp - ν_calc in MHz)
        if asg_res.assigned_entries:
            res_x = np.array([(e.freq_calc_mhz - origin_mhz) / 1000.0 for e in asg_res.assigned_entries])
            res_y = np.array([e.residual_mhz for e in asg_res.assigned_entries])
            ax2.vlines(res_x, 0.0, res_y, color="#10b981", lw=0.9, alpha=0.7)
            ax2.scatter(res_x, res_y, s=18, color="#059669", edgecolors="#064e3b", linewidths=0.5, zorder=4,
                        label=f"Residual ν_exp - ν_calc (RMS = {asg_res.rms_residual_mhz:.2f} MHz)")
            ax2.axhspan(-asg_res.rms_residual_mhz, asg_res.rms_residual_mhz, color="#10b981", alpha=0.12, label="Banda ±1 RMS")

        ax2.axhline(0.0, color="#1e293b", lw=0.8, ls="--")
        ax2.set_xlabel("Desplazamiento de Frecuencia Relativa ν - ν₀ (GHz)", fontsize=10.0)
        ax2.set_ylabel("Residual (MHz)", fontsize=9.5)
        ax2.set_ylim(-tol_mhz * 1.15, tol_mhz * 1.15)
        ax2.legend(loc="upper right", fontsize=8.0)
        ax2.grid(True, ls="--", alpha=0.35)

        fig.tight_layout()

        return (
            fig,
            asg_res.report_text,
            asg_res.asc_preview_text,
            asg_res.assigned_table_rows[:150],
            asg_res.asc_file_path
        )

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en asignación automática: {str(e)}", ha="center", color="red")
        ax.axis("off")
        return fig, f"Error detallado: {repr(e)}", "", [], None


def export_spectrum_high_dpi(
    A0, B0, C0,
    A1, B1, C1,
    T_rot, fwhm_mhz,
    mu_a, mu_b, mu_c,
    J_max,
    preset_name,
    band_type_mode,
    hamiltonian_model,
    dJ_khz, dJK_khz, dK_khz, delJ_khz, delK_khz,
    exp_source_mode,
    uploaded_exp_file,
    show_subbands,
    show_sim,
    overlay_exp,
    mirror_exp,
    alpha_sim,
    alpha_exp,
    offset_exp,
    zoom_x_factor,
    pan_x_pct,
    zoom_y_factor,
    pan_y_offset,
    export_format_choice
):
    """Renders the exact current view of the spectrum and saves it at 300 DPI or 600 DPI (PNG / PDF)."""
    is_600 = "600" in str(export_format_choice)
    is_pdf = "PDF" in str(export_format_choice)
    dpi_val = 600 if is_600 else 300
    ext = "pdf" if is_pdf else "png"

    fig, _ = simulate_rovibronic_spectrum(
        A0, B0, C0, A1, B1, C1,
        T_rot, fwhm_mhz, mu_a, mu_b, mu_c, J_max,
        preset_name=preset_name,
        band_type_mode=band_type_mode,
        hamiltonian_model=hamiltonian_model,
        dJ_khz=dJ_khz, dJK_khz=dJK_khz, dK_khz=dK_khz, delJ_khz=delJ_khz, delK_khz=delK_khz,
        exp_source_mode=exp_source_mode,
        uploaded_exp_file=uploaded_exp_file,
        show_subbands=show_subbands,
        show_sim=show_sim,
        overlay_exp=overlay_exp,
        mirror_exp=mirror_exp,
        alpha_sim=alpha_sim,
        alpha_exp=alpha_exp,
        offset_exp=offset_exp,
        zoom_x_factor=zoom_x_factor,
        pan_x_pct=pan_x_pct,
        zoom_y_factor=zoom_y_factor,
        pan_y_offset=pan_y_offset,
        render_dpi=dpi_val
    )

    safe_mol = str(preset_name).split("(")[0].strip().replace(" ", "_").replace("-", "_")
    filename = f"SpecWatson_{safe_mol}_{dpi_val}DPI.{ext}"
    out_path = os.path.join(tempfile.gettempdir(), filename)
    fig.savefig(out_path, dpi=dpi_val, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def reset_view_controls():
    """Resets visibility, opacity, vertical offset, and zoom/pan controls to defaults (Mirror Mode ON)."""
    return False, True, True, True, 0.85, 0.85, 0.0, 1.0, 0.0, 1.0, 0.0


def run_autonomous_fit_demo(molecule_choice, fit_band_mode, fit_ham_mode, noise_pct, max_iter):
    """Callback for Tab 2B: Live Differential Evolution + Simplex Contour Fitting."""
    t0 = time.perf_counter()
    plt.close("all")

    try:
        p = PRESETS.get(
            molecule_choice,
            PRESETS["Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)"]
        )
        sel_crit, dip, sel_desc = _resolve_selection_criterion(
            fit_band_mode, p["mu_a"], p["mu_b"], p["mu_c"]
        )
        is_rigid = "Rotor Rígido" in str(fit_ham_mode)
        dJ_val = 0.0 if is_rigid else 0.00025

        guess_A = p["A1"] - 28.0
        guess_B = p["B1"] + 12.0
        guess_C = max(10.0, p["C1"] - 10.0)
        bounds = [
            (p["A1"] - 45.0, p["A1"] + 45.0),
            (p["B1"] - 25.0, p["B1"] + 25.0),
            (max(5.0, p["C1"] - 20.0), p["C1"] + 20.0),
            (-15.0, 15.0)
        ]
        j_fit = 8
        origin_mhz = float(p.get("origin_mhz", 30000.0 * WAVENUMBER_TO_MHZ))
        origin_cm1 = origin_mhz * MHZ_TO_WAVENUMBER

        s0 = MoleculeState("S0", A=p["A0"], B=p["B0"], C=p["C0"], Delta_J=dJ_val)
        s1_true = MoleculeState("S1_true", A=p["A1"], B=p["B1"], C=p["C1"], Delta_J=dJ_val, origin=origin_cm1)
        s1_guess = MoleculeState("S1_guess", A=guess_A, B=guess_B, C=guess_C, Delta_J=dJ_val, origin=origin_cm1)

        g_man = AsymmetricRotorHamiltonian(s0).compute_manifold(j_fit)
        e_man_true = AsymmetricRotorHamiltonian(s1_true).compute_manifold(j_fit)

        calc_true = RovibronicTransitionCalculator(
            g_man, e_man_true, origin_cm1, p["Trot"], dip, selection_criterion=sel_crit
        )
        lines_true = calc_true.calculate_transitions(j_fit, j_fit)

        sim_true = SpectrumSimulator(lines_true, fwhm_mhz=p["fwhm"], profile="gaussian")
        freq_axis, y_clean = sim_true.simulate(grid_points=1200)

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, (float(noise_pct) / 100.0) * np.max(y_clean), size=len(y_clean))
        y_exp = np.clip(y_clean + noise, 0.0, None)
        if y_exp.max() > 0:
            y_exp /= y_exp.max()

        fitter = ContourFitter(
            exp_freq_mhz=freq_axis,
            exp_intensity=y_exp,
            ground_state=s0,
            initial_excited_state=s1_guess,
            J_max=j_fit,
            fwhm_mhz=p["fwhm"],
            temperature_K=p["Trot"],
            dipole_fractions=dip,
            selection_criterion=sel_crit
        )

        y_init = fitter._forward_model(guess_A, guess_B, guess_C, 0.0)
        if y_init.max() > 0:
            y_init /= y_init.max()

        de_res = fitter.fit_global(bounds=bounds, maxiter=int(max_iter), popsize=8)
        local_res = fitter.refine_local(de_res["best_params"])
        best_p = local_res["refined_params"]
        corr_pct = -fitter.objective_cross_correlation(best_p) * 100.0

        y_fit = fitter._forward_model(*best_p)
        if y_fit.max() > 0:
            y_fit /= y_fit.max()

        rel_f = freq_axis - origin_mhz

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10.5, 5.5), dpi=130,
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Mirror plot: Experimental on TOP (+Y) and Fitted Simulation on BOTTOM (-Y)
        ax1.plot(rel_f, y_exp, color="#ea580c", lw=1.25, alpha=0.90, label=f"🟠 Experimental Arriba +Y ({p.get('ref', 'Literatura')[:24]} + {noise_pct}% ruido)")
        ax1.fill_between(rel_f, 0.0, y_exp, color="#f97316", alpha=0.12)
        ax1.plot(rel_f, -y_init, color="#f59e0b", ls=":", lw=1.3, label=f"Semilla Inicial Ab Initio Invertida (A′={guess_A:.0f} MHz)")
        ax1.plot(rel_f, -y_fit, color="#1d4ed8", ls="-", lw=1.6, label=f"🔵 Ajuste Híbrido DE + Simplex Abajo -Y (r = {corr_pct:.2f}%)")
        ax1.fill_between(rel_f, 0.0, -y_fit, color="#3b82f6", alpha=0.15)
        ax1.axhline(0.0, color="#334155", lw=0.9)
        ax1.set_ylabel("Intensidad (+Exp / -Sim)", fontsize=10)
        ax1.set_title(
            f"Ajuste Evolutivo en Modo Espejo [{('Rotor Rígido' if is_rigid else 'Watson')} · Tipo {sel_crit.upper()}]: {molecule_choice}",
            fontsize=10.8, fontweight="bold"
        )
        ax1.legend(loc="upper right", fontsize=8.3)
        ax1.grid(True, ls="--", alpha=0.4)

        residual = y_exp - y_fit
        ax2.plot(rel_f, residual, color="#10b981", lw=1.0, label="Residual (I_exp - I_fit)")
        ax2.axhline(0, color="black", lw=0.7, ls="--")
        ax2.set_xlabel("Frecuencia Relativa ν - ν₀ (MHz)", fontsize=10)
        ax2.set_ylabel("Residual", fontsize=10)
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()

        elapsed_s = time.perf_counter() - t0
        report = (
            f"✓ Ajuste Híbrido (Evolución Diferencial + Simplex) completado en {elapsed_s:.2f} s\n"
            f"• Método Hamiltoniano: {'Rotor Rígido Puro (ΔJ=0)' if is_rigid else 'Watson Reducción A'}\n"
            f"• Criterio de Selección: {sel_desc}\n"
            f"• Correlación Cruzada Alcanzada: {corr_pct:.2f}%\n"
            f"• A′ Recuperada: {best_p[0]:.2f} MHz  (Valor Real: {p['A1']:.2f} MHz | Error: {abs(best_p[0]-p['A1']):.2f} MHz)\n"
            f"• B′ Recuperada: {best_p[1]:.2f} MHz  (Valor Real: {p['B1']:.2f} MHz | Error: {abs(best_p[1]-p['B1']):.2f} MHz)\n"
            f"• C′ Recuperada: {best_p[2]:.2f} MHz  (Valor Real: {p['C1']:.2f} MHz | Error: {abs(best_p[2]-p['C1']):.2f} MHz)"
        )
        return fig, report

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en ajuste: {str(e)}", ha="center", color="red")
        return fig, f"Error detallado: {repr(e)}"


def run_quantum_3d_tab(query_input, engine_choice, num_confs, s1_expansion):
    """Callback for Tab 3: 3D Quantum Chemistry & Inertia Tensor Calculator."""
    plt.close("all")
    try:
        res = run_quantum_3d_pipeline(
            smiles_or_name=query_input,
            engine_choice=engine_choice,
            num_conformers=int(num_confs),
            pi_star_expansion_pct=float(s1_expansion)
        )
        fig_pas = plot_molecular_pas_figure(res)
        viewer_html = render_3dmol_pas_html(res)

        if res.is_linear:
            planarity_note = "Molécula LINEAL Simétrica (Ia = 0, rotor lineal con constante única B₀ = C₀)"
        elif abs(res.inertial_defect) < 0.8:
            planarity_note = "Molécula PLANA en el plano ab (ΔI ≈ 0)"
        else:
            planarity_note = "Molécula NO PLANA / Con estructura 3D fuera del plano"

        k_val = res.kappa
        if k_val <= -0.95:
            k_desc = "Trompo Cuasi-Simétrico Prolato Extremo (tipo aguja/cigarro, A ≫ B ≈ C)"
        elif k_val <= -0.60:
            k_desc = "Rotor Asimétrico Prolato (alargado en el eje a, A > B ≈ C)"
        elif k_val < 0.60:
            k_desc = "Rotor de Asimetría Intermedia (A > B > C)"
        else:
            k_desc = "Rotor Asimétrico Oblato (aplanado tipo disco, A ≈ B > C)"

        fracs = {"a (Tipo a)": res.mu_a_frac, "b (Tipo b)": res.mu_b_frac, "c (Tipo c)": res.mu_c_frac}
        dom_axis = max(fracs, key=fracs.get)

        lit_block = ""
        if res.formula in EXPERIMENTAL_LITERATURE_DB:
            lit = EXPERIMENTAL_LITERATURE_DB[res.formula]
            if res.is_linear:
                err_b = abs(res.B0_mhz - float(lit["B0"])) / float(lit["B0"]) * 100.0
                lit_block = (
                    f"\n📊 VALIDACIÓN CONTRA LITERATURA EXPERIMENTAL ({lit['ref']}):\n"
                    f"   • B₀ = C₀ Experimental: {lit['B0']:.2f} MHz  vs  Calculado 3D: {res.B0_mhz:.2f} MHz (Desviación: {err_b:.2f}%)"
                )
            else:
                err_b = abs(res.B0_mhz - float(lit["B0"])) / float(lit["C0"]) * 100.0 if False else abs(res.B0_mhz - float(lit["B0"])) / float(lit["B0"]) * 100.0
                err_c = abs(res.C0_mhz - float(lit["C0"])) / float(lit["C0"]) * 100.0
                lit_block = (
                    f"\n📊 VALIDACIÓN CONTRA LITERATURA EXPERIMENTAL ({lit['ref']}):\n"
                    f"   • Experimental (S₀): A″={lit['A0']:.2f}, B″={lit['B0']:.2f}, C″={lit['C0']:.2f} MHz "
                    f"(Desviación geométrica 3D sin ajuste: B={err_b:.1f}%, C={err_c:.1f}%)"
                )

        report = (
            f"✓ Geometría 3D y Tensor de Inercia calculados en {res.elapsed_sec*1000:.1f} ms\n"
            f"• Fórmula Molecular: {res.formula} ({res.num_atoms} átomos)  |  SMILES: {res.smiles}\n"
            f"• Motor Cuántico: {res.engine_used}  |  Hardware: {res.device_used}  |  Energía: {res.energy_kcal_or_eh}\n"
            f"• Momentos Principales de Inercia: Ia = {res.Ia:.2f}, Ib = {res.Ib:.2f}, Ic = {res.Ic:.2f} u·Å²\n"
            f"• Defecto de Inercia Cuántico: ΔI = Ic - Ia - Ib = {res.inertial_defect:+.4f} u·Å² ({planarity_note})\n"
            f"• Constantes Estado Basal (S₀): A″ = {res.A0_mhz:.2f} MHz, B″ = {res.B0_mhz:.2f} MHz, C″ = {res.C0_mhz:.2f} MHz\n"
            f"• Factor de Asimetría de Ray: κ = {res.kappa:.4f} → {k_desc}\n"
            f"• Momento Dipolar Permanente: |μ| = {res.dipole_total_debye:.2f} D (Orientado principalmente sobre Eje {dom_axis}: μₐ²={res.mu_a_frac:.2f}, μ_b²={res.mu_b_frac:.2f}, μ_c²={res.mu_c_frac:.2f})"
            f"{lit_block}"
        )

        return (
            fig_pas,
            viewer_html,
            report,
            res.A0_mhz, res.B0_mhz, res.C0_mhz,
            res.A1_mhz, res.B1_mhz, res.C1_mhz,
            res.mu_a_frac, res.mu_b_frac, res.mu_c_frac
        )
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"⚠️ {str(e)}", ha="center", va="center", color="#dc2626", fontsize=11, wrap=True)
        ax.axis("off")
        err_html = f"<div style='padding:20px;color:#ef4444;font-weight:bold;'>{str(e)}</div>"
        return (
            fig, err_html, f"Error: {str(e)}",
            2253.24, 594.21, 470.37,
            2250.61, 586.35, 465.29,
            0.0, 1.0, 0.0
        )


def select_catalog_molecule(catalog_choice):
    default_key = "Carbazol (C12H9N - Aromático Tricíclico | Pratt & Álvarez-Valtierra)"
    info = MOLECULE_CATALOG.get(catalog_choice, MOLECULE_CATALOG[default_key])
    return info["smiles"]


def load_preset_values(preset_name):
    default_key = "Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)"
    p = PRESETS.get(preset_name, PRESETS[default_key])
    return (
        p["A0"], p["B0"], p["C0"],
        p["A1"], p["B1"], p["C1"],
        p["Trot"], p["fwhm"],
        p["mu_a"], p["mu_b"], p["mu_c"],
        p["Jmax"],
        p.get("band_mode", BAND_TYPE_CHOICES[0]),
        p.get("ham_mode", HAMILTONIAN_CHOICES[0]),
        p.get("exp_source", EXP_SOURCE_CHOICES[2]),
    )


def build_gradio_app():
    import gradio as gr

    with gr.Blocks(title="SpecWatson · Espectroscopía Rovibrónica & Química Cuántica GPU") as demo:
        gr.Markdown(
            """
            # 🔬 SpecWatson: Suite de Espectroscopía Rovibrónica & Química Cuántica en GPU
            **Universidad de Guanajuato · División de Ciencias e Ingenierías (Campus León)**  
            **Autor:** Miguel Díaz (Maestría en Ciencias) · **Dirección y Respaldo Científico:** Dr. Leonardo Álvarez-Valtierra & Dr. Alejandro Gil-Villegas (SNII Nivel III)  
            *Simulación cuántica por Rotor Rígido y Hamiltoniano de Watson (Base de Wang), Reglas de Selección Tipo a/b/c, Asignación Automática de Líneas Cuánticas (`.asc` estilo JB95) desde datos crudos del espectrómetro (`PmtFla`) y Química Cuántica 3D.*
            """
        )

        with gr.Tabs() as main_tabs:
            # =========================================================================
            # TAB 1: SIMULACIÓN DIRECTA DE ESPECTROS (WANG) + ARCHIVOS ESPECTRÓMETRO
            # =========================================================================
            with gr.TabItem("⚡ 1. Simulador Cuántico de Espectros (Rotor Rígido / Watson)", id="tab_sim"):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)",
                        label="📁 1. Sistema Molecular (Incluye Datos Experimentales Reales de Carbazol del Dr. Leonardo)",
                        scale=3
                    )
                    show_sim_chk = gr.Checkbox(
                        value=True,
                        label="🔵 Mostrar Simulado (SpecWatson)",
                        scale=1
                    )
                    overlay_exp_chk = gr.Checkbox(
                        value=True,
                        label="🟠 Mostrar Experimental",
                        scale=1
                    )
                    mirror_exp_chk = gr.Checkbox(
                        value=True,
                        label="🪞 Modo Espejo JCP (Exp Arriba +Y / Sim Abajo -Y)",
                        scale=1
                    )

                with gr.Row():
                    hamiltonian_dropdown = gr.Dropdown(
                        choices=HAMILTONIAN_CHOICES,
                        value=HAMILTONIAN_CHOICES[0],
                        label="⚙️ 2. Método Hamiltoniano (Rotor Rígido Puro vs. Distorsión Centrífuga de Watson)",
                        scale=2
                    )
                    band_type_dropdown = gr.Dropdown(
                        choices=BAND_TYPE_CHOICES,
                        value=BAND_TYPE_CHOICES[0],
                        label="🎯 3. Criterio de Selección de Espectros (Tipo a, Tipo b, Tipo c o Híbrido)",
                        scale=2
                    )
                    exp_source_dropdown = gr.Dropdown(
                        choices=EXP_SOURCE_CHOICES,
                        value=EXP_SOURCE_CHOICES[0],
                        label="📡 4. Fuente de Datos Experimentales a Comparar",
                        scale=2
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Estado Basal (S₀) y Excitado (S₁) - MHz")
                        with gr.Row():
                            a0 = gr.Number(value=2253.2409, label="A″ Basal (MHz)")
                            b0 = gr.Number(value=594.2103, label="B″ Basal (MHz)")
                            c0 = gr.Number(value=470.3670, label="C″ Basal (MHz)")

                        with gr.Row():
                            a1 = gr.Number(value=2250.6104, label="A′ Excitado (MHz)")
                            b1 = gr.Number(value=586.3482, label="B′ Excitado (MHz)")
                            c1 = gr.Number(value=465.2923, label="C′ Excitado (MHz)")

                        gr.Markdown("### Condiciones del Jet Supersónico")
                        with gr.Row():
                            trot = gr.Slider(0.5, 15.0, value=2.6, step=0.1, label="Temp. Rotacional (K)")
                            fwhm = gr.Slider(5.0, 80.0, value=45.0, step=1.0, label="Ancho Láser FWHM (MHz)")
                            jmax = gr.Slider(4, 28, value=22, step=1, label="J Máximo")

                        gr.Markdown("### Pesos Dipolares de Transición (μₐ², μ_b², μ_c²)")
                        with gr.Row():
                            mu_a = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μₐ² (Tipo a)")
                            mu_b = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="μ_b² (Tipo b)")
                            mu_c = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μ_c² (Tipo c)")

                        with gr.Accordion("🔬 Constantes de Distorsión Centrífuga de Watson (Se apagan a 0 en Rotor Rígido)", open=False):
                            with gr.Row():
                                dJ_in = gr.Number(value=0.25, label="ΔJ (kHz)")
                                dJK_in = gr.Number(value=-0.60, label="ΔJK (kHz)")
                                dK_in = gr.Number(value=1.20, label="ΔK (kHz)")
                            with gr.Row():
                                delJ_in = gr.Number(value=0.04, label="δJ (kHz)")
                                delK_in = gr.Number(value=0.30, label="δK (kHz)")

                        with gr.Accordion("📤 Cargar Archivos Directos del Espectrómetro (.asc, PmtFla, .dat, .csv)", open=True):
                            gr.Markdown(
                                "*Sube cualquier archivo crudo del espectrómetro (`PmtFla`, `.dat`, `.csv`) o lista de líneas de JB95 (`.asc` como `IAR.asc`):*"
                            )
                            exp_file_upload = gr.File(
                                label="Subir Archivo del Espectrómetro (.asc, PmtFla, .dat, .txt, .csv)",
                                file_count="single"
                            )
                            btn_load_exp = gr.Button(
                                "📥 Procesar Archivo Experimental y Extraer Constantes",
                                variant="secondary",
                                size="sm"
                            )
                            upload_status_out = gr.Textbox(
                                label="Estado del Archivo Experimental Cargado",
                                value="✅ Pre-cargados en memoria: 'recursos/PmtFla' (1,919 puntos espectrómetro) y 'recursos/IAR.asc' (2,705 líneas JB95 Tipo B).",
                                lines=2
                            )

                        btn_sim = gr.Button("⚡ Simular y Comparar Espectro Cuántico", variant="primary", size="lg")
                        btn_goto_assign = gr.Button("🎯 Ir a Asignación Automática de Líneas (.asc / JB95) desde PmtFla →", variant="secondary")

                    with gr.Column(scale=2):
                        plot_out = gr.Plot(label="Espectro Experimental (Arriba +Y) vs. Simulado SpecWatson (Abajo -Y)")

                        with gr.Accordion("🎛️ Controles de Visualización: Sub-bandas (a/b/c), Opacidad, Offset, Zoom y Exportación (300/600 DPI)", open=True):
                            with gr.Row():
                                show_subbands_chk = gr.Checkbox(
                                    value=False,
                                    label="🎨 Desglosar sub-espectros individuales Tipo a, Tipo b y Tipo c en la gráfica",
                                    scale=2
                                )
                                alpha_sim_slider = gr.Slider(0.05, 1.0, value=0.85, step=0.05, label="🔵 Opacidad Simulado", scale=1)
                                alpha_exp_slider = gr.Slider(0.05, 1.0, value=0.85, step=0.05, label="🟠 Opacidad Experimental", scale=1)
                                offset_exp_slider = gr.Slider(-1.2, 1.2, value=0.0, step=0.05, label="↕️ Separar Experimental (ΔY)", scale=1)

                            with gr.Row():
                                zoom_x_slider = gr.Slider(1.0, 12.0, value=1.0, step=0.5, label="🔍 Zoom Horizontal (1x–12x)")
                                pan_x_slider = gr.Slider(-100.0, 100.0, value=0.0, step=2.0, label="↔️ Desplazar Izq / Der (%)")
                                zoom_y_slider = gr.Slider(0.3, 4.0, value=1.0, step=0.1, label="🔍 Zoom Vertical (Altura)")
                                pan_y_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="↕️ Desplazar Vista Arriba / Abajo")

                            with gr.Row():
                                btn_reset_view = gr.Button("🔄 Restablecer Vista / Zoom / Opacidad", variant="secondary", scale=1)
                                export_dpi_dropdown = gr.Dropdown(
                                    choices=[
                                        "600 DPI — PNG Ultra-Alta Resolución (Calidad Revista JCP)",
                                        "300 DPI — PNG Alta Resolución (Tesis / Presentación)",
                                        "600 DPI — PDF Vectorial (Artículo LaTeX)"
                                    ],
                                    value="600 DPI — PNG Ultra-Alta Resolución (Calidad Revista JCP)",
                                    label="🖨️ Formato y Resolución de Exportación",
                                    scale=2
                                )
                                btn_export_dpi = gr.Button("📥 Exportar Gráfica Actual (300 / 600 DPI)", variant="primary", scale=1)

                            export_file_out = gr.File(label="📄 Archivo Exportado (Haz clic en el enlace derecho para descargar a tu PC)")

                        stats_out = gr.Textbox(label="Diagnóstico Cuántico: Método, Reglas de Selección (a/b/c), Factor κ y Correlación Experimental", lines=7)

                        with gr.Accordion("📖 Guía Conceptual Rápida: ¿Qué significan Espectros Tipo a/b/c, Momentos Dipolares (μ), Factor κ y Rotor Rígido?", open=False):
                            gr.Markdown(
                                r"""
                                * **1. ¿Qué son los espectros Tipo $a$, Tipo $b$ y Tipo $c$?**  
                                  Toda molécula tiene 3 ejes principales de inercia perpendiculares entre sí: **$a$** (el eje más largo, menor inercia $I_a$), **$b$** (eje corto en el plano molecular, inercia intermedia $I_b$) y **$c$** (perpendicular al plano aromático, mayor inercia $I_c$). Según hacia qué eje oscile el **momento dipolar de transición electrónica** al absorber el fotón UV, cambian las reglas cuánticas de selección $(\Delta K_a, \Delta K_c)$:
                                  * **Tipo $a$ ($\mu_a^2 = 1$, $\Delta K_a = 0, \pm 2\text{ [par]}$, $\Delta K_c = \pm 1\text{ [impar]}$):** Genera una rama $Q$ central muy intensa y compacta en $\nu_0$.
                                  * **Tipo $b$ ($\mu_b^2 = 1$, $\Delta K_a = \pm 1\text{ [impar]}$, $\Delta K_c = \pm 1\text{ [impar]}$):** **Caso del Carbazol** (donde el eje de simetría $C_2$ sobre el enlace N–H es el eje $b$). Produce una depresión o hueco característico justo en el origen $\nu_0$ y sub-bandas laterales $K_a$ bien resueltas.
                                  * **Tipo $c$ ($\mu_c^2 = 1$, $\Delta K_a = \pm 1\text{ [impar]}$, $\Delta K_c = 0, \pm 2\text{ [par]}$):** Transición perpendicular al plano molecular.
                                  * **Banda Híbrida:** Cuando la molécula carece de simetría alta (ej. 1-Naftol, Indol, Resveratrol), el dipolo apunta en diagonal entre los ejes y el espectro es una suma ponderada $\mu_a^2 I_a + \mu_b^2 I_b + \mu_c^2 I_c$.
                                * **2. ¿Qué es el Factor $\kappa$ (Parámetro de Asimetría de Ray)?**  
                                  Definido como $\kappa = \frac{2B - A - C}{A - C}$, mide la forma geométrica del rotor entre **$-1$ y $+1$**:
                                  * **$\kappa = -1$:** Trompo simétrico **prolato** (forma alargada de cigarro: $A > B = C$). El Carbazol tiene $\kappa = -0.8611$ (rotor asimétrico casi prolato).
                                  * **$\kappa = 0$:** Rotor de máxima asimetría ($B$ está exactamente a la mitad entre $A$ y $C$).
                                  * **$\kappa = +1$:** Trompo simétrico **oblato** (forma de moneda/disco: $A = B > C$).
                                * **3. ¿Qué diferencia hay entre "Rotor Rígido Puro" y "Hamiltoniano de Watson"?**  
                                  * **Rotor Rígido Puro:** Considera la molécula como una estructura indeformable caracterizada únicamente por $(A, B, C)$ con constantes de distorsión iguales a cero ($\Delta_J = \Delta_{JK} = \Delta_K = \delta_J = \delta_K = 0$). A las bajas temperaturas del jet supersónico ($T_{\text{rot}} \approx 2 - 4\text{ K}$, $J \le 25$), la fuerza centrífuga es despreciable, por lo que programas clásicos como **JB95** resuelven el Carbazol con el modelo de Rotor Rígido.
                                  * **Hamiltoniano de Watson (Reducción A):** Añade los 5 términos cuárticos de distorsión centrífuga para corregir el estiramiento de enlaces cuando se pueblan niveles $J$ muy altos.
                                """
                            )

            # =========================================================================
            # TAB 2: ASIGNACIÓN AUTOMÁTICA DE LÍNEAS (.asc / JB95) & AJUSTE EVOLUTIVO
            # =========================================================================
            with gr.TabItem("🎯 2. Asignación Automática de Líneas (.asc / JB95) & Ajuste Evolutivo", id="tab_fit"):
                gr.Markdown(
                    r"""
                    ### A. Asignador Automático de Líneas Cuánticas (`A`, `B`, `C` $J' K_a' K_c' \leftarrow J'' K_a'' K_c''$) desde Espectrómetro Crudo (`PmtFla`)
                    En programas clásicos como **JB95**, el usuario debía hacer clic manualmente pico por pico sobre el archivo crudo `PmtFla` para asignarle sus números cuánticos y generar el archivo `.asc` (`IAR.asc`).  
                    Aquí **SpecWatson** detecta automáticamente los máximos experimentales con interpolación parabólica sub-MHz, los empareja con las transiciones cuánticas teóricas (`B J' Ka' Kc' ← J'' Ka'' Kc''`), calcula el error residual ($\nu_{\text{exp}} - \nu_{\text{calc}}$) y **genera el archivo `.asc` completo listo para descargar**.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        asg_source_dropdown = gr.Dropdown(
                            choices=[
                                "📡 Espectrómetro Real Pre-cargado: PmtFla (Carbazol - Dr. Leonardo)",
                                "📤 Usar Archivo Experimental Crudo Subido Abajo (.dat, PmtFla, .csv)"
                            ],
                            value="📡 Espectrómetro Real Pre-cargado: PmtFla (Carbazol - Dr. Leonardo)",
                            label="1. Archivo Experimental Crudo a Asignar"
                        )
                        asg_raw_upload = gr.File(
                            label="Opcional: Subir otro archivo crudo del espectrómetro (PmtFla, .dat, .csv)",
                            file_count="single"
                        )
                        with gr.Row():
                            asg_a0 = gr.Number(value=2253.2409, label="A″ Basal (MHz)")
                            asg_b0 = gr.Number(value=594.2103, label="B″ Basal (MHz)")
                            asg_c0 = gr.Number(value=470.3670, label="C″ Basal (MHz)")
                        with gr.Row():
                            asg_a1 = gr.Number(value=2250.6104, label="A′ Excitado (MHz)")
                            asg_b1 = gr.Number(value=586.3482, label="B′ Excitado (MHz)")
                            asg_c1 = gr.Number(value=465.2923, label="C′ Excitado (MHz)")

                        asg_origin = gr.Number(value=923604613.0295, label="Origen de Banda ν₀ (MHz) [Ej. 923604613.03 MHz]")

                        with gr.Row():
                            asg_band_mode = gr.Dropdown(
                                choices=BAND_TYPE_CHOICES[:3],
                                value=BAND_TYPE_CHOICES[0],
                                label="Criterio de Selección (Letra en .asc: B, A o C)"
                            )
                            asg_ham_mode = gr.Dropdown(
                                choices=HAMILTONIAN_CHOICES,
                                value=HAMILTONIAN_CHOICES[0],
                                label="Modelo Hamiltoniano"
                            )

                        with gr.Row():
                            asg_trot = gr.Slider(0.5, 12.0, value=2.6, step=0.1, label="T_rot (K)")
                            asg_fwhm = gr.Slider(5.0, 80.0, value=45.0, step=1.0, label="FWHM (MHz)")
                            asg_jmax = gr.Slider(6, 30, value=26, step=1, label="J Máximo")

                        with gr.Row():
                            asg_tol = gr.Slider(1.0, 20.0, value=5.5, step=0.5, label="Tolerancia de Asignación |ν_exp - ν_calc| (MHz)")
                            asg_min_pk = gr.Slider(2.0, 25.0, value=6.0, step=1.0, label="Altura Mín. de Pico Exp (%)")

                        asg_refine_chk = gr.Checkbox(
                            value=False,
                            label="🔧 Refinar (A′, B′, C′, ν₀) por Mínimos Cuadrados usando las líneas asignadas"
                        )
                        btn_run_assign = gr.Button(
                            "🎯 Asignar Líneas Cuánticas Automáticamente y Generar Archivo .asc (JB95)",
                            variant="primary",
                            size="lg"
                        )
                        asg_file_out = gr.File(
                            label="📥 Descargar Archivo .asc Generado (Formato Exacto IAR.asc / JB95)"
                        )

                    with gr.Column(scale=2):
                        asg_plot_out = gr.Plot(
                            label="Vista Espejo JCP (Experimental Arriba +Y / Simulado y Líneas Stick Asignadas Abajo -Y) + Residuales"
                        )
                        asg_report_out = gr.Textbox(
                            label="Resumen Estadístico de la Asignación Cuántica Automática",
                            lines=6
                        )
                        asg_preview_out = gr.Textbox(
                            label="Vista Previa del Archivo .asc Generado (Formato Exacto JB95: Tipo J' Ka' Kc' J'' Ka'' Kc'' Int Freq_Exp Freq_Calc Residual)",
                            lines=9
                        )
                        asg_table_out = gr.Dataframe(
                            headers=["Tipo", "Nivel Sup (J' Ka' Kc')", "Nivel Inf (J'' Ka'' Kc'')", "Rama", "Int (JB95)", "Freq Exp (MHz)", "Freq Calc (MHz)", "Residual (MHz)"],
                            label="Tabla Interactiva de Transiciones Cuánticas Asignadas"
                        )

                gr.Markdown(
                    """
                    ---
                    ### B. Ajuste Autónomo de Contorno por Evolución Diferencial (Sin Asignación de Líneas)
                    Prueba cómo el algoritmo híbrido de **Evolución Diferencial + Simplex** deduce las constantes del estado excitado $(A', B', C')$ partiendo de valores iniciales erróneos sobre el contorno completo.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        fit_mol = gr.Dropdown(
                            choices=list(PRESETS.keys())[:9],
                            value="Carbazol (Álvarez-Valtierra & Pratt 2006 | Datos Reales PmtFla & IAR.asc)",
                            label="1. Molécula Objetivo (Incluye Carbazol Dr. Leonardo)"
                        )
                        fit_band_dropdown = gr.Dropdown(
                            choices=BAND_TYPE_CHOICES,
                            value=BAND_TYPE_CHOICES[0],
                            label="2. Criterio de Selección Espectral (Tipo a / b / c / Híbrido)"
                        )
                        fit_ham_dropdown = gr.Dropdown(
                            choices=HAMILTONIAN_CHOICES,
                            value=HAMILTONIAN_CHOICES[0],
                            label="3. Modelo Hamiltoniano durante el Ajuste"
                        )
                        fit_noise = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Ruido Experimental Gaussiano (%)")
                        fit_iter = gr.Slider(5, 30, value=12, step=1, label="Generaciones Evolutivas (Iteraciones)")
                        btn_fit = gr.Button("🚀 Ejecutar Ajuste Evolutivo de Contorno", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        fit_plot_out = gr.Plot(label="Convergencia del Contorno en Modo Espejo y Espectro de Residuales")
                        fit_report_out = gr.Textbox(label="Parámetros Moleculares Recuperados vs. Referencia Experimental", lines=7)

            # =========================================================================
            # TAB 3: QUÍMICA CUÁNTICA 3D & DIBUJO MOLECULAR (TENSOR DE INERCIA -> A,B,C)
            # =========================================================================
            with gr.TabItem("🧪 3. Constructor Molecular 3D & Química Cuántica (GPU)", id="tab_qchem"):
                gr.Markdown(
                    r"""
                    ### Del Dibujo Molecular o Nombre Químico a las Constantes Rotacionales (A, B, C), Factor κ y Dipolo (μ)
                    Dibuja una molécula, elige una del catálogo (incluyendo **Carbazol, H₂O, CO₂, SO₂, H₂CO, Indol, Resveratrol**) o escribe su **nombre químico / código SMILES**. El motor optimiza la geometría 3D, diagonaliza el **Tensor de Inercia Isotópico**, obtiene **(A, B, C)**, el **Parámetro de Asimetría de Ray ($\kappa$)** y el **Momento Dipolar ($\vec{\mu}$)**, validándolos contra literatura experimental.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        q_catalog = gr.Dropdown(
                            choices=list(MOLECULE_CATALOG.keys()),
                            value="Carbazol (C12H9N - Aromático Tricíclico | Pratt & Álvarez-Valtierra)",
                            label="1. Catálogo de Moléculas (Aromáticos LIF + Sencillas HITRAN/JPL)"
                        )
                        q_input = gr.Textbox(
                            value="c1ccc2c(c1)[nH]c3ccccc23",
                            label="2. Código SMILES o Nombre (ej. Carbazol, Agua, CO2, SO2, Formaldehído, Indol)",
                            placeholder="Escribe nombre químico o pega el SMILES aquí..."
                        )
                        q_engine = gr.Dropdown(
                            choices=[
                                "RDKit ETKDGv3 + MMFF94s (Instantáneo ~0.5 s)",
                                "IA Cuántica GPU: TorchANI ANI-2x (Potencial Neuronal ωB97X)",
                                "Ab Initio DFT: PySCF / GPU4PySCF (B3LYP/STO-3G)"
                            ],
                            value="RDKit ETKDGv3 + MMFF94s (Instantáneo ~0.5 s)",
                            label="3. Motor de Optimización Cuántica / Geometría 3D"
                        )
                        with gr.Row():
                            q_confs = gr.Slider(1, 30, value=10, step=1, label="Búsqueda Conformacional (N Confórmeros)")
                            q_s1_exp = gr.Slider(0.2, 3.0, value=1.2, step=0.1, label="Expansión Aromática S₁ π→π* (%)")

                        btn_qchem = gr.Button("🧪 Optimizar Geometría 3D y Calcular Constantes (A, B, C, κ, μ)", variant="primary", size="lg")
                        btn_send_sim = gr.Button("⚡ Enviar (A, B, C, μ) Calculados al Simulador de Espectros (Pestaña 1)", variant="secondary")

                        with gr.Accordion("✏️ Pizarra Interactiva 2D: Dibuja Enlaces y Anillos Manualmente", open=False):
                            gr.Markdown("*Dibuja o modifica la molécula abajo, presiona **Copiar SMILES** y pégalo en la casilla 2:*")
                            gr.HTML(value=render_jsme_sketcher_html())

                    with gr.Column(scale=2):
                        q_plot_out = gr.Plot(label="Proyección en Ejes Principales de Inercia (a, b) y Constantes Rotacionales")
                        q_3d_out = gr.HTML(label="Visor Molecular 3D Interactivo (WebGL) con Ejes Principales a, b, c y Dipolo μ")
                        q_report_out = gr.Textbox(label="Reporte del Tensor de Inercia, Factor κ, Dipolo μ y Validación Experimental", lines=10)

                        with gr.Row(visible=False):
                            calc_a0 = gr.Number(value=2253.24)
                            calc_b0 = gr.Number(value=594.21)
                            calc_c0 = gr.Number(value=470.37)
                            calc_a1 = gr.Number(value=2250.61)
                            calc_b1 = gr.Number(value=586.35)
                            calc_c1 = gr.Number(value=465.29)
                            calc_mua = gr.Number(value=0.0)
                            calc_mub = gr.Number(value=1.0)
                            calc_muc = gr.Number(value=0.0)

        # Callbacks for Tab 1
        preset_outputs = [
            a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax,
            band_type_dropdown, hamiltonian_dropdown, exp_source_dropdown
        ]
        view_controls = [
            show_subbands_chk, show_sim_chk, overlay_exp_chk, mirror_exp_chk,
            alpha_sim_slider, alpha_exp_slider, offset_exp_slider,
            zoom_x_slider, pan_x_slider, zoom_y_slider, pan_y_slider
        ]
        all_sim_inputs = [
            a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax,
            preset_dropdown,
            band_type_dropdown, hamiltonian_dropdown,
            dJ_in, dJK_in, dK_in, delJ_in, delK_in,
            exp_source_dropdown, exp_file_upload,
            show_subbands_chk, show_sim_chk, overlay_exp_chk, mirror_exp_chk,
            alpha_sim_slider, alpha_exp_slider, offset_exp_slider,
            zoom_x_slider, pan_x_slider, zoom_y_slider, pan_y_slider
        ]

        preset_dropdown.change(
            fn=load_preset_values,
            inputs=[preset_dropdown],
            outputs=preset_outputs
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        band_type_dropdown.change(
            fn=on_band_mode_change,
            inputs=[band_type_dropdown],
            outputs=[mu_a, mu_b, mu_c]
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        hamiltonian_dropdown.change(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        exp_source_dropdown.change(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        btn_load_exp.click(
            fn=handle_uploaded_exp_file,
            inputs=[
                exp_file_upload,
                a0, b0, c0, a1, b1, c1,
                band_type_dropdown, hamiltonian_dropdown, fwhm
            ],
            outputs=[
                a0, b0, c0, a1, b1, c1,
                band_type_dropdown, hamiltonian_dropdown, exp_source_dropdown,
                upload_status_out
            ]
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        for ctrl in view_controls:
            ctrl.change(
                fn=simulate_rovibronic_spectrum,
                inputs=all_sim_inputs,
                outputs=[plot_out, stats_out]
            )

        btn_reset_view.click(
            fn=reset_view_controls,
            inputs=[],
            outputs=view_controls
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        btn_export_dpi.click(
            fn=export_spectrum_high_dpi,
            inputs=all_sim_inputs + [export_dpi_dropdown],
            outputs=[export_file_out]
        )

        btn_sim.click(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        # Jump from Tab 1 to Tab 2 Automatic Line Assigner and run assignment
        asg_inputs = [
            asg_source_dropdown, asg_raw_upload,
            asg_a0, asg_b0, asg_c0,
            asg_a1, asg_b1, asg_c1,
            asg_origin,
            asg_trot, asg_fwhm, asg_jmax,
            asg_band_mode, asg_ham_mode,
            asg_tol, asg_min_pk, asg_refine_chk
        ]
        asg_outputs = [asg_plot_out, asg_report_out, asg_preview_out, asg_table_out, asg_file_out]

        btn_goto_assign.click(
            fn=lambda x0, y0, z0, x1, y1, z1: (x0, y0, z0, x1, y1, z1, gr.Tabs(selected="tab_fit")),
            inputs=[a0, b0, c0, a1, b1, c1],
            outputs=[asg_a0, asg_b0, asg_c0, asg_a1, asg_b1, asg_c1, main_tabs]
        ).then(
            fn=run_automatic_line_assignment_ui,
            inputs=asg_inputs,
            outputs=asg_outputs
        )

        # Callbacks for Tab 2
        btn_run_assign.click(
            fn=run_automatic_line_assignment_ui,
            inputs=asg_inputs,
            outputs=asg_outputs
        )

        btn_fit.click(
            fn=run_autonomous_fit_demo,
            inputs=[fit_mol, fit_band_dropdown, fit_ham_dropdown, fit_noise, fit_iter],
            outputs=[fit_plot_out, fit_report_out]
        )

        # Callbacks for Tab 3
        qchem_outputs = [
            q_plot_out, q_3d_out, q_report_out,
            calc_a0, calc_b0, calc_c0,
            calc_a1, calc_b1, calc_c1,
            calc_mua, calc_mub, calc_muc
        ]

        q_catalog.change(
            fn=select_catalog_molecule,
            inputs=[q_catalog],
            outputs=[q_input]
        ).then(
            fn=run_quantum_3d_tab,
            inputs=[q_input, q_engine, q_confs, q_s1_exp],
            outputs=qchem_outputs
        )

        btn_qchem.click(
            fn=run_quantum_3d_tab,
            inputs=[q_input, q_engine, q_confs, q_s1_exp],
            outputs=qchem_outputs
        )

        def transfer_to_tab1(ca0, cb0, cc0, ca1, cb1, cc1, cmua, cmub, cmuc):
            if cmub >= 0.90:
                bmode = BAND_TYPE_CHOICES[0]
            elif cmua >= 0.90:
                bmode = BAND_TYPE_CHOICES[1]
            elif cmuc >= 0.90:
                bmode = BAND_TYPE_CHOICES[2]
            else:
                bmode = BAND_TYPE_CHOICES[3]
            return ca0, cb0, cc0, ca1, cb1, cc1, cmua, cmub, cmuc, bmode, True, False, gr.Tabs(selected="tab_sim")

        btn_send_sim.click(
            fn=transfer_to_tab1,
            inputs=[calc_a0, calc_b0, calc_c0, calc_a1, calc_b1, calc_c1, calc_mua, calc_mub, calc_muc],
            outputs=[a0, b0, c0, a1, b1, c1, mu_a, mu_b, mu_c, band_type_dropdown, show_sim_chk, overlay_exp_chk, main_tabs]
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        # Run initial simulations on page load (Tab 1, Tab 2 Auto-Assigner, and Tab 3)
        demo.load(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        ).then(
            fn=run_automatic_line_assignment_ui,
            inputs=asg_inputs,
            outputs=asg_outputs
        ).then(
            fn=run_quantum_3d_tab,
            inputs=[q_input, q_engine, q_confs, q_s1_exp],
            outputs=qchem_outputs
        )

        gr.Markdown(
            """
            ---
            *SpecWatson es software científico desarrollado en la División de Ciencias e Ingenierías (DCI-UG, Campus León).*  
            *Agradecimientos: Apoyo a través del programa de Ayudantía de Investigador SNII Nivel III del **Dr. Alejandro Gil-Villegas**.*
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
