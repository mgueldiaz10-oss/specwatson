"""
SpecWatson Interactive Web GUI (Gradio)
=======================================
Cloud / Local Interactive Interface for High-Resolution Rovibronic Spectroscopy:
  1. Direct Quantum Spectrum Simulator (Watson A-Reduced Hamiltonian + Wang Factorization)
  2. Autonomous Contour Fitter (Differential Evolution + Nelder-Mead Simplex)
  3. 3D Quantum Chemistry & Molecular Sketcher (RDKit / TorchANI GPU / GPU4PySCF -> Inertia Tensor -> A, B, C)

Author: Miguel Díaz (Universidad de Guanajuato - DCI León)
Advisor: Dr. Leonardo Álvarez-Valtierra
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator
from specwatson.spectrum import SpectrumSimulator
from specwatson.fitting import ContourFitter
from specwatson.quantum_chem import (
    MOLECULE_CATALOG,
    run_quantum_3d_pipeline,
    plot_molecular_pas_figure,
    render_3dmol_pas_html,
    render_jsme_sketcher_html,
)

# Pre-configured molecular presets for Tab 1
PRESETS = {
    "Carbazol (Álvarez-Valtierra & Pratt 2006)": {
        "A0": 2178.60, "B0": 1251.20, "C0": 794.50,
        "A1": 2145.30, "B1": 1242.10, "C1": 788.90,
        "Trot": 3.5, "fwhm": 25.0,
        "mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0,
        "Jmax": 12
    },
    "trans-Resveratrol (Polifenol - Confórmero I Plano)": {
        "A0": 1684.50, "B0": 218.40, "C0": 193.35,
        "A1": 1652.10, "B1": 216.85, "C1": 191.80,
        "Trot": 3.0, "fwhm": 18.0,
        "mu_a": 0.82, "mu_b": 0.18, "mu_c": 0.0,
        "Jmax": 14
    },
    "trans-Resveratrol (Polifenol - Confórmero II Torsional)": {
        "A0": 1592.30, "B0": 229.15, "C0": 204.80,
        "A1": 1564.80, "B1": 227.40, "C1": 203.10,
        "Trot": 3.0, "fwhm": 18.0,
        "mu_a": 0.65, "mu_b": 0.25, "mu_c": 0.10,
        "Jmax": 14
    },
    "1-Naftol (Banda Híbrida a/b en Jet Supersónico)": {
        "A0": 1948.50, "B0": 1132.40, "C0": 716.20,
        "A1": 1912.10, "B1": 1118.80, "C1": 706.50,
        "Trot": 4.0, "fwhm": 22.0,
        "mu_a": 0.60, "mu_b": 0.40, "mu_c": 0.0,
        "Jmax": 12
    }
}


def simulate_rovibronic_spectrum(
    A0, B0, C0,
    A1, B1, C1,
    T_rot, fwhm_mhz,
    mu_a, mu_b, mu_c,
    J_max
):
    """Callback for Tab 1: Direct Quantum Spectrum Simulation."""
    t0 = time.perf_counter()
    plt.close("all")

    try:
        A0, B0, C0 = float(A0), float(B0), float(C0)
        A1, B1, C1 = float(A1), float(B1), float(C1)
        T_rot = max(0.2, float(T_rot))
        fwhm_mhz = max(2.0, float(fwhm_mhz))
        J_max = max(2, min(int(J_max), 25))

        if not (A0 >= B0 >= C0 > 0) or not (A1 >= B1 >= C1 > 0):
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
            ax.text(
                0.5, 0.5,
                "⚠️ Restricción Física: Se requiere A ≥ B ≥ C > 0 MHz en ambos estados.",
                ha="center", va="center", fontsize=12, color="#dc2626", fontweight="bold"
            )
            ax.axis("off")
            return fig, "Error de validación física: verifica que A ≥ B ≥ C > 0 MHz."

        state_s0 = MoleculeState(
            name="S0", A=A0, B=B0, C=C0,
            Delta_J=0.00025, Delta_JK=-0.00060, Delta_K=0.00120
        )
        state_s1 = MoleculeState(
            name="S1", A=A1, B=B1, C=C1,
            Delta_J=0.00024, Delta_JK=-0.00058, Delta_K=0.00115,
            origin=30000.0
        )

        ham_s0 = AsymmetricRotorHamiltonian(state_s0)
        ham_s1 = AsymmetricRotorHamiltonian(state_s1)

        manifold_s0 = ham_s0.compute_manifold(J_max=J_max)
        manifold_s1 = ham_s1.compute_manifold(J_max=J_max)

        dip_sum = float(mu_a) + float(mu_b) + float(mu_c)
        if dip_sum <= 1e-6:
            dip = (1.0, 0.0, 0.0)
        else:
            dip = (float(mu_a) / dip_sum, float(mu_b) / dip_sum, float(mu_c) / dip_sum)

        calc = RovibronicTransitionCalculator(
            ground_manifold=manifold_s0,
            excited_manifold=manifold_s1,
            band_origin_cm1=30000.0,
            temperature_K=T_rot,
            dipole_fractions=dip
        )
        lines = calc.calculate_transitions(J_max_ground=J_max, J_max_excited=J_max)

        sim = SpectrumSimulator(lines=lines, fwhm_mhz=fwhm_mhz, profile="voigt")
        freq_axis, intensity = sim.simulate(grid_points=2500)

        origin_mhz = 30000.0 * 29979.2458
        rel_freq_ghz = (freq_axis - origin_mhz) / 1000.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        kappa_s0 = state_s0.kappa
        kappa_s1 = state_s1.kappa

        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=130)
        ax.plot(rel_freq_ghz, intensity, color="#1d4ed8", lw=1.35, label="Contorno Voigt Convolucionado")
        ax.fill_between(rel_freq_ghz, intensity, color="#3b82f6", alpha=0.16)
        ax.axvline(0.0, color="#ef4444", ls="--", lw=0.9, alpha=0.7, label="Origen de Banda ν₀")

        ax.set_title(
            f"Espectro Rovibrónico S₁ ← S₀ (J_max={J_max}, T_rot={T_rot:.1f} K, FWHM={fwhm_mhz:.0f} MHz)",
            fontsize=11.5, fontweight="bold", pad=10
        )
        ax.set_xlabel("Desplazamiento de Frecuencia Relativa ν - ν₀ (GHz)", fontsize=10.5)
        ax.set_ylabel("Intensidad Normalizada (u.a.)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

        stats = (
            f"✓ Simulación cuántica completada en {elapsed_ms:.1f} ms\n"
            f"• Transiciones rovibrónicas activas evaluadas: {len(lines):,}\n"
            f"• Parámetro de asimetría de Ray: κ(S₀) = {kappa_s0:.4f}  |  κ(S₁) = {kappa_s1:.4f}\n"
            f"• Cambio rotacional al excitar: ΔA = {A1 - A0:+.2f} MHz, ΔB = {B1 - B0:+.2f} MHz, ΔC = {C1 - C0:+.2f} MHz\n"
            f"• Pesos de selección dipolar: μₐ²={dip[0]:.2f}, μ_b²={dip[1]:.2f}, μ_c²={dip[2]:.2f} ( bloques de Wang E⁺, E⁻, O⁺, O⁻ )"
        )
        return fig, stats

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en simulación: {str(e)}", ha="center", va="center", color="red")
        ax.axis("off")
        return fig, f"Error detallado: {repr(e)}"


def run_autonomous_fit_demo(molecule_choice, noise_pct, max_iter):
    """Callback for Tab 2: Live Differential Evolution + Simplex Contour Fitting."""
    t0 = time.perf_counter()
    plt.close("all")

    try:
        if "Resveratrol" in molecule_choice:
            p = PRESETS["trans-Resveratrol (Polifenol - Confórmero I Plano)"]
            guess_A, guess_B, guess_C = p["A1"] - 25.0, p["B1"] + 6.0, p["C1"] - 5.0
            bounds = [
                (p["A1"] - 35.0, p["A1"] + 35.0),
                (p["B1"] - 15.0, p["B1"] + 15.0),
                (p["C1"] - 15.0, p["C1"] + 15.0),
                (-15.0, 15.0)
            ]
            j_fit = 7
        else:
            p = PRESETS["Carbazol (Álvarez-Valtierra & Pratt 2006)"]
            guess_A, guess_B, guess_C = 2110.0, 1260.0, 775.0
            bounds = [(2100.0, 2190.0), (1210.0, 1280.0), (760.0, 820.0), (-20.0, 20.0)]
            j_fit = 7

        s0 = MoleculeState("S0", A=p["A0"], B=p["B0"], C=p["C0"])
        s1_true = MoleculeState("S1_true", A=p["A1"], B=p["B1"], C=p["C1"], origin=30000.0)
        s1_guess = MoleculeState("S1_guess", A=guess_A, B=guess_B, C=guess_C, origin=30000.0)

        g_man = AsymmetricRotorHamiltonian(s0).compute_manifold(j_fit)
        e_man_true = AsymmetricRotorHamiltonian(s1_true).compute_manifold(j_fit)
        dip = (p["mu_a"], p["mu_b"], p["mu_c"])

        calc_true = RovibronicTransitionCalculator(g_man, e_man_true, 30000.0, p["Trot"], dip)
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
            dipole_fractions=dip
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

        origin_mhz = 30000.0 * 29979.2458
        rel_f = freq_axis - origin_mhz

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10.5, 5.5), dpi=130,
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        ax1.plot(rel_f, y_exp, color="#1e293b", lw=1.2, alpha=0.85, label=f"Experimental Sintético ({noise_pct}% ruido)")
        ax1.plot(rel_f, y_init, color="#f59e0b", ls=":", lw=1.3, label=f"Semilla Inicial Ab Initio (A′={guess_A:.0f} MHz)")
        ax1.plot(rel_f, y_fit, color="#ef4444", ls="--", lw=1.8, label=f"Ajuste Híbrido DE + Simplex (r = {corr_pct:.2f}%)")
        ax1.set_ylabel("Intensidad Normalizada", fontsize=10)
        ax1.set_title(f"Ajuste de Contorno por Evolución Diferencial: {molecule_choice}", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
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

        planarity_note = (
            "Molécula PLANA en el plano ab (ΔI ≈ 0)"
            if abs(res.inertial_defect) < 0.8
            else "Molécula NO PLANA / Con estructura 3D fuera del plano"
        )

        report = (
            f"✓ Geometría 3D y Tensor de Inercia calculados en {res.elapsed_sec*1000:.1f} ms\n"
            f"• Fórmula Molecular: {res.formula} ({res.num_atoms} átomos)  |  SMILES: {res.smiles}\n"
            f"• Motor Cuántico: {res.engine_used}\n"
            f"• Hardware Detectado: {res.device_used}  |  Energía: {res.energy_kcal_or_eh}\n"
            f"• Momentos Principales de Inercia: Ia = {res.Ia:.2f}, Ib = {res.Ib:.2f}, Ic = {res.Ic:.2f} u·Å²\n"
            f"• Defecto de Inercia Cuántico: ΔI = Ic - Ia - Ib = {res.inertial_defect:+.4f} u·Å² ({planarity_note})\n"
            f"• Constantes Estado Basal (S₀): A″ = {res.A0_mhz:.2f} MHz, B″ = {res.B0_mhz:.2f} MHz, C″ = {res.C0_mhz:.2f} MHz (κ = {res.kappa:.4f})\n"
            f"• Constantes Estado Excitado Est. (S₁): A′ = {res.A1_mhz:.2f} MHz, B′ = {res.B1_mhz:.2f} MHz, C′ = {res.C1_mhz:.2f} MHz\n"
            f"• Momento Dipolar: |μ| = {res.dipole_total_debye:.2f} D  →  Fracciones: μₐ²={res.mu_a_frac:.2f}, μ_b²={res.mu_b_frac:.2f}, μ_c²={res.mu_c_frac:.2f}"
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
            2178.60, 1251.20, 794.50,
            2145.30, 1242.10, 788.90,
            1.0, 0.0, 0.0
        )


def select_catalog_molecule(catalog_choice):
    info = MOLECULE_CATALOG.get(catalog_choice, MOLECULE_CATALOG["Carbazol (C12H9N - Aromático Tricíclico)"])
    return info["smiles"]


def load_preset_values(preset_name):
    p = PRESETS.get(preset_name, PRESETS["Carbazol (Álvarez-Valtierra & Pratt 2006)"])
    return (
        p["A0"], p["B0"], p["C0"],
        p["A1"], p["B1"], p["C1"],
        p["Trot"], p["fwhm"],
        p["mu_a"], p["mu_b"], p["mu_c"],
        p["Jmax"]
    )


def build_gradio_app():
    import gradio as gr

    with gr.Blocks(title="SpecWatson · Espectroscopía Rovibrónica & Química Cuántica GPU") as demo:
        gr.Markdown(
            """
            # 🔬 SpecWatson: Suite de Espectroscopía Rovibrónica & Química Cuántica en GPU
            **Universidad de Guanajuato · División de Ciencias e Ingenierías (Campus León)**  
            **Desarrollado por:** Miguel Díaz (Maestría en Ciencias) · **Director:** Dr. Leonardo Álvarez-Valtierra  
            *De la estructura molecular 3D y el tensor de inercia al Hamiltoniano de Watson y el ajuste evolutivo de espectros láser.*
            """
        )

        with gr.Tabs() as main_tabs:
            # =========================================================================
            # TAB 1: SIMULACIÓN DIRECTA DE ESPECTROS (WANG)
            # =========================================================================
            with gr.TabItem("⚡ 1. Simulador Cuántico de Espectros (Wang)", id="tab_sim"):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Carbazol (Álvarez-Valtierra & Pratt 2006)",
                        label="📁 Seleccionar Sistema Molecular del Laboratorio (Preset)"
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Estado Basal (S₀) - MHz")
                        with gr.Row():
                            a0 = gr.Number(value=2178.60, label="A″ Basal (MHz)")
                            b0 = gr.Number(value=1251.20, label="B″ Basal (MHz)")
                            c0 = gr.Number(value=794.50, label="C″ Basal (MHz)")

                        gr.Markdown("### 2. Estado Excitado (S₁) - MHz")
                        with gr.Row():
                            a1 = gr.Number(value=2145.30, label="A′ Excitado (MHz)")
                            b1 = gr.Number(value=1242.10, label="B′ Excitado (MHz)")
                            c1 = gr.Number(value=788.90, label="C′ Excitado (MHz)")

                        gr.Markdown("### 3. Condiciones Experimentales del Jet")
                        with gr.Row():
                            trot = gr.Slider(0.5, 10.0, value=3.5, step=0.1, label="Temp. Rotacional (K)")
                            fwhm = gr.Slider(5.0, 50.0, value=25.0, step=1.0, label="Ancho Láser FWHM (MHz)")
                            jmax = gr.Slider(4, 22, value=12, step=1, label="J Máximo")

                        gr.Markdown("### 4. Componentes Dipolares (Banda Híbrida)")
                        with gr.Row():
                            mu_a = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="μₐ² (tipo a)")
                            mu_b = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μ_b² (tipo b)")
                            mu_c = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μ_c² (tipo c)")

                        btn_sim = gr.Button("⚡ Simular Espectro Cuántico", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        plot_out = gr.Plot(label="Espectro Simulado en Tiempo Real")
                        stats_out = gr.Textbox(label="Diagnóstico Cuántico & Métricas de Wang", lines=6)

            # =========================================================================
            # TAB 2: AJUSTE AUTÓNOMO POR EVOLUCIÓN DIFERENCIAL
            # =========================================================================
            with gr.TabItem("🧬 2. Ajuste Autónomo de Contorno (Evolución Diferencial)", id="tab_fit"):
                gr.Markdown(
                    """
                    ### Motor de Problema Inverso sin Asignación Manual
                    Prueba cómo el algoritmo híbrido de **Evolución Diferencial + Simplex** deduce las constantes del estado excitado (A′, B′, C′) partiendo de valores iniciales erróneos sobre un espectro con ruido experimental.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        fit_mol = gr.Dropdown(
                            choices=[
                                "Carbazol (Álvarez-Valtierra & Pratt 2006)",
                                "trans-Resveratrol (Polifenol - Confórmero I Plano)"
                            ],
                            value="Carbazol (Álvarez-Valtierra & Pratt 2006)",
                            label="Molécula Objetivo"
                        )
                        fit_noise = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Ruido Experimental Gaussiano (%)")
                        fit_iter = gr.Slider(5, 30, value=12, step=1, label="Generaciones Evolutivas (Iteraciones)")
                        btn_fit = gr.Button("🚀 Ejecutar Ajuste Evolutivo Autónomo", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        fit_plot_out = gr.Plot(label="Convergencia del Contorno y Espectro de Residuales")
                        fit_report_out = gr.Textbox(label="Parámetros Moleculares Recuperados", lines=6)

            # =========================================================================
            # TAB 3: QUÍMICA CUÁNTICA 3D & DIBUJO MOLECULAR (TENSOR DE INERCIA -> A,B,C)
            # =========================================================================
            with gr.TabItem("🧪 3. Constructor Molecular 3D & Química Cuántica (GPU)", id="tab_qchem"):
                gr.Markdown(
                    """
                    ### Del Dibujo Molecular o Nombre Químico a las Constantes Rotacionales (A, B, C) en Segundos
                    Dibuja una molécula, elige una del catálogo o escribe su **nombre químico / código SMILES**. El motor construye la estructura 3D, optimiza su geometría mediante **Campos de Fuerza (`MMFF94s`)**, **Potenciales Neuronales Cuánticos en GPU (`ANI-2x / ωB97X`)** o **DFT Ab Initio (`PySCF / GPU4PySCF`)**, diagonaliza el **Tensor de Inercia Isotópico** y obtiene automáticamente **(A, B, C)**, el **Defecto de Inercia (ΔI)** y los **Ejes Principales (a, b, c)**.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        q_catalog = gr.Dropdown(
                            choices=list(MOLECULE_CATALOG.keys()),
                            value="Carbazol (C12H9N - Aromático Tricíclico)",
                            label="1. Catálogo Rápido de Moléculas Espectroscópicas"
                        )
                        q_input = gr.Textbox(
                            value="c1ccc2c(c1)[nH]c3ccccc23",
                            label="2. Código SMILES o Nombre de la Molécula (ej. Indole, Caffeine, Aspirin, Fluorene)",
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

                        btn_qchem = gr.Button("🧪 Optimizar Geometría 3D y Calcular Constantes (A, B, C)", variant="primary", size="lg")
                        btn_send_sim = gr.Button("⚡ Enviar (A, B, C, μ) Calculados al Simulador de Espectros (Pestaña 1)", variant="secondary")

                        with gr.Accordion("✏️ Pizarra Interactiva 2D: Dibuja Enlaces y Anillos Manualmente", open=False):
                            gr.Markdown("*Dibuja o modifica la molécula abajo, presiona **Copiar SMILES** y pégalo en la casilla 2:*")
                            gr.HTML(value=render_jsme_sketcher_html())

                    with gr.Column(scale=2):
                        q_plot_out = gr.Plot(label="Proyección en Ejes Principales de Inercia (a, b) y Constantes Rotacionales")
                        q_3d_out = gr.HTML(label="Visor Molecular 3D Interactivo (WebGL) con Ejes Principales a, b, c y Dipolo μ")
                        q_report_out = gr.Textbox(label="Reporte del Tensor de Inercia, Defecto Cuántico ΔI y Constantes (MHz)", lines=8)

                        # Hidden state boxes holding the computed constants so they can be transferred to Tab 1
                        with gr.Row(visible=False):
                            calc_a0 = gr.Number(value=2178.60)
                            calc_b0 = gr.Number(value=1251.20)
                            calc_c0 = gr.Number(value=794.50)
                            calc_a1 = gr.Number(value=2145.30)
                            calc_b1 = gr.Number(value=1242.10)
                            calc_c1 = gr.Number(value=788.90)
                            calc_mua = gr.Number(value=1.0)
                            calc_mub = gr.Number(value=0.0)
                            calc_muc = gr.Number(value=0.0)

        # Callbacks for Tab 1
        all_sim_inputs = [a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax]

        preset_dropdown.change(
            fn=load_preset_values,
            inputs=[preset_dropdown],
            outputs=all_sim_inputs
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        btn_sim.click(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        # Callbacks for Tab 2
        btn_fit.click(
            fn=run_autonomous_fit_demo,
            inputs=[fit_mol, fit_noise, fit_iter],
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

        # Transfer calculated (A, B, C, mu) from Tab 3 directly to Tab 1 and simulate spectrum!
        def transfer_to_tab1(ca0, cb0, cc0, ca1, cb1, cc1, cmua, cmub, cmuc):
            return ca0, cb0, cc0, ca1, cb1, cc1, cmua, cmub, cmuc, gr.Tabs(selected="tab_sim")

        btn_send_sim.click(
            fn=transfer_to_tab1,
            inputs=[calc_a0, calc_b0, calc_c0, calc_a1, calc_b1, calc_c1, calc_mua, calc_mub, calc_muc],
            outputs=[a0, b0, c0, a1, b1, c1, mu_a, mu_b, mu_c, main_tabs]
        ).then(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        )

        # Run initial simulations on page load
        demo.load(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
        ).then(
            fn=run_quantum_3d_tab,
            inputs=[q_input, q_engine, q_confs, q_s1_exp],
            outputs=qchem_outputs
        )

        gr.Markdown(
            """
            ---
            *SpecWatson es software científico de código abierto desarrollado en la Universidad de Guanajuato.*  
            Presentación y Artículos PDF: [mgueldiaz10-oss.github.io/specwatson](https://mgueldiaz10-oss.github.io/specwatson/)
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
