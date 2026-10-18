"""
SpecWatson GUI: High-Resolution Rovibronic Spectroscopy Suite
Powered by Wang Symmetrized Factorization & Differential Evolution
Author: Miguel Diaz & Dr. Leonardo Alvarez-Valtierra (Universidad de Guanajuato)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local specwatson package is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.wang import WangHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator
from specwatson.spectrum import SpectrumSimulator
from specwatson.fitting import ContourFitter

# Presets for quick lab demos
PRESETS = {
    "Carbazol (Álvarez-Valtierra & Pratt 2006)": {
        "A0": 2178.60, "B0": 1251.20, "C0": 794.50,
        "A1": 2145.30, "B1": 1242.10, "C1": 788.90,
        "Trot": 3.5, "fwhm": 25.0, "mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0,
        "Jmax": 12
    },
    "trans-Resveratrol (Polifenol - Confórmero I Plano)": {
        "A0": 1385.40, "B0": 186.20, "C0": 164.10,
        "A1": 1362.80, "B1": 184.90, "C1": 162.80,
        "Trot": 2.5, "fwhm": 20.0, "mu_a": 0.82, "mu_b": 0.18, "mu_c": 0.0,
        "Jmax": 10
    },
    "trans-Resveratrol (Confórmero II - No Plano / Rotámero 2)": {
        "A0": 1442.10, "B0": 179.50, "C0": 159.80,
        "A1": 1418.00, "B1": 178.00, "C1": 158.20,
        "Trot": 2.5, "fwhm": 20.0, "mu_a": 0.65, "mu_b": 0.35, "mu_c": 0.0,
        "Jmax": 10
    }
}


def simulate_rovibronic_spectrum(A0, B0, C0, A1, B1, C1, Trot, fwhm, mu_a, mu_b, mu_c, Jmax):
    """Computes quantum energy levels and convolves Voigt spectrum."""
    plt.close("all")
    try:
        t0 = time.perf_counter()
        state_s0 = MoleculeState(name="S0", A=float(A0), B=float(B0), C=float(C0))
        state_s1 = MoleculeState(name="S1", A=float(A1), B=float(B1), C=float(C1), origin=30000.0)

        j_limit = int(Jmax)

        # Benchmark Wang vs Full at J = j_limit
        wang_s0 = WangHamiltonian(state_s0)
        e_w, _ = wang_s0.solve_j_wang(j_limit)

        g_ham = AsymmetricRotorHamiltonian(state_s0)
        e_ham = AsymmetricRotorHamiltonian(state_s1)

        e_f, _, _ = g_ham.solve_j(j_limit)
        max_wang_err = float(np.max(np.abs(e_f - e_w)))

        g_man = g_ham.compute_manifold(j_limit)
        e_man = e_ham.compute_manifold(j_limit)

        calc = RovibronicTransitionCalculator(
            g_man, e_man,
            band_origin_cm1=30000.0,
            temperature_K=float(Trot),
            dipole_fractions=(float(mu_a), float(mu_b), float(mu_c))
        )
        lines = calc.calculate_transitions(j_limit, j_limit)

        if not lines:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No se encontraron transiciones permitidas (verifica que μ_a, μ_b o μ_c > 0)", ha="center")
            return fig, "Sin transiciones calculadas."

        origin_mhz = 30000.0 * 29979.2458
        sim_voigt = SpectrumSimulator(lines, fwhm_mhz=float(fwhm), profile="voigt", lorentz_fraction=0.35)
        grid_x, grid_y = sim_voigt.simulate(grid_points=2500)

        rel_x = grid_x - origin_mhz
        norm_y = grid_y / grid_y.max() if grid_y.max() > 0 else grid_y

        st_x = np.array([l.frequency_mhz for l in lines]) - origin_mhz
        st_y = np.array([l.intensity for l in lines])
        if st_y.max() > 0:
            st_y /= st_y.max()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Plotting
        fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=130)
        ax.vlines(st_x, 0, st_y * 0.88, color="#f59e0b", alpha=0.45, lw=0.8,
                  label=f"Transiciones discretas ({len(lines):,} líneas)")
        ax.plot(rel_x, norm_y, color="#0284c7", lw=1.8,
                label=f"Convolución Voigt (FWHM = {fwhm:.1f} MHz, T = {Trot:.1f} K)")

        ax.set_title("Espectro Electrónico Rovibrónico S₁ ← S₀ (SpecWatson)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frecuencia Relativa ν - ν₀ (MHz)", fontsize=11)
        ax.set_ylabel("Intensidad Normalizada (u.a.)", fontsize=11)
        ax.set_ylim(-0.03, 1.08)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()

        k_s0 = state_s0.kappa
        k_s1 = state_s1.kappa
        tipo_rotor = "Prolado (κ < 0)" if k_s0 < 0 else "Oblado (κ > 0)"

        stats_msg = (
            f"✓ Simulación cuántica completada en {elapsed_ms:.1f} ms\n"
            f"• Total de transiciones rovibrónicas activas: {len(lines):,}\n"
            f"• Rango espectral: [{rel_x.min():.1f}, {rel_x.max():.1f}] MHz\n"
            f"• Parámetro de asimetría de Ray: κ(S₀) = {k_s0:.4f}, κ(S₁) = {k_s1:.4f} [{tipo_rotor}]\n"
            f"• Precisión Factorización de Wang (J={j_limit}): Error máx = {max_wang_err:.3e} MHz\n"
            f"• Cambios rotacionales: ΔA = {float(A1)-float(A0):+.2f} MHz, ΔB = {float(B1)-float(B0):+.2f} MHz, ΔC = {float(C1)-float(C0):+.2f} MHz"
        )
        return fig, stats_msg

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en simulación: {str(e)}", ha="center", color="red")
        return fig, f"Error detallado: {repr(e)}"


def run_autonomous_fit_demo(molecule_choice, noise_pct, max_iter):
    """Runs a live Differential Evolution contour fit on synthetic noisy data."""
    plt.close("all")
    try:
        t0 = time.perf_counter()
        if "Resveratrol" in molecule_choice:
            p = PRESETS["trans-Resveratrol (Polifenol - Confórmero I Plano)"]
            guess_A, guess_B, guess_C = p["A1"] - 22.0, p["B1"] + 6.0, p["C1"] - 5.0
            bounds = [(p["A1"] - 35.0, p["A1"] + 35.0),
                      (p["B1"] - 15.0, p["B1"] + 15.0),
                      (p["C1"] - 15.0, p["C1"] + 15.0),
                      (-15.0, 15.0)]
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.5), dpi=130,
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        ax1.plot(rel_f, y_exp, color="#1e293b", lw=1.2, alpha=0.85, label=f"Experimental Sintético ({noise_pct}% ruido)")
        ax1.plot(rel_f, y_init, color="#f59e0b", ls=":", lw=1.3, label=f"Semilla Inicial Distorsionada (A′={guess_A:.0f} MHz)")
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
            f"✓ Ajuste por Evolución Diferencial completado en {elapsed_s:.2f} s\n"
            f"• Correlación Cruzada Alcanzada: {corr_pct:.2f}%\n"
            f"• A' Recuperada: {best_p[0]:.2f} MHz  (Valor Real: {p['A1']:.2f} MHz | Error: {abs(best_p[0]-p['A1']):.2f} MHz)\n"
            f"• B' Recuperada: {best_p[1]:.2f} MHz  (Valor Real: {p['B1']:.2f} MHz | Error: {abs(best_p[1]-p['B1']):.2f} MHz)\n"
            f"• C' Recuperada: {best_p[2]:.2f} MHz  (Valor Real: {p['C1']:.2f} MHz | Error: {abs(best_p[2]-p['C1']):.2f} MHz)"
        )
        return fig, report

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error en ajuste: {str(e)}", ha="center", color="red")
        return fig, f"Error detallado: {repr(e)}"


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

    with gr.Blocks(title="SpecWatson · Espectroscopía Rovibrónica de Alta Resolución") as demo:
        gr.Markdown(
            """
            # 🔬 SpecWatson: Suite de Espectroscopía Rovibrónica de Alta Resolución
            **Universidad de Guanajuato · División de Ciencias e Ingenierías (Campus León)**  
            **Desarrollado por:** Miguel Díaz · **Director:** Dr. Leonardo Álvarez-Valtierra  
            *Simulación directa de Hamiltonianos de Watson, factorización tridiagonal de Wang y ajuste de contorno automatizado.*
            """
        )

        with gr.Tabs():
            # TAB 1: SIMULACIÓN DIRECTA
            with gr.TabItem("⚡ 1. Simulador Cuántico de Espectros (Wang)"):
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

            # TAB 2: AJUSTE AUTÓNOMO POR EVOLUCIÓN DIFERENCIAL
            with gr.TabItem("🧬 2. Ajuste Autónomo de Contorno (Evolución Diferencial)"):
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

        # Run initial simulation automatically when page opens
        demo.load(
            fn=simulate_rovibronic_spectrum,
            inputs=all_sim_inputs,
            outputs=[plot_out, stats_out]
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
