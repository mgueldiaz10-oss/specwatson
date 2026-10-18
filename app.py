"""
SpecWatson GUI: High-Resolution Rovibronic Spectroscopy Suite
Powered by Wang Symmetrized Factorization & Differential Evolution
Author: Miguel Diaz & Dr. Leonardo Alvarez-Valtierra (Universidad de Guanajuato)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure local specwatson package is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator
from specwatson.spectrum import SpectrumSimulator

# Presets for quick lab demos
PRESETS = {
    "Carbazol (Álvarez-Valtierra & Pratt 2006)": {
        "A0": 2178.60, "B0": 1251.20, "C0": 794.50,
        "A1": 2145.30, "B1": 1242.10, "C1": 788.90,
        "Trot": 3.5, "fwhm": 25.0, "mu_a": 1.0, "mu_b": 0.0, "mu_c": 0.0,
        "Jmax": 12
    },
    "trans-Resveratrol (Polifenol - Confórmero I)": {
        "A0": 1385.40, "B0": 186.20, "C0": 164.10,
        "A1": 1362.80, "B1": 184.90, "C1": 162.80,
        "Trot": 2.5, "fwhm": 20.0, "mu_a": 0.905, "mu_b": 0.424, "mu_c": 0.0,
        "Jmax": 10
    },
    "trans-Resveratrol (Confórmero II - Rechazado)": {
        "A0": 1442.10, "B0": 179.50, "C0": 159.80,
        "A1": 1418.00, "B1": 178.00, "C1": 158.20,
        "Trot": 2.5, "fwhm": 20.0, "mu_a": 0.806, "mu_b": 0.591, "mu_c": 0.0,
        "Jmax": 10
    }
}

def simulate_rovibronic_spectrum(A0, B0, C0, A1, B1, C1, Trot, fwhm, mu_a, mu_b, mu_c, Jmax):
    """Computes quantum energy levels and convolves Voigt spectrum."""
    state_s0 = MoleculeState(name="S0", A=float(A0), B=float(B0), C=float(C0))
    state_s1 = MoleculeState(name="S1", A=float(A1), B=float(B1), C=float(C1), origin=30000.0)
    
    g_ham = AsymmetricRotorHamiltonian(state_s0)
    e_ham = AsymmetricRotorHamiltonian(state_s1)
    
    j_limit = int(Jmax)
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
        ax.text(0.5, 0.5, "No se encontraron transiciones permitidas", ha="center")
        return fig, "Sin transiciones calculadas."
    
    origin_mhz = 30000.0 * 29979.2458
    sim_voigt = SpectrumSimulator(lines, fwhm_mhz=float(fwhm), profile="voigt", lorentz_fraction=0.35)
    grid_x, grid_y = sim_voigt.simulate(grid_points=2500)
    
    rel_x = grid_x - origin_mhz
    if grid_y.max() > 0:
        norm_y = grid_y / grid_y.max()
    else:
        norm_y = grid_y
        
    st_x = np.array([l.frequency_mhz for l in lines]) - origin_mhz
    st_y = np.array([l.intensity for l in lines])
    if st_y.max() > 0:
        st_y /= st_y.max()
        
    # Plotting
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    ax.plot(rel_x, norm_y, color='#0284c7', lw=1.8, label=f'Convolución Voigt (FWHM = {fwhm} MHz)')
    ax.vlines(st_x, 0, st_y * 0.9, color='#f59e0b', alpha=0.45, lw=0.8, label=f'Líneas discretas ({len(lines)} transiciones)')
    
    ax.set_title("Espectro Electrónico Rovibrónico Simulado (SpecWatson)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frecuencia Relativa $\\nu - \\nu_0$ (MHz)", fontsize=11)
    ax.set_ylabel("Intensidad Fluorescencia Normalizada", fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    
    stats_msg = (
        f"✓ Cálculo cuántico exitoso:\n"
        f"• Total transiciones rovibrónicas: {len(lines):,}\n"
        f"• Rango espectral: [{rel_x.min():.1f}, {rel_x.max():.1f}] MHz\n"
        f"• Parámetro de Ray κ(S0): {state_s0.ray_asymmetry():.4f} (Prolado)\n"
        f"• Parámetro de Ray κ(S1): {state_s1.ray_asymmetry():.4f}\n"
        f"• Estado: Listo para ajuste de contorno o exportación."
    )
    return fig, stats_msg

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
    
    custom_css = """
    body { font-family: 'Inter', system-ui, sans-serif; }
    .gradio-container { max-width: 1200px !important; }
    """
    
    with gr.Blocks(title="SpecWatson · High-Resolution Rovibronic Spectroscopy", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔬 SpecWatson: Suite de Espectroscopía Rovibrónica de Alta Resolución
            ### Universidad de Guanajuato · División de Ciencias e Ingenierías (Campus León)
            **Desarrollado por:** Miguel Díaz · **Director:** Dr. Leonardo Álvarez-Valtierra  
            *Simulación directa de Hamiltonianos de Watson, factorización tridiagonal de Wang y ajuste de contorno automatizado.*
            """
        )
        
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Carbazol (Álvarez-Valtierra & Pratt 2006)",
                label="📁 Cargar Ejemplo del Laboratorio (Preset)"
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Estado Basal ($S_0$) - MHz")
                with gr.Row():
                    a0 = gr.Number(value=2178.60, label="A'' (MHz)")
                    b0 = gr.Number(value=1251.20, label="B'' (MHz)")
                    c0 = gr.Number(value=794.50, label="C'' (MHz)")
                
                gr.Markdown("### 2. Estado Excitado ($S_1$) - MHz")
                with gr.Row():
                    a1 = gr.Number(value=2145.30, label="A' (MHz)")
                    b1 = gr.Number(value=1242.10, label="B' (MHz)")
                    c1 = gr.Number(value=788.90, label="C' (MHz)")
                
                gr.Markdown("### 3. Condiciones Experimentales del Jet")
                with gr.Row():
                    trot = gr.Slider(0.5, 10.0, value=3.5, step=0.1, label="Temp. Rotacional (K)")
                    fwhm = gr.Slider(5.0, 50.0, value=25.0, step=1.0, label="Ancho Láser FWHM (MHz)")
                    jmax = gr.Slider(4, 25, value=12, step=1, label="J Máximo")
                
                gr.Markdown("### 4. Componentes Dipolares de Transición (Híbrida)")
                with gr.Row():
                    mu_a = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="μ_a (tipo a)")
                    mu_b = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μ_b (tipo b)")
                    mu_c = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="μ_c (tipo c)")
                
                btn_sim = gr.Button("⚡ Simular Espectro Cuántico", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                plot_out = gr.Plot(label="Espectro Simulado en Tiempo Real")
                stats_out = gr.Textbox(label="Diagnóstico Cuántico & Métricas de Wang", lines=6)
        
        # Callbacks
        preset_dropdown.change(
            fn=load_preset_values,
            inputs=[preset_dropdown],
            outputs=[a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax]
        )
        
        btn_sim.click(
            fn=simulate_rovibronic_spectrum,
            inputs=[a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax],
            outputs=[plot_out, stats_out]
        )
        
        # Run initial simulation on load
        demo.load(
            fn=simulate_rovibronic_spectrum,
            inputs=[a0, b0, c0, a1, b1, c1, trot, fwhm, mu_a, mu_b, mu_c, jmax],
            outputs=[plot_out, stats_out]
        )
        
        gr.Markdown(
            """
            ---
            *SpecWatson es software libre de código abierto desarrollado para la Universidad de Guanajuato.*  
            Repositorio y Artículos: [github.com/mgueldiaz10-oss/specwatson](https://github.com/mgueldiaz10-oss/specwatson)
            """
        )
        
    return demo

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
