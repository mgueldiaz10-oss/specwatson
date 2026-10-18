"""
Benchmark 3: Automated Contour Fitting on Noisy Carbazole S1 <- S0 Spectrum
Demonstrates parameter recovery without manual line assignments.
Exports Figure 3 for the publication manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator
from specwatson.spectrum import SpectrumSimulator
from specwatson.fitting import ContourFitter


def run_fitting_experiment():
    print("=" * 70)
    print("3. AUTOMATED CONTOUR FITTING BENCHMARK (GENETIC ALGORITHM)")
    print("=" * 70)

    # Ground State (Fixed from microwave or ab initio)
    ground_state = MoleculeState(
        name="Carbazole_S0",
        A=2178.60, B=1251.20, C=794.50,
        Delta_J=0.00025, Delta_JK=-0.00062, Delta_K=0.00120,
        origin=0.0
    )

    # True Excited State (S1)
    true_excited = MoleculeState(
        name="Carbazole_S1_True",
        A=2145.30, B=1242.10, C=788.90,
        Delta_J=0.00024, Delta_JK=-0.00060, Delta_K=0.00115,
        origin=30000.0 # cm^-1
    )

    J_FIT = 10
    T_ROT = 3.5 # Kelvin
    FWHM = 25.0 # MHz

    # Generate synthetic "experimental" spectrum
    g_ham = AsymmetricRotorHamiltonian(ground_state)
    e_ham = AsymmetricRotorHamiltonian(true_excited)

    g_man = g_ham.compute_manifold(J_FIT)
    e_man = e_ham.compute_manifold(J_FIT)

    calc = RovibronicTransitionCalculator(g_man, e_man, band_origin_cm1=30000.0, temperature_K=T_ROT, dipole_fractions=(1.0, 0.0, 0.0))
    lines = calc.calculate_transitions(J_FIT, J_FIT)

    sim = SpectrumSimulator(lines, fwhm_mhz=FWHM, profile="voigt", lorentz_fraction=0.3)
    x_freq, y_clean = sim.simulate(grid_points=2500)

    # Add 3% Gaussian experimental noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, size=len(y_clean))
    y_exp = np.clip(y_clean + noise, 0, None)
    y_exp /= np.max(y_exp)

    # Initial guess state with perturbed parameters (+/- 30-50 MHz offset)
    initial_guess = MoleculeState(
        name="Carbazole_S1_Guess",
        A=2110.0, B=1260.0, C=775.0, # significantly off!
        origin=30000.0
    )

    print(f"True S1 Parameters:    A' = {true_excited.A:.2f} MHz, B' = {true_excited.B:.2f} MHz, C' = {true_excited.C:.2f} MHz")
    print(f"Initial Guess (Off):   A' = {initial_guess.A:.2f} MHz, B' = {initial_guess.B:.2f} MHz, C' = {initial_guess.C:.2f} MHz")

    fitter = ContourFitter(
        exp_freq_mhz=x_freq,
        exp_intensity=y_exp,
        ground_state=ground_state,
        initial_excited_state=initial_guess,
        J_max=8, # Fast evaluation during global search
        fwhm_mhz=FWHM,
        temperature_K=T_ROT,
        dipole_fractions=(1.0, 0.0, 0.0)
    )

    bounds = [
        (2100.0, 2200.0), # A'
        (1200.0, 1280.0), # B'
        (760.0, 810.0),   # C'
        (-10.0, 10.0)     # Origin shift (MHz)
    ]

    print("\nRunning Differential Evolution optimizer...")
    res_global = fitter.fit_global(bounds=bounds, maxiter=15, popsize=8)
    best_p = res_global["best_params"]
    print(f"Global Optimization Complete: Best Correlation = {res_global['best_correlation']*100:.2f}%")

    print("\nRunning Local Simplex (Nelder-Mead) Refinement...")
    res_local = fitter.refine_local(best_p)
    refined_p = res_local["refined_params"]

    A_fit, B_fit, C_fit, shift_fit = refined_p
    print("\n" + "=" * 55)
    print("FINAL FITTED PARAMETERS vs EXPERIMENTAL GROUND TRUTH")
    print("=" * 55)
    print(f"A' : True = {true_excited.A:8.2f} MHz | Fitted = {A_fit:8.2f} MHz | Residual = {abs(A_fit - true_excited.A):.2f} MHz")
    print(f"B' : True = {true_excited.B:8.2f} MHz | Fitted = {B_fit:8.2f} MHz | Residual = {abs(B_fit - true_excited.B):.2f} MHz")
    print(f"C' : True = {true_excited.C:8.2f} MHz | Fitted = {C_fit:8.2f} MHz | Residual = {abs(C_fit - true_excited.C):.2f} MHz")
    print(f"Band Origin Shift: {shift_fit:.3f} MHz")
    print("=" * 55)

    # Generate Figure 3
    fitted_state = MoleculeState("S1_Fit", A=A_fit, B=B_fit, C=C_fit, origin=30000.0)
    e_fit_ham = AsymmetricRotorHamiltonian(fitted_state)
    e_fit_man = e_fit_ham.compute_manifold(J_FIT)

    calc_fit = RovibronicTransitionCalculator(g_man, e_fit_man, band_origin_cm1=30000.0, temperature_K=T_ROT, dipole_fractions=(1.0, 0.0, 0.0))
    lines_fit = calc_fit.calculate_transitions(J_FIT, J_FIT)
    for l in lines_fit:
        l.frequency_mhz += shift_fit

    sim_fit = SpectrumSimulator(lines_fit, fwhm_mhz=FWHM, profile="voigt", lorentz_fraction=0.3)
    _, y_fit = sim_fit.simulate(grid_points=len(x_freq), freq_range_mhz=(np.min(x_freq), np.max(x_freq)))

    # Initial guess spectrum for comparison
    e_guess_ham = AsymmetricRotorHamiltonian(initial_guess)
    e_guess_man = e_guess_ham.compute_manifold(J_FIT)
    calc_guess = RovibronicTransitionCalculator(g_man, e_guess_man, band_origin_cm1=30000.0, temperature_K=T_ROT, dipole_fractions=(1.0, 0.0, 0.0))
    lines_guess = calc_guess.calculate_transitions(J_FIT, J_FIT)
    sim_guess = SpectrumSimulator(lines_guess, fwhm_mhz=FWHM, profile="voigt", lorentz_fraction=0.3)
    _, y_guess = sim_guess.simulate(grid_points=len(x_freq), freq_range_mhz=(np.min(x_freq), np.max(x_freq)))

    rel_freq = x_freq - (30000.0 * 29979.2458)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=300)

    # Upper panel: Spectra
    ax1.plot(rel_freq, y_exp, color='black', lw=1.2, alpha=0.9, label='Synthetic Experimental (3% Noise, Voigt FWHM=25 MHz)')
    ax1.plot(rel_freq, y_guess, color='#e67e22', lw=1.2, linestyle=':', label=f"Initial Guess (A'={initial_guess.A:.0f}, B'={initial_guess.B:.0f}, C'={initial_guess.C:.0f} MHz)")
    ax1.plot(rel_freq, y_fit, color='#c0392b', lw=1.5, linestyle='--', label=f"Automated Fit (A'={A_fit:.1f}, B'={B_fit:.1f}, C'={C_fit:.1f} MHz)")

    ax1.set_ylabel('Fluorescence Intensity (Arb. Units)', fontsize=11, fontweight='bold')
    ax1.set_title(r'Automated Contour Fitting of the $S_1 \leftarrow S_0$ Band Origin of Carbazole', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Lower panel: Residuals
    residuals = y_exp - y_fit
    ax2.plot(rel_freq, residuals, color='#27ae60', lw=1.0, label='Residual $(I_{exp} - I_{calc})$')
    ax2.axhline(0, color='gray', linestyle='-', lw=0.8)
    ax2.set_xlabel(r'Relative Frequency $\nu - \nu_0$ (MHz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual', fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.25, 0.25)
    ax2.legend(frameon=True, facecolor='white', loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join("figures", "fig3_contour_fit.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"\n[OK] Saved Figure 3: {fig3_path}")


if __name__ == "__main__":
    run_fitting_experiment()
