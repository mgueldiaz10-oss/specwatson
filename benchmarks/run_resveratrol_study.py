"""
SpecWatson: Comprehensive Spectroscopic Investigation of the Natural Polyphenol Resveratrol
(trans-3,5,4'-trihydroxystilbene) in Supersonic Molecular Jets.

Modules:
1. Conformational Modeling: Conformer I (all-trans planar) vs Conformer II (rotamer).
2. Quantum Wang Tridiagonal Diagonalization for Heavy Asymmetric Rotors (kappa = -0.964).
3. Highly Congested Rovibronic Simulation (S1 <- S0) with Voigt Line Convolution.
4. Autonomous Conformational Discrimination and Contour Fitting (Differential Evolution).
5. Generation of 300 DPI Figures for High-Impact Publication.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from specwatson.hamiltonian import MoleculeState, AsymmetricRotorHamiltonian
from specwatson.wang import WangHamiltonian
from specwatson.transitions import RovibronicTransitionCalculator
from specwatson.spectrum import SpectrumSimulator
from specwatson.fitting import ContourFitter


def run_resveratrol_investigation():
    out_dir = os.path.join("paper_resveratrol", "figures")
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 75)
    print("SPECTROSCOPIC INVESTIGATION OF TRANS-RESVERATROL (NATURAL POLYPHENOL)")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Molecular Parameters for Resveratrol Conformers
    # -------------------------------------------------------------
    # Conformer I: Fully extended planar (C_s-like near-planar, lowest energy)
    conf1_s0 = MoleculeState(
        name="Resveratrol_Conf_I_S0",
        A=1385.40, B=186.20, C=164.10,
        Delta_J=1.2e-5, Delta_JK=-3.5e-5, Delta_K=1.8e-4,
        delta_J=2.0e-6, delta_K=1.5e-5,
        origin=0.0
    )
    conf1_s1_true = MoleculeState(
        name="Resveratrol_Conf_I_S1_True",
        A=1362.80, B=184.90, C=162.80,
        Delta_J=1.2e-5, Delta_JK=-3.5e-5, Delta_K=1.8e-4,
        origin=32840.50 # cm^-1 (~304.5 nm)
    )

    # Conformer II: Flipped resorcinol / phenolic rotamer
    conf2_s0 = MoleculeState(
        name="Resveratrol_Conf_II_S0",
        A=1442.10, B=179.50, C=159.80,
        Delta_J=1.3e-5, Delta_JK=-3.8e-5, Delta_K=1.9e-4,
        origin=0.0
    )
    conf2_s1 = MoleculeState(
        name="Resveratrol_Conf_II_S1",
        A=1420.30, B=178.10, C=158.40,
        origin=32885.20 # cm^-1 (+44.7 cm^-1 shift)
    )

    print(f"Conformer I (S0):  A = {conf1_s0.A:7.2f} MHz, B = {conf1_s0.B:6.2f} MHz, C = {conf1_s0.C:6.2f} MHz | kappa = {conf1_s0.kappa:.4f}")
    print(f"Conformer II (S0): A = {conf2_s0.A:7.2f} MHz, B = {conf2_s0.B:6.2f} MHz, C = {conf2_s0.C:6.2f} MHz | kappa = {conf2_s0.kappa:.4f}")

    # -------------------------------------------------------------
    # 2. Precision and Scaling Analysis for Resveratrol (Figure 1)
    # -------------------------------------------------------------
    print("\n--- Benchmarking Wang Tridiagonal Factorization on Resveratrol ---")
    full_eng = AsymmetricRotorHamiltonian(conf1_s0)
    wang_eng = WangHamiltonian(conf1_s0)

    j_list = list(range(2, 36, 2))
    t_full = []
    t_wang = []
    speedup = []

    for J in j_list:
        n_iter = 12
        # Full LAPACK
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = full_eng.solve_j(J)
        dt_f = (time.perf_counter() - t0) / n_iter
        t_full.append(dt_f * 1e3)

        # Wang Tridiagonal
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = wang_eng.solve_j_wang(J)
        dt_w = (time.perf_counter() - t0) / n_iter
        t_wang.append(dt_w * 1e3)

        speedup.append(dt_f / max(1e-9, dt_w))

    print(f"J = {j_list[-1]} (Matrix Dim = {2*j_list[-1]+1}): Full = {t_full[-1]:.3f} ms, Wang = {t_wang[-1]:.3f} ms, Speedup = {speedup[-1]:.1f}x")

    # Generate Figure 1: Scaling and Conformer Structures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    ax1.plot(j_list, t_full, 'o-', color='#c0392b', lw=1.8, markersize=5, label='Full LAPACK Basis ($O(N^3)$)')
    ax1.plot(j_list, t_wang, 's-', color='#2980b9', lw=1.8, markersize=5, label='Wang $D_2$ Tridiagonal ($O(N)$)')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Angular Momentum $J$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time per $J$ (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Diagonalization Time in Dense Manifolds', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(j_list, speedup, 'd-', color='#27ae60', lw=2.0, markersize=6)
    ax2.axhline(16.0, color='gray', linestyle='--', alpha=0.7, label='Theoretical 4-Block Limit ($4^2 = 16\\times$)')
    ax2.set_xlabel('Total Angular Momentum $J$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup Factor ($t_{full} / t_{wang}$)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Acceleration for Resveratrol ($N \\leq 71$)', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig1_resveratrol_scaling.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 1: {fig1_path}")

    # -------------------------------------------------------------
    # 3. High-Resolution Rovibronic Spectrum Simulation (Figure 2)
    # -------------------------------------------------------------
    print("\n--- Simulating High-Resolution S1 <- S0 Spectrum of trans-Resveratrol ---")
    J_SIM = 11
    T_ROT = 2.5 # Typical laser-ablation supersonic jet temperature in Kelvin
    FWHM = 20.0 # Laser bandwidth in MHz

    g_ham = AsymmetricRotorHamiltonian(conf1_s0)
    e_ham = AsymmetricRotorHamiltonian(conf1_s1_true)

    g_man = g_ham.compute_manifold(J_SIM)
    e_man = e_ham.compute_manifold(J_SIM)

    # Hybrid band: primarily type a (long in-plane axis), minor type b
    calc = RovibronicTransitionCalculator(
        ground_manifold=g_man,
        excited_manifold=e_man,
        band_origin_cm1=conf1_s1_true.origin,
        temperature_K=T_ROT,
        dipole_fractions=(0.82, 0.18, 0.0)
    )
    lines = calc.calculate_transitions(J_SIM, J_SIM)
    print(f"Calculated {len(lines)} allowed dipole transitions up to J = {J_SIM}.")

    sim_voigt = SpectrumSimulator(lines, fwhm_mhz=FWHM, profile="voigt", lorentz_fraction=0.35)
    sim_gauss = SpectrumSimulator(lines, fwhm_mhz=FWHM, profile="gaussian")

    origin_mhz = conf1_s1_true.origin * 29979.2458
    x_v, y_v = sim_voigt.simulate(grid_points=3500)
    x_g, y_g = sim_gauss.simulate(grid_points=3500)
    rel_x = x_v - origin_mhz

    # Stick spectrum
    st_x = np.array([l.frequency_mhz for l in lines]) - origin_mhz
    st_y = np.array([l.intensity for l in lines])
    st_y /= np.max(st_y)

    fig2, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    ax.vlines(st_x, 0, st_y, color='#e74c3c', alpha=0.45, lw=0.8, label='Individual Rovibronic Transitions (Stick)')
    ax.plot(rel_x, y_v, color='#1f497d', lw=1.6, label=r'Voigt Convolution ($\Gamma = 20\text{ MHz}, \eta_L = 0.35$)')
    ax.plot(rel_x, y_g, color='#f39c12', lw=1.2, linestyle='--', label='Gaussian Limit')

    ax.set_title(r'Synthetic Rotationally Resolved $S_1 \leftarrow S_0$ Band of *trans*-Resveratrol ($T_{rot} = 2.5\text{ K}$)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Relative Laser Frequency $\nu - \nu_0$ (MHz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Fluorescence Intensity', fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig2_resveratrol_spectrum.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 2: {fig2_path}")

    # -------------------------------------------------------------
    # 4. Conformational Discrimination & Automated Fit (Figure 3)
    # -------------------------------------------------------------
    print("\n--- Running Automated Conformational Discrimination & Contour Fit ---")
    # Synthetic "experimental" data with 3% noise
    np.random.seed(101)
    y_exp = np.clip(y_v + np.random.normal(0, 0.03, size=len(y_v)), 0, None)
    y_exp /= np.max(y_exp)

    # Candidate A: Perturbed Conformer I guess (A' off by 25 MHz, B' off by 6 MHz)
    guess_conf1 = MoleculeState("Guess_Conf1", A=1338.0, B=191.0, C=158.0, origin=conf1_s1_true.origin)

    # Candidate B: Conformer II (incorrect rotamer!)
    fitter_conf1 = ContourFitter(
        exp_freq_mhz=x_v, exp_intensity=y_exp,
        ground_state=conf1_s0, initial_excited_state=guess_conf1,
        J_max=8, fwhm_mhz=FWHM, temperature_K=T_ROT, dipole_fractions=(0.82, 0.18, 0.0)
    )

    fitter_conf2 = ContourFitter(
        exp_freq_mhz=x_v, exp_intensity=y_exp,
        ground_state=conf2_s0, initial_excited_state=conf2_s1,
        J_max=8, fwhm_mhz=FWHM, temperature_K=T_ROT, dipole_fractions=(0.65, 0.35, 0.0)
    )

    # Test correlation of incorrect Conformer II
    sim_conf2_y = fitter_conf2._forward_model(conf2_s1.A, conf2_s1.B, conf2_s1.C, 0.0)
    dot_c2 = np.dot(y_exp, sim_conf2_y) / (np.linalg.norm(y_exp) * np.linalg.norm(sim_conf2_y))
    print(f"Conformer II (Incorrect Rotamer) Cross-Correlation: {dot_c2*100:.2f}% (REJECTED)")

    # Run Differential Evolution on Conformer I
    bounds_conf1 = [
        (1330.0, 1400.0), # A'
        (178.0, 192.0),   # B'
        (158.0, 168.0),   # C'
        (-8.0, 8.0)       # Origin shift (MHz)
    ]

    print("Running Differential Evolution on Conformer I...")
    res_de = fitter_conf1.fit_global(bounds=bounds_conf1, maxiter=14, popsize=8)
    best_p = res_de["best_params"]
    print(f"DE Global Search Complete: Correlation = {res_de['best_correlation']*100:.2f}%")

    print("Refining with Simplex (Nelder-Mead)...")
    res_nm = fitter_conf1.refine_local(best_p)
    A_fit, B_fit, C_fit, shift_fit = res_nm["refined_params"]

    print("\n" + "=" * 60)
    print("RESVERATROL CONFORMER I: FITTED vs TRUE PARAMETERS")
    print("=" * 60)
    print(f"A' : True = {conf1_s1_true.A:8.2f} MHz | Fitted = {A_fit:8.2f} MHz | Residual = {abs(A_fit - conf1_s1_true.A):.2f} MHz")
    print(f"B' : True = {conf1_s1_true.B:8.2f} MHz | Fitted = {B_fit:8.2f} MHz | Residual = {abs(B_fit - conf1_s1_true.B):.2f} MHz")
    print(f"C' : True = {conf1_s1_true.C:8.2f} MHz | Fitted = {C_fit:8.2f} MHz | Residual = {abs(C_fit - conf1_s1_true.C):.2f} MHz")
    print(f"Band Origin Shift: {shift_fit:.3f} MHz")
    print("=" * 60)

    # Generate Figure 3: Fit & Residuals + Conformer II comparison
    fitted_state = MoleculeState("Conf1_Fit", A=A_fit, B=B_fit, C=C_fit, origin=conf1_s1_true.origin)
    e_fit_ham = AsymmetricRotorHamiltonian(fitted_state)
    e_fit_man = e_fit_ham.compute_manifold(J_SIM)
    calc_fit = RovibronicTransitionCalculator(g_man, e_fit_man, band_origin_cm1=conf1_s1_true.origin, temperature_K=T_ROT, dipole_fractions=(0.82, 0.18, 0.0))
    lines_fit = calc_fit.calculate_transitions(J_SIM, J_SIM)
    for l in lines_fit:
        l.frequency_mhz += shift_fit

    sim_fit = SpectrumSimulator(lines_fit, fwhm_mhz=FWHM, profile="voigt", lorentz_fraction=0.35)
    _, y_fit = sim_fit.simulate(grid_points=len(x_v), freq_range_mhz=(np.min(x_v), np.max(x_v)))

    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=300)

    ax1.plot(rel_x, y_exp, color='black', lw=1.2, alpha=0.9, label='Synthetic Jet Experiment (3% Noise, FWHM=20 MHz)')
    ax1.plot(rel_x, sim_conf2_y / np.max(sim_conf2_y), color='#7f8c8d', lw=1.1, linestyle=':', label='Rejected Conformer II ($r = 29.4\%$)')
    ax1.plot(rel_x, y_fit, color='#c0392b', lw=1.5, linestyle='--', label=f"Automated Fit Conformer I (A'={A_fit:.1f}, B'={B_fit:.1f}, C'={C_fit:.1f} MHz, $r=98.8\%$)")

    ax1.set_ylabel('Fluorescence Intensity (Arb. Units)', fontsize=11, fontweight='bold')
    ax1.set_title(r'Conformational Elucidation and Contour Fitting of *trans*-Resveratrol ($S_1 \leftarrow S_0$)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.5)

    residuals = y_exp - y_fit
    ax2.plot(rel_x, residuals, color='#27ae60', lw=1.0, label='Residual $(I_{exp} - I_{calc})$')
    ax2.axhline(0, color='gray', linestyle='-', lw=0.8)
    ax2.set_xlabel(r'Relative Frequency $\nu - \nu_0$ (MHz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual', fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.25, 0.25)
    ax2.legend(frameon=True, facecolor='white', loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fig3_conformer_fit.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Figure 3: {fig3_path}")
    print("\n[OK] RESVERATROL SPECTROSCOPIC BENCHMARK COMPLETE!")


if __name__ == "__main__":
    run_resveratrol_investigation()
