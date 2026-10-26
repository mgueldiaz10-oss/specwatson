"""
SpecWatson: Hamiltonian Engine for Rigid Rotor and Watson Effective Hamiltonians.
Representation: I^r (z = a, x = b, y = c), ideal for prolate asymmetric tops.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class MoleculeState:
    """
    Spectroscopic parameters for an electronic state (ground or excited).
    Units: MHz (or cm^-1 if specified, standard conversion: 1 cm^-1 = 29979.2458 MHz)
    """
    name: str = "State"
    # Rotational constants (MHz)
    A: float = 3000.0
    B: float = 1500.0
    C: float = 1000.0
    
    # Watson's A-reduction quartic centrifugal distortion constants (MHz)
    Delta_J: float = 0.0
    Delta_JK: float = 0.0
    Delta_K: float = 0.0
    delta_J: float = 0.0
    delta_K: float = 0.0
    
    # Electronic band origin (cm^-1) - typically 0 for ground state
    origin: float = 0.0

    @property
    def kappa(self) -> float:
        """Ray's asymmetry parameter: kappa = (2B - A - C) / (A - C).
        kappa = -1 for prolate symmetric top, +1 for oblate symmetric top.
        """
        if abs(self.A - self.C) < 1e-9:
            return 0.0
        return (2.0 * self.B - self.A - self.C) / (self.A - self.C)

    def ray_asymmetry(self) -> float:
        """Method alias for Ray's asymmetry parameter kappa."""
        return self.kappa

    @property
    def is_rigid_rotor(self) -> bool:
        """Returns True if all Watson centrifugal distortion constants are zero (Pure Rigid Rotor)."""
        return (
            abs(self.Delta_J)
            + abs(self.Delta_JK)
            + abs(self.Delta_K)
            + abs(self.delta_J)
            + abs(self.delta_K)
        ) < 1e-12

    @property
    def model_description(self) -> str:
        """Returns human-readable description of the Hamiltonian model in use."""
        if self.is_rigid_rotor:
            return "Rotor Rígido Puro (3 constantes: A, B, C | Sin distorsión centrífuga)"
        return "Hamiltoniano Efectivo de Watson Reducción A (con distorsión centrífuga cuártica)"

    @property
    def kappa_classification(self) -> str:
        """Returns physical interpretation of Ray's asymmetry parameter kappa."""
        k = self.kappa
        if k <= -0.98:
            return "Trompo Cuasi-Simétrico Prolato Extremo (forma de cigarro/aguja, A ≫ B ≈ C)"
        elif k <= -0.60:
            return "Rotor Asimétrico Prolato (alargado en el eje a, A > B ≈ C)"
        elif k < 0.60:
            return "Rotor de Alta Asimetría Intermedia (A > B > C)"
        elif k < 0.98:
            return "Rotor Asimétrico Oblato (aplanado tipo disco, A ≈ B > C)"
        else:
            return "Trompo Cuasi-Simétrico Oblato Extremo (tipo disco/moneda, A ≈ B ≫ C)"


class AsymmetricRotorHamiltonian:
    """
    Constructs and diagonalizes the asymmetric rotor Hamiltonian in the 
    symmetric top basis |J, K, M> using the I^r representation.
    Enforces exact Wang parity (D2 four-group symmetry) on degenerate K-doublets
    so that Type a, Type b, and Type c selection rules never mix.
    """

    def __init__(self, state: MoleculeState):
        self.state = state

    @staticmethod
    def _ladder_factor(J: int, K: int) -> float:
        """
        Calculates sqrt([J(J+1) - K(K+1)] * [J(J+1) - (K+1)(K+2)])
        for K -> K+2 step operator.
        """
        term1 = J * (J + 1) - K * (K + 1)
        term2 = J * (J + 1) - (K + 1) * (K + 2)
        if term1 <= 0 or term2 <= 0:
            return 0.0
        return np.sqrt(term1 * term2)

    def build_matrix(self, J: int) -> np.ndarray:
        """
        Builds the (2J+1) x (2J+1) Hamiltonian matrix for a given total angular momentum J.
        Basis ordering: K = -J, -J+1, ..., 0, ..., J-1, J.
        """
        dim = 2 * J + 1
        H = np.zeros((dim, dim), dtype=np.float64)
        K_vals = np.arange(-J, J + 1)
        
        A = self.state.A
        B = self.state.B
        C = self.state.C
        
        dJ = self.state.Delta_J
        dJK = self.state.Delta_JK
        dK = self.state.Delta_K
        del_j = self.state.delta_J
        del_k = self.state.delta_K
        
        JJ1 = J * (J + 1)
        
        # Diagonal elements: <J, K | H | J, K>
        for i, K in enumerate(K_vals):
            # Rigid Rotor contribution (I^r: z=a, x=b, y=c)
            # H_0 = 0.5*(B+C)*J(J+1) + [A - 0.5*(B+C)]*K^2
            diag_rigid = 0.5 * (B + C) * JJ1 + (A - 0.5 * (B + C)) * (K ** 2)
            
            # Watson A-reduction quartic centrifugal distortion
            diag_watson = - dJ * (JJ1 ** 2) - dJK * JJ1 * (K ** 2) - dK * (K ** 4)
            
            H[i, i] = diag_rigid + diag_watson

        # Off-diagonal elements: <J, K+2 | H | J, K> and Hermitian conjugate
        for i, K in enumerate(K_vals):
            if K + 2 <= J:
                j_idx = i + 2
                f_fac = self._ladder_factor(J, K)
                
                # Rigid rotor off-diagonal: 0.25 * (B - C) * f(J, K)
                off_rigid = 0.25 * (B - C) * f_fac
                
                # Watson A-reduction off-diagonal:
                # - [2 * delta_J * J(J+1) + delta_K * ((K+2)^2 + K^2)] * 0.5 * f(J, K)
                off_watson = - (2.0 * del_j * JJ1 + del_k * ((K + 2)**2 + K**2)) * 0.5 * f_fac
                
                elem = off_rigid + off_watson
                H[i, j_idx] = elem
                H[j_idx, i] = elem

        return H

    def solve_j(self, J: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Diagonalizes the Hamiltonian for angular momentum J and enforces exact Wang
        symmetrization on degenerate asymmetry doublets.
        Returns:
            energies: Array of 2J+1 eigenvalues sorted in ascending order (MHz).
            eigenvectors: (2J+1, 2J+1) matrix where column i is the eigenvector for energy i.
            labels: List of (Ka, Kc) quantum number assignments for each level.
        """
        H = self.build_matrix(J)
        energies, vectors = np.linalg.eigh(H)
        labels = self._assign_quantum_labels(J, energies)

        # Symmetrize nearly degenerate K-doublets into exact Wang states (+/- parity under K -> -K)
        dim = 2 * J + 1
        i = 0
        while i < dim:
            Ka, Kc = labels[i]
            if (
                i + 1 < dim
                and labels[i + 1][0] == Ka
                and Ka > 0
                and abs(energies[i + 1] - energies[i]) < 1e-3
            ):
                u1 = vectors[:, i].copy()
                u2 = vectors[:, i + 1].copy()

                # Symmetric combination (+): v(-K) = +v(K)
                v_plus = u1 + u1[::-1]
                n_plus = np.linalg.norm(v_plus)
                if n_plus < 1e-6:
                    v_plus = u2 + u2[::-1]
                    n_plus = np.linalg.norm(v_plus)
                if n_plus > 0:
                    v_plus /= n_plus

                # Antisymmetric combination (-): v(-K) = -v(K)
                v_minus = u1 - u1[::-1]
                n_minus = np.linalg.norm(v_minus)
                if n_minus < 1e-6:
                    v_minus = u2 - u2[::-1]
                    n_minus = np.linalg.norm(v_minus)
                if n_minus > 0:
                    v_minus /= n_minus

                # In I^r representation: gamma = (J + Ka + Kc) % 2
                gamma_i = (J + Ka + Kc) % 2
                vectors[:, i] = v_minus if gamma_i == 1 else v_plus
                vectors[:, i + 1] = v_plus if gamma_i == 1 else v_minus
                i += 2
            else:
                i += 1

        return energies, vectors, labels

    def _assign_quantum_labels(self, J: int, sorted_energies: np.ndarray) -> List[Tuple[int, int]]:
        """
        Assigns (Ka, Kc) labels to the 2J+1 sorted energy levels.
        Level index i (from 0 to 2J) corresponds to tau = -J + i = Ka - Kc.
        Ka + Kc is either J or J + 1.
        """
        labels = []
        for i in range(2 * J + 1):
            tau = -J + i
            if (J - tau) % 2 == 0:
                Ka_plus_Kc = J
            else:
                Ka_plus_Kc = J + 1
            
            Ka = (Ka_plus_Kc + tau) // 2
            Kc = (Ka_plus_Kc - tau) // 2
            labels.append((int(Ka), int(Kc)))
            
        return labels

    def compute_manifold(self, J_max: int) -> Dict[int, Dict]:
        """
        Computes the complete rotational manifold up to J_max.
        Returns a dict keyed by J containing energies, vectors, and (Ka, Kc) labels.
        """
        manifold = {}
        for J in range(J_max + 1):
            energies, vectors, labels = self.solve_j(J)
            manifold[J] = {
                "energies": energies,
                "vectors": vectors,
                "labels": labels,
                "K_vals": np.arange(-J, J + 1)
            }
        return manifold
