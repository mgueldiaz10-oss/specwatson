"""
SpecWatson: Wang Symmetrized Basis Factorization Engine.
Implements the Wang transformation for asymmetric rotor effective Hamiltonians
under the four-group D2 (Vierergruppe).
Factors the (2J+1) x (2J+1) Hamiltonian into 4 independent tridiagonal sub-matrices:
E+, E-, O+, O-.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import eigh_tridiagonal, eigh
from .hamiltonian import MoleculeState


class WangHamiltonian:
    """
    Constructs and diagonalizes the asymmetric rotor Hamiltonian in the 
    Wang symmetrized basis |J, K, gamma> (representation I^r: z=a, x=b, y=c).
    
    Sub-blocks:
        - E+ : Even K (0, 2, 4, ...), gamma = 0 (+). Dimension: int(J/2) + 1
        - E- : Even K (2, 4, 6, ...), gamma = 1 (-). Dimension: int(J/2)
        - O+ : Odd  K (1, 3, 5, ...), gamma = 0 (+). Dimension: int((J+1)/2)
        - O- : Odd  K (1, 3, 5, ...), gamma = 1 (-). Dimension: int((J+1)/2)
    """

    def __init__(self, state: MoleculeState):
        self.state = state

    @staticmethod
    def _ladder_factor(J: int, K: int) -> float:
        """Computes sqrt([J(J+1) - K(K+1)] * [J(J+1) - (K+1)(K+2)])"""
        term1 = J * (J + 1) - K * (K + 1)
        term2 = J * (J + 1) - (K + 1) * (K + 2)
        if term1 <= 0 or term2 <= 0:
            return 0.0
        return np.sqrt(term1 * term2)

    def _diag_element(self, J: int, K: int) -> float:
        """Computes standard <J, K | H | J, K> diagonal matrix element."""
        JJ1 = J * (J + 1)
        A, B, C = self.state.A, self.state.B, self.state.C
        dJ, dJK, dK = self.state.Delta_J, self.state.Delta_JK, self.state.Delta_K
        
        # Rigid rotor
        diag_rigid = 0.5 * (B + C) * JJ1 + (A - 0.5 * (B + C)) * (K ** 2)
        # Watson A-reduction quartic distortion
        diag_watson = - dJ * (JJ1 ** 2) - dJK * JJ1 * (K ** 2) - dK * (K ** 4)
        return float(diag_rigid + diag_watson)

    def _offdiag_element(self, J: int, K: int) -> float:
        """Computes <J, K+2 | H | J, K> off-diagonal element."""
        JJ1 = J * (J + 1)
        B, C = self.state.B, self.state.C
        del_j, del_k = self.state.delta_J, self.state.delta_K
        f_fac = self._ladder_factor(J, K)
        
        off_rigid = 0.25 * (B - C) * f_fac
        off_watson = - (2.0 * del_j * JJ1 + del_k * ((K + 2)**2 + K**2)) * 0.5 * f_fac
        return float(off_rigid + off_watson)

    def _h1_minus1_element(self, J: int) -> float:
        """
        Computes <J, 1 | H | J, -1> coupling across K=0 for odd K levels.
        f(J, -1) = 0.5 * J * (J + 1).
        """
        JJ1 = J * (J + 1)
        B, C = self.state.B, self.state.C
        del_j, del_k = self.state.delta_J, self.state.delta_K
        
        # Rigid: 0.25 * (B - C) * (0.5 * JJ1) * 2 = 0.25 * (B - C) * JJ1
        off_rigid = 0.25 * (B - C) * JJ1
        # Watson: - (2*del_j*JJ1 + 2*del_k)*0.5 * (0.5 * JJ1) * 2 = - (del_j*JJ1 + del_k)*JJ1
        off_watson = - (del_j * JJ1 + del_k) * JJ1
        return float(off_rigid + off_watson)

    def solve_block_E_plus(self, J: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Solves E+ block: Even K = 0, 2, 4, ..."""
        K_vals = list(range(0, J + 1, 2))
        dim = len(K_vals)
        if dim == 0:
            return np.array([]), np.empty((0, 0)), []
        if dim == 1:
            val = self._diag_element(J, 0)
            return np.array([val]), np.ones((1, 1)), K_vals

        d = np.zeros(dim)
        e = np.zeros(dim - 1)

        # K = 0
        d[0] = self._diag_element(J, 0)
        # Coupling <J, 2, 0 | H | J, 0, 0> = sqrt(2) * H_{2, 0}
        e[0] = np.sqrt(2.0) * self._offdiag_element(J, 0)

        for i in range(1, dim):
            K = K_vals[i]
            d[i] = self._diag_element(J, K)
            if i < dim - 1:
                e[i] = self._offdiag_element(J, K)

        vals, vecs = eigh_tridiagonal(d, e)
        return vals, vecs, K_vals

    def solve_block_E_minus(self, J: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Solves E- block: Even K = 2, 4, 6, ..."""
        K_vals = list(range(2, J + 1, 2))
        dim = len(K_vals)
        if dim == 0:
            return np.array([]), np.empty((0, 0)), []
        if dim == 1:
            val = self._diag_element(J, 2)
            return np.array([val]), np.ones((1, 1)), K_vals

        d = np.zeros(dim)
        e = np.zeros(dim - 1)

        for i in range(dim):
            K = K_vals[i]
            d[i] = self._diag_element(J, K)
            if i < dim - 1:
                e[i] = self._offdiag_element(J, K)

        vals, vecs = eigh_tridiagonal(d, e)
        return vals, vecs, K_vals

    def solve_block_O_plus(self, J: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Solves O+ block: Odd K = 1, 3, 5, ..., gamma = 0 (+)"""
        K_vals = list(range(1, J + 1, 2))
        dim = len(K_vals)
        if dim == 0:
            return np.array([]), np.empty((0, 0)), []
        if dim == 1:
            val = self._diag_element(J, 1) + self._h1_minus1_element(J)
            return np.array([val]), np.ones((1, 1)), K_vals

        d = np.zeros(dim)
        e = np.zeros(dim - 1)

        # K = 1 has diagonal shift + H_{1, -1}
        d[0] = self._diag_element(J, 1) + self._h1_minus1_element(J)
        e[0] = self._offdiag_element(J, 1)

        for i in range(1, dim):
            K = K_vals[i]
            d[i] = self._diag_element(J, K)
            if i < dim - 1:
                e[i] = self._offdiag_element(J, K)

        vals, vecs = eigh_tridiagonal(d, e)
        return vals, vecs, K_vals

    def solve_block_O_minus(self, J: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Solves O- block: Odd K = 1, 3, 5, ..., gamma = 1 (-)"""
        K_vals = list(range(1, J + 1, 2))
        dim = len(K_vals)
        if dim == 0:
            return np.array([]), np.empty((0, 0)), []
        if dim == 1:
            val = self._diag_element(J, 1) - self._h1_minus1_element(J)
            return np.array([val]), np.ones((1, 1)), K_vals

        d = np.zeros(dim)
        e = np.zeros(dim - 1)

        # K = 1 has diagonal shift - H_{1, -1}
        d[0] = self._diag_element(J, 1) - self._h1_minus1_element(J)
        e[0] = self._offdiag_element(J, 1)

        for i in range(1, dim):
            K = K_vals[i]
            d[i] = self._diag_element(J, K)
            if i < dim - 1:
                e[i] = self._offdiag_element(J, K)

        vals, vecs = eigh_tridiagonal(d, e)
        return vals, vecs, K_vals

    def solve_j_wang(self, J: int) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        Diagonalizes all 4 Wang sub-blocks for a given J.
        Returns:
            all_energies: Sorted 1D array of (2J+1) eigenvalues (MHz).
            blocks_data: Dictionary with eigenvalues, eigenvectors, and K_vals for each block.
        """
        e_p_vals, e_p_vecs, e_p_k = self.solve_block_E_plus(J)
        e_m_vals, e_m_vecs, e_m_k = self.solve_block_E_minus(J)
        o_p_vals, o_p_vecs, o_p_k = self.solve_block_O_plus(J)
        o_m_vals, o_m_vecs, o_m_k = self.solve_block_O_minus(J)

        blocks_data = {
            "E+": {"vals": e_p_vals, "vecs": e_p_vecs, "K": e_p_k},
            "E-": {"vals": e_m_vals, "vecs": e_m_vecs, "K": e_m_k},
            "O+": {"vals": o_p_vals, "vecs": o_p_vecs, "K": o_p_k},
            "O-": {"vals": o_m_vals, "vecs": o_m_vecs, "K": o_m_k},
        }

        # Combine and sort all eigenvalues
        combined = []
        for b_name, b_info in blocks_data.items():
            for v in b_info["vals"]:
                combined.append((float(v), b_name))
        
        combined.sort(key=lambda x: x[0])
        sorted_energies = np.array([x[0] for x in combined])

        return sorted_energies, blocks_data

    def compute_manifold_wang(self, J_max: int) -> Dict[int, Dict]:
        """Computes the complete rotational manifold up to J_max using Wang factorization."""
        manifold = {}
        for J in range(J_max + 1):
            energies, blocks = self.solve_j_wang(J)
            manifold[J] = {
                "energies": energies,
                "blocks": blocks,
            }
        return manifold
