"""
AsymRotor — Rotationally resolved band of an asymmetric top (rigid rotor).
==========================================================================
Direct 1:1 Python implementation of Dr. Mario Ulises Delgado-Jaime's
`AsymRotor.m` / `asymrotor.py` engine for high-resolution electronic and
rovibronic spectroscopy.

Usage (compatible with Dr. Mario's call signature):
---------------------------------------------------
    from asymrotor import asym_rotor

    y = asym_rotor(x, A, B, C, dA, dB, dC, Nu0, Trot, Theta, Phi, GW, LW,
                   JMax=0, Unit=1, SpinAxis=0, SpinOdd=1.0, T2=0.0, W2=0.0)

    y, lines = asym_rotor(x, A, B, C, dA, dB, dC, Nu0, Trot, Theta, Phi, GW, LW,
                          JMax=0, Unit=1, SpinAxis=0, SpinOdd=1.0, T2=0.0, W2=0.0,
                          lines=True)
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline

# LRU Cache for AsymRotorLines (stores up to 8 parameter sets)
_ASYM_ROTOR_CACHE: "OrderedDict[Tuple[float, ...], Dict[str, np.ndarray]]" = OrderedDict()
_MAX_CACHE_SIZE = 8


def planar_top_c(A: float, B: float) -> float:
    """
    Planar molecule approximation proposed by Dr. Mario Delgado-Jaime:
        I_c = I_a + I_b  ==>  C ≈ (A * B) / (A + B)
    Particularly useful for planar aromatic tops such as Triphenylene (C18H12).
    """
    A, B = float(A), float(B)
    if A + B <= 0:
        return 0.0
    return (A * B) / (A + B)


def ray_kappa(A: float, B: float, C: float) -> float:
    """
    Ray's asymmetry parameter:
        κ = (2B - A - C) / (A - C)
    κ = -1 for a prolate symmetric top (B = C)
    κ = +1 for an oblate symmetric top (A = B)
    """
    A, B, C = float(A), float(B), float(C)
    if abs(A - C) < 1e-12:
        return 0.0
    return (2.0 * B - A - C) / (A - C)


def _block_parity(blk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (kp, P) for Wang block indices 1..4."""
    blk_arr = np.asarray(blk, dtype=int)
    kp = (blk_arr >= 3).astype(int)
    P = 1 - 2 * (((blk_arr - 1) % 2) == 1).astype(int)
    return kp, P


def _cg_matrix(Jl: int, Ju: int, q: int) -> np.ndarray:
    """
    Clebsch-Gordan matrix M(K'+Ju, K+Jl) = <Jl K; 1 q | Ju K+q>, K' = K + q.
    Exact analytical implementation matching CGMatrix in AsymRotor.m.
    """
    K = np.arange(-Jl, Jl + 1, dtype=int)
    Mz = K + q
    ok = np.abs(Mz) <= Ju
    K = K[ok]
    Mz_f = Mz[ok].astype(float)
    j = float(Jl)
    dJ = Ju - Jl

    if dJ == 1:
        if q == 1:
            v = np.sqrt((j + Mz_f) * (j + Mz_f + 1.0) / ((2.0 * j + 1.0) * (2.0 * j + 2.0)))
        elif q == 0:
            v = np.sqrt((j - Mz_f + 1.0) * (j + Mz_f + 1.0) / ((2.0 * j + 1.0) * (j + 1.0)))
        else:  # q == -1
            v = np.sqrt((j - Mz_f) * (j - Mz_f + 1.0) / ((2.0 * j + 1.0) * (2.0 * j + 2.0)))
    elif dJ == 0:
        if j == 0:
            v = np.zeros_like(Mz_f)
        elif q == 1:
            v = -np.sqrt((j + Mz_f) * (j - Mz_f + 1.0) / (2.0 * j * (j + 1.0)))
        elif q == 0:
            v = Mz_f / np.sqrt(j * (j + 1.0))
        else:  # q == -1
            v = np.sqrt((j - Mz_f) * (j + Mz_f + 1.0) / (2.0 * j * (j + 1.0)))
    elif dJ == -1:
        if q == 1:
            v = np.sqrt((j - Mz_f) * (j - Mz_f + 1.0) / (2.0 * j * (2.0 * j + 1.0)))
        elif q == 0:
            v = -np.sqrt((j - Mz_f) * (j + Mz_f) / (j * (2.0 * j + 1.0)))
        else:  # q == -1
            v = np.sqrt((j + Mz_f + 1.0) * (j + Mz_f) / (2.0 * j * (2.0 * j + 1.0)))
    else:
        v = np.zeros_like(Mz_f)

    M = np.zeros((2 * Ju + 1, 2 * Jl + 1), dtype=float)
    M[Mz[ok] + Ju, K + Jl] = v
    return M


def rotor_levels(J: int, A: float, B: float, C: float) -> Dict[str, np.ndarray]:
    """
    Levels of one J block of H = A Ja^2 + B Jb^2 + C Jc^2 in representation I^r,
    split into the four Wang symmetry blocks (K even/odd, sym/anti) so that every
    eigenvector has exact symmetry even when K doublets are degenerate to machine precision.
    """
    K = np.arange(-J, J + 1, dtype=int)
    Kf = K.astype(float)
    n = 2 * J + 1
    jj1 = float(J * (J + 1))

    H = np.diag(A * (Kf ** 2) + 0.5 * (B + C) * (jj1 - Kf ** 2))
    if J >= 1:
        k = Kf[:-2]
        v = 0.25 * (B - C) * np.sqrt((jj1 - k * (k + 1.0)) * (jj1 - (k + 1.0) * (k + 2.0)))
        H += np.diag(v, 2) + np.diag(v, -2)

    E = np.zeros(n, dtype=float)
    V = np.zeros((n, n), dtype=float)
    Blk = np.zeros(n, dtype=int)
    c = 0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for kp in (0, 1):
        Kpos = K[(K >= 0) & (np.mod(K, 2) == kp)]
        for s in (1, -1):
            kk = Kpos[(Kpos > 0) | (s > 0)]  # K = 0 is symmetric only
            m = len(kk)
            if m == 0:
                continue
            Ub = np.zeros((n, m), dtype=float)
            cols = np.arange(m)
            Ub[J + kk, cols] = inv_sqrt2
            Ub[J - kk, cols] = s * inv_sqrt2
            zero_mask = (kk == 0)
            if np.any(zero_mask):
                Ub[J, zero_mask] = 1.0

            Hb = Ub.T @ H @ Ub
            Db, Vb = np.linalg.eigh(0.5 * (Hb + Hb.T))
            E[c:c + m] = Db
            V[:, c:c + m] = Ub @ Vb
            Blk[c:c + m] = 2 * kp + (1 if s < 0 else 0) + 1
            c += m

    o = np.argsort(E, kind="mergesort")
    E = E[o]
    V = V[:, o]
    Blk = Blk[o]

    kp_arr, P_arr = _block_parity(Blk)
    # Symmetry under C2 rotations about a, b, c
    Oa = (kp_arr == 1)
    Ob = ((J % 2) != (P_arr < 0))
    Oc = np.logical_xor(Oa, Ob)
    Odd = np.column_stack((Oa, Ob, Oc))

    # Ka, Kc labels from energy order, corrected with exact Wang parities for degenerate doublets
    nn = np.arange(n, dtype=int)
    Ka = (nn + 1) // 2
    Kc = J - (nn // 2)
    Bad = ((Ka % 2) != Oa) | ((Kc % 2) != Oc)
    for i in range(n - 1):
        if (
            Bad[i] and Bad[i + 1]
            and (Ka[i + 1] % 2 == Oa[i]) and (Kc[i + 1] % 2 == Oc[i])
            and (Ka[i] % 2 == Oa[i + 1]) and (Kc[i] % 2 == Oc[i + 1])
        ):
            Ka[i], Ka[i + 1] = Ka[i + 1], Ka[i]
            Kc[i], Kc[i + 1] = Kc[i + 1], Kc[i]
            Bad[i] = False
            Bad[i + 1] = False

    Lab = np.column_stack((Ka, Kc))
    return {"E": E, "V": V, "Blk": Blk, "Odd": Odd, "Lab": Lab}


def asym_rotor_lines(
    A: float, B: float, C: float,
    Au: float, Bu: float, Cu: float,
    JMax: int
) -> Dict[str, np.ndarray]:
    """
    Computes every transition J'' <= JMax with its line strength S,
    cached by rotational constants (A, B, C, Au, Bu, Cu, JMax).
    """
    key = (
        round(float(A), 9), round(float(B), 9), round(float(C), 9),
        round(float(Au), 9), round(float(Bu), 9), round(float(Cu), 9),
        int(JMax)
    )
    if key in _ASYM_ROTOR_CACHE:
        _ASYM_ROTOR_CACHE.move_to_end(key, last=False)
        return _ASYM_ROTOR_CACHE[key]

    Lo = [rotor_levels(J, A, B, C) for J in range(JMax + 1)]
    Hi = [rotor_levels(J, Au, Bu, Cu) for J in range(JMax + 2)]

    Scut = 1e-12
    NuC, SC, TC, ElC, OddC, LabC = [], [], [], [], [], []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for Jl in range(JMax + 1):
        L = Lo[Jl]
        K_l = np.arange(-Jl, Jl + 1, dtype=int)
        for Ju in range(max(Jl - 1, 0), Jl + 2):
            if Jl == 0 and Ju == 0:
                continue
            U = Hi[Ju]
            K_u = np.arange(-Ju, Ju + 1, dtype=int)
            dJ = Ju - Jl

            M0 = _cg_matrix(Jl, Ju, 0)
            Mp = _cg_matrix(Jl, Ju, 1)
            Mm = _cg_matrix(Jl, Ju, -1)
            Mt = (M0, (Mm - Mp) * inv_sqrt2, (Mm + Mp) * inv_sqrt2)

            for bl in range(1, 5):
                il = np.flatnonzero(L["Blk"] == bl)
                if len(il) == 0:
                    continue
                kp_arr, P_arr = _block_parity(np.array([bl]))
                kp, P = int(kp_arr[0]), int(P_arr[0])
                parity_factor = -1 if (abs(dJ) % 2 == 1) else 1

                for Ty in (1, 2, 3):
                    if Ty == 1:
                        kpu = kp
                        Pu = -P * parity_factor
                    elif Ty == 2:
                        kpu = 1 - kp
                        Pu = P * parity_factor
                    else:
                        kpu = 1 - kp
                        Pu = -P * parity_factor

                    bu = 2 * kpu + (1 if Pu < 0 else 0) + 1
                    iu = np.flatnonzero(U["Blk"] == bu)
                    if len(iu) == 0:
                        continue

                    rl = np.flatnonzero(np.mod(K_l, 2) == kp)
                    ru = np.flatnonzero(np.mod(K_u, 2) == kpu)

                    sub_Mt = Mt[Ty - 1][np.ix_(ru, rl)]
                    R = U["V"][np.ix_(ru, iu)].T @ (sub_Mt @ L["V"][np.ix_(rl, il)])
                    S = (2 * Jl + 1) * (R ** 2)

                    f_idx, i_idx = np.nonzero(S > Scut)
                    if len(f_idx) == 0:
                        continue

                    fi = iu[f_idx]
                    ii = il[i_idx]
                    n_tr = len(f_idx)

                    NuC.append(U["E"][fi] - L["E"][ii])
                    SC.append(S[f_idx, i_idx])
                    TC.append(np.full(n_tr, Ty, dtype=int))
                    ElC.append(L["E"][ii])
                    OddC.append(L["Odd"][ii, :])
                    LabC.append(np.column_stack((
                        np.full(n_tr, Jl, dtype=int),
                        L["Lab"][ii, :],
                        np.full(n_tr, Ju, dtype=int),
                        U["Lab"][fi, :]
                    )))

    if NuC:
        lst = {
            "Nu": np.concatenate(NuC),
            "S": np.concatenate(SC),
            "Type": np.concatenate(TC),
            "Elow": np.concatenate(ElC),
            "Odd": np.vstack(OddC),
            "Lab": np.vstack(LabC),
        }
    else:
        lst = {
            "Nu": np.zeros(0),
            "S": np.zeros(0),
            "Type": np.zeros(0, dtype=int),
            "Elow": np.zeros(0),
            "Odd": np.zeros((0, 3), dtype=bool),
            "Lab": np.zeros((0, 6), dtype=int),
        }

    _ASYM_ROTOR_CACHE[key] = lst
    while len(_ASYM_ROTOR_CACHE) > _MAX_CACHE_SIZE:
        _ASYM_ROTOR_CACHE.popitem(last=True)

    return lst


def asym_rotor(
    x: np.ndarray,
    A: float, B: float, C: float,
    dA: float, dB: float, dC: float,
    Nu0: float,
    Trot: float,
    Theta: float, Phi: float,
    GW: float, LW: float,
    JMax: int = 0,
    Unit: int = 1,
    SpinAxis: int = 0,
    SpinOdd: float = 1.0,
    T2: float = 0.0,
    W2: float = 0.0,
    lines: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Rotationally resolved band of an asymmetric top (rigid rotor) with unit area.
    Exact Python implementation of Dr. Mario Ulises Delgado-Jaime's `AsymRotor.m`.

    Parameters
    ----------
    x : array-like
        Frequency axis, in the units of the constants (MHz by default).
    A, B, C : float
        Rotational constants of the LOWER state (S0).
    dA, dB, dC : float
        Upper minus lower differences: A' = A + dA, B' = B + dB, C' = C + dC.
    Nu0 : float
        Band origin, in the units of x.
    Trot : float
        Rotational temperature in K.
    Theta, Phi : float
        Direction of the transition dipole moment in degrees:
        Theta from the a-axis, Phi about it from b towards c.
        a-type: (0, 0), b-type: (90, 0), c-type: (90, 90).
    GW, LW : float
        Gaussian (Doppler) and Lorentzian (lifetime) FWHM, in units of x.
    JMax : int, optional
        Highest J of the lower state (0 picks automatically from Trot and C).
    Unit : int, optional
        1 = MHz (default), 2 = GHz, 3 = cm^-1.
    SpinAxis : int, optional
        C2 symmetry axis exchanging identical nuclei: 0 = none, 1 = a, 2 = b, 3 = c.
    SpinOdd : float, optional
        Statistical weight of lower levels odd under C2 relative to even ones
        (for Carbazole, C2 along b exchanges 4 proton pairs -> SpinOdd = 15/17).
    T2, W2 : float, optional
        Second temperature (K) and its weight for a two-temperature supersonic beam.
    lines : bool, optional
        If True, returns `(y, Lines)` where `Lines` is a dict with keys
        `Nu, I, Type, Jl, Kal, Kcl, Ju, Kau, Kcu, Elow`.
    """
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)
    empty_lines = {
        "Nu": np.zeros(0), "I": np.zeros(0), "Type": np.zeros(0, dtype=int),
        "Jl": np.zeros(0, dtype=int), "Kal": np.zeros(0, dtype=int), "Kcl": np.zeros(0, dtype=int),
        "Ju": np.zeros(0, dtype=int), "Kau": np.zeros(0, dtype=int), "Kcu": np.zeros(0, dtype=int),
        "Elow": np.zeros(0)
    }
    if x_arr.size == 0:
        return (y, empty_lines) if lines else y

    JMax = 0 if JMax is None else int(JMax)
    Unit = 1 if Unit is None else int(round(Unit))
    SpinAxis = 0 if SpinAxis is None else int(round(SpinAxis))
    SpinOdd = 1.0 if SpinOdd is None else float(SpinOdd)
    T2 = 0.0 if T2 is None else float(T2)
    W2 = 0.0 if W2 is None else float(W2)

    if Unit == 2:
        kB = 20.83661912      # GHz / K
    elif Unit == 3:
        kB = 0.69503476       # cm^-1 / K
    else:
        kB = 20836.61912      # MHz / K

    Up = [float(A + dA), float(B + dB), float(C + dC)]
    params_check = np.array([A, B, C, Up[0], Up[1], Up[2], Nu0, Trot, GW, LW], dtype=float)
    if not np.all(np.isfinite(params_check)) or np.any(params_check[:6] <= 0) or Trot <= 0:
        raise ValueError("AsymRotor:BadInput — Rotational constants (lower and upper) and Trot must be finite and positive.")

    if JMax <= 0:
        kTm = kB * Trot
        if W2 > 0 and T2 > 0:
            kTm = max(kTm, kB * T2)
        JMax = int(10 * np.ceil(np.sqrt(14.0 * kTm / min(float(C), Up[2])) / 10.0))
    JMax = min(int(round(JMax)), 400)

    Lst = asym_rotor_lines(float(A), float(B), float(C), Up[0], Up[1], Up[2], JMax)
    if Lst["Nu"].size == 0:
        return (y, empty_lines) if lines else y

    # --- Intensities ---
    th_rad = np.deg2rad(float(Theta))
    ph_rad = np.deg2rad(float(Phi))
    mu2 = np.array([
        np.cos(th_rad) ** 2,
        (np.sin(th_rad) * np.cos(ph_rad)) ** 2,
        (np.sin(th_rad) * np.sin(ph_rad)) ** 2,
    ], dtype=float)

    I = mu2[Lst["Type"] - 1] * Lst["S"]
    Pop = np.exp(-Lst["Elow"] / (kB * Trot))
    if W2 > 0 and T2 > 0:
        Pop = Pop + W2 * np.exp(-Lst["Elow"] / (kB * T2))
    I = I * Pop

    if SpinAxis in (1, 2, 3):
        odd_mask = Lst["Odd"][:, SpinAxis - 1]
        I = np.where(odd_mask, I * SpinOdd, I)

    Tot = float(np.sum(I))
    if not (Tot > 0):
        return (y, empty_lines) if lines else y

    I = I / Tot
    Nu = float(Nu0) + Lst["Nu"]

    lines_out = empty_lines
    if lines:
        keep = I > (1e-6 * np.max(I))
        f_idx = np.flatnonzero(keep)
        o_sort = np.argsort(Nu[f_idx])
        f = f_idx[o_sort]
        lines_out = {
            "Nu": Nu[f],
            "I": I[f],
            "Type": Lst["Type"][f],
            "Jl": Lst["Lab"][f, 0],
            "Kal": Lst["Lab"][f, 1],
            "Kcl": Lst["Lab"][f, 2],
            "Ju": Lst["Lab"][f, 3],
            "Kau": Lst["Lab"][f, 4],
            "Kcu": Lst["Lab"][f, 5],
            "Elow": Lst["Elow"][f],
        }

    # --- Exact Voigt Broadening via Fine-Grid FFT ---
    GW = abs(float(GW))
    LW = abs(float(LW))
    fV = 0.5346 * LW + np.sqrt(0.2166 * (LW ** 2) + (GW ** 2))  # Olivero-Longbothum Voigt FWHM

    xc = x_arr.ravel()
    xs = np.sort(xc)
    if not (fV > 0):
        dx = np.diff(xs)
        dx = dx[dx > 0]
        if dx.size == 0:
            return (y, lines_out) if lines else y
        fV = float(np.min(dx))
        GW = fV

    Pad = 60.0 * LW + 6.0 * GW + 2.0 * fV
    gaps = np.flatnonzero(np.diff(xs) > 4.0 * Pad) + 1
    Brk = np.concatenate(([0], gaps, [xs.size]))
    yc = np.zeros_like(xc, dtype=float)

    for s in range(len(Brk) - 1):
        lo = xs[Brk[s]] - Pad
        hi = xs[Brk[s + 1] - 1] + Pad
        in_mask = (Nu >= lo) & (Nu <= hi)
        if not np.any(in_mask):
            continue

        d = fV / 40.0
        if (hi - lo) / d > (2 ** 22):
            d = (hi - lo) / (2 ** 22)

        i0 = int(np.floor(lo / d))
        i1 = int(np.ceil(hi / d))
        N = i1 - i0 + 1

        p = Nu[in_mask] / d - i0
        k = np.floor(p).astype(int)
        t = p - k
        k = np.clip(k, 0, N - 1)

        Sk = np.zeros(N + 1, dtype=float)
        np.add.at(Sk, k, I[in_mask] * (1.0 - t))
        np.add.at(Sk, k + 1, I[in_mask] * t)

        Nf = 1 << int(np.ceil(np.log2(max(2, N + 1))))
        fq_idx = np.concatenate((np.arange(0, Nf // 2 + 1), np.arange(-(Nf // 2 - 1), 0)))
        fq = fq_idx.astype(float) / (Nf * d)

        Kv = np.exp(-np.pi * LW * np.abs(fq) - ((np.pi * GW * fq) ** 2) / (4.0 * np.log(2.0)))
        g = np.real(np.fft.ifft(np.fft.fft(Sk, n=Nf) * Kv)) / d
        xg = np.arange(i0, i1 + 2, dtype=float) * d

        sel = (xc >= lo) & (xc <= hi)
        if np.any(sel):
            cs = CubicSpline(xg, g[:N + 1])
            yc[sel] = cs(xc[sel])

    y_out = yc.reshape(x_arr.shape)
    return (y_out, lines_out) if lines else y_out


def convergent_grid_fit(
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    A: float, B: float, C: float,
    dA: float, dB: float, dC: float,
    Nu0: float,
    Shift: float,
    Trot: float,
    Theta: float, Phi: float,
    GW: float, LW: float,
    JMax: int = 35,
    Unit: int = 1,
    SpinAxis: int = 0,
    SpinOdd: float = 1.0,
    T2: float = 0.0,
    W2: float = 0.0,
    planar_chk: bool = False,
    oblate_chk: bool = False,
    grid_size: int = 5,
    n_levels: int = 3,
    span_p1_mhz: float = 4.0,
    span_p2_mhz: float = 4.0,
    span_shift_mhz: float = 120.0,
    contraction: float = 0.42,
    polish_final: bool = True
) -> Dict[str, object]:
    """
    Método de Rejilla Convergente Multinivel (Coarse-to-Fine Convergent Grid Search)
    basado en la arquitectura de caché de `AsymRotor` (Dr. Mario Ulises Delgado-Jaime):

        "On a grid over the rotational constants each grid point pays for one build,
         and the fit at that point is cheap."

    En cada nivel k = 1 .. n_levels:
      1. Construye una malla 2D de (grid_size × grid_size) nodos alrededor de la mejor
         solución del nivel previo:
         - Si `oblate_chk` es True (ej. Trifenileno D3h, donde B=A y C=A/2):
           barre p1 = A (Estado Base S0) y p2 = dA (Cambio S1 <- S0).
         - Si `oblate_chk` es False (ej. Carbazol o rotor asimétrico general):
           barre p1 = dA y p2 = dB, determinando dC mediante la condición plana
           C' = A'B'/(A'+B') (si `planar_chk=True`) o conservando el defecto de inercia.
      2. En cada nodo (p1, p2) evalúa `asym_rotor` UNA SOLA VEZ sobre una malla extendida
         y barre instantáneamente 21 valores de corrimiento de calibración (`Shift`) junto
         con la solución analítica exacta de mínimos cuadrados para `Int1` (>0) y `Offset`,
         sin re-diagonalizar matrices ni repetir la FFT.
      3. Contrae el radio de la rejilla por el factor `contraction` en cada nivel sucesivo
         hasta converger al mínimo global y opcionalmente aplica un pulido sub-MHz final.
    """
    from scipy.optimize import least_squares

    x_arr = np.asarray(x_exp, dtype=float).ravel()
    y_arr = np.asarray(y_exp, dtype=float).ravel()

    A0, B0, C0 = float(A), float(B), float(C)
    dA0, dB0, dC0 = float(dA), float(dB), float(dC)
    Nu0_f = float(Nu0)
    Shift0 = float(Shift)

    if oblate_chk:
        B0 = A0
        dB0 = dA0
    if planar_chk:
        C0 = planar_top_c(A0, B0)
        dC0 = planar_top_c(A0 + dA0, B0 + dB0) - C0

    # Defecto rotacional inicial en dC para cuando planar_chk es False
    dc_planar_init = planar_top_c(A0 + dA0, B0 + dB0) - planar_top_c(A0, B0)
    dc_defect_offset = dC0 - dc_planar_init

    grid_size = max(3, min(int(grid_size), 11))
    n_levels = max(1, min(int(n_levels), 6))
    j_grid = min(int(JMax) if int(JMax) > 0 else 24, 22)

    # Pre-calcular estadísticas de y_exp para regresión lineal analítica vectorizada
    y_mean = float(np.mean(y_arr))
    dy_exp = y_arr - y_mean
    norm_dy_exp = float(np.linalg.norm(dy_exp)) + 1e-15

    # Eje extendido uniforme para evaluar traslaciones de `Shift` por interpolación rápida
    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    max_pad = max(250.0, abs(float(span_shift_mhz)) * 1.5)
    n_pad = min(4500, max(2000, int((x_max - x_min + 2.0 * max_pad) / 2.5)))
    x_pad = np.linspace(x_min - max_pad, x_max + max_pad, n_pad)

    def _params_from_node(p1: float, p2: float) -> Tuple[float, float, float, float, float, float]:
        if oblate_chk:
            a_v = max(20.0, float(p1))
            b_v = a_v
            c_v = planar_top_c(a_v, b_v) if planar_chk else max(10.0, C0 * (a_v / max(A0, 1e-6)))
            da_v = float(p2)
            db_v = da_v
            if planar_chk:
                cu_v = planar_top_c(a_v + da_v, b_v + db_v)
                dc_v = cu_v - c_v
            else:
                dc_v = 0.5 * da_v
        else:
            a_v, b_v = A0, B0
            c_v = planar_top_c(a_v, b_v) if planar_chk else C0
            da_v = float(p1)
            db_v = float(p2)
            if planar_chk:
                dc_v = planar_top_c(a_v + da_v, b_v + db_v) - c_v
            else:
                dc_v = (planar_top_c(a_v + da_v, b_v + db_v) - planar_top_c(a_v, b_v)) + dc_defect_offset
        return a_v, b_v, c_v, da_v, db_v, dc_v

    node_cache: Dict[Tuple[float, float, float, float], Dict[str, float]] = {}

    def _eval_node(p1: float, p2: float, sh_center: float, sh_span: float) -> Dict[str, float]:
        cache_key = (round(p1, 6), round(p2, 6), round(sh_center, 3), round(sh_span, 3))
        if cache_key in node_cache:
            return node_cache[cache_key]

        a_v, b_v, c_v, da_v, db_v, dc_v = _params_from_node(p1, p2)
        if (a_v + da_v) <= 5.0 or (b_v + db_v) <= 5.0 or (c_v + dc_v) <= 5.0:
            res_bad = {
                "rms": 1e9, "corr": 0.0, "shift": sh_center,
                "int1": 1.0, "offset": y_mean,
                "A": a_v, "B": b_v, "C": c_v, "dA": da_v, "dB": db_v, "dC": dc_v
            }
            node_cache[cache_key] = res_bad
            return res_bad

        # Una sola llamada a `asym_rotor` en este nodo de la rejilla (con Shift = sh_center)
        y_pad = asym_rotor(
            x_pad, a_v, b_v, c_v, da_v, db_v, dc_v,
            Nu0_f + sh_center, Trot, Theta, Phi, GW, LW,
            JMax=j_grid, Unit=Unit, SpinAxis=SpinAxis, SpinOdd=SpinOdd, T2=T2, W2=W2, lines=False
        )

        # Barrer 21 desplazamientos finos de calibración (`Shift`) sin re-diagonalizar
        deltas = np.linspace(-sh_span, sh_span, 21) if sh_span > 1e-4 else np.array([0.0])
        best_rms = 1e9
        best_corr = -1.0
        best_sh = sh_center
        best_int1 = 1.0
        best_off = y_mean

        for d_sh in deltas:
            cand_sh = sh_center + float(d_sh)
            y_s = np.interp(x_arr - float(d_sh), x_pad, y_pad)
            ys_mean = float(np.mean(y_s))
            dys = y_s - ys_mean
            denom = float(np.dot(dys, dys))
            if denom <= 1e-20:
                continue
            cov_xy = float(np.dot(dys, dy_exp))
            int1_val = max(1e-3, cov_xy / denom)
            off_val = y_mean - int1_val * ys_mean
            resid = y_arr - (int1_val * y_s + off_val)
            rms_val = float(np.sqrt(np.mean(resid ** 2)))
            corr_val = float(cov_xy / (np.sqrt(denom) * norm_dy_exp)) * 100.0
            if rms_val < best_rms:
                best_rms = rms_val
                best_corr = corr_val
                best_sh = cand_sh
                best_int1 = int1_val
                best_off = off_val

        out = {
            "rms": best_rms,
            "corr": best_corr,
            "shift": best_sh,
            "int1": best_int1,
            "offset": best_off,
            "A": a_v, "B": b_v, "C": c_v,
            "dA": da_v, "dB": db_v, "dC": dc_v,
        }
        node_cache[cache_key] = out
        return out

    # Centros iniciales según modo de simetría
    if oblate_chk:
        cur_p1 = A0
        cur_p2 = dA0
        axis1_label = "A″ Basal S₀ (MHz)"
        axis2_label = "dA = A′ - A″ (MHz)"
    else:
        cur_p1 = dA0
        cur_p2 = dB0
        axis1_label = "dA = A′ - A″ (MHz)"
        axis2_label = "dB = B′ - B″ (MHz)"

    cur_sh = Shift0
    cur_span1 = max(0.05, float(span_p1_mhz))
    cur_span2 = max(0.05, float(span_p2_mhz))
    cur_sh_span = max(0.0, float(span_shift_mhz))

    level_history = []
    best_overall = None
    total_grid_nodes = 0

    for lv in range(1, n_levels + 1):
        p1_vec = np.linspace(cur_p1 - cur_span1, cur_p1 + cur_span1, grid_size)
        p2_vec = np.linspace(cur_p2 - cur_span2, cur_p2 + cur_span2, grid_size)
        rms_mat = np.zeros((grid_size, grid_size), dtype=float)
        corr_mat = np.zeros((grid_size, grid_size), dtype=float)

        lv_best = None
        lv_best_ij = (grid_size // 2, grid_size // 2)

        for j_idx, p2_val in enumerate(p2_vec):
            for i_idx, p1_val in enumerate(p1_vec):
                total_grid_nodes += 1
                node_res = _eval_node(float(p1_val), float(p2_val), cur_sh, cur_sh_span)
                rms_mat[j_idx, i_idx] = node_res["rms"]
                corr_mat[j_idx, i_idx] = node_res["corr"]
                if lv_best is None or node_res["rms"] < lv_best["rms"]:
                    lv_best = dict(node_res)
                    lv_best["p1"] = float(p1_val)
                    lv_best["p2"] = float(p2_val)
                    lv_best_ij = (i_idx, j_idx)

        cur_p1 = lv_best["p1"]
        cur_p2 = lv_best["p2"]
        cur_sh = lv_best["shift"]
        if best_overall is None or lv_best["rms"] < best_overall["rms"]:
            best_overall = dict(lv_best)

        level_history.append({
            "level": lv,
            "p1_vec": p1_vec,
            "p2_vec": p2_vec,
            "rms_mat": rms_mat,
            "corr_mat": corr_mat,
            "span1": cur_span1,
            "span2": cur_span2,
            "step1": (2.0 * cur_span1) / max(1, grid_size - 1),
            "step2": (2.0 * cur_span2) / max(1, grid_size - 1),
            "best_p1": cur_p1,
            "best_p2": cur_p2,
            "best_shift": cur_sh,
            "best_rms": lv_best["rms"],
            "best_corr": lv_best["corr"],
            "best_ij": lv_best_ij,
        })

        # Contraer la ventana para el siguiente nivel (Zoom-In Convergente)
        cur_span1 *= float(contraction)
        cur_span2 *= float(contraction)
        cur_sh_span *= float(contraction)

    # Pulido continuo sub-MHz final a partir del mejor punto de la Rejilla Convergente
    if polish_final:
        def _polish_res(vec):
            nd = _eval_node(float(vec[0]), float(vec[1]), float(vec[2]), 0.0)
            y_u = asym_rotor(
                x_arr, nd["A"], nd["B"], nd["C"], nd["dA"], nd["dB"], nd["dC"],
                Nu0_f + float(vec[2]), Trot, Theta, Phi, GW, LW,
                JMax=j_grid, Unit=Unit, SpinAxis=SpinAxis, SpinOdd=SpinOdd, T2=T2, W2=W2, lines=False
            )
            M_lin = np.column_stack((y_u, np.ones_like(y_u)))
            coef, _, _, _ = np.linalg.lstsq(M_lin, y_arr, rcond=None)
            int_c = max(1e-3, float(coef[0]))
            return y_arr - (int_c * y_u + float(coef[1]))

        p_init = np.array([best_overall["p1"], best_overall["p2"], best_overall["shift"]], dtype=float)
        lsq = least_squares(_polish_res, p_init, diff_step=0.01, max_nfev=12)
        pol_node = _eval_node(float(lsq.x[0]), float(lsq.x[1]), float(lsq.x[2]), 0.0)
        if pol_node["rms"] <= best_overall["rms"]:
            pol_node["p1"] = float(lsq.x[0])
            pol_node["p2"] = float(lsq.x[1])
            best_overall = pol_node

    # Evaluar con el JMax completo del usuario en el óptimo global encontrado
    y_final = asym_rotor(
        x_arr,
        best_overall["A"], best_overall["B"], best_overall["C"],
        best_overall["dA"], best_overall["dB"], best_overall["dC"],
        Nu0_f + best_overall["shift"],
        Trot, Theta, Phi, GW, LW,
        JMax=int(JMax), Unit=Unit, SpinAxis=SpinAxis, SpinOdd=SpinOdd, T2=T2, W2=W2, lines=False
    )
    M_lin = np.column_stack((y_final, np.ones_like(y_final)))
    coef, _, _, _ = np.linalg.lstsq(M_lin, y_arr, rcond=None)
    final_int1 = max(1.0, float(coef[0]))
    final_off = float(coef[1])
    final_model = final_int1 * y_final + final_off
    final_rms = float(np.sqrt(np.mean((y_arr - final_model) ** 2)))
    final_corr = float(np.corrcoef(y_arr, final_model)[0, 1] * 100.0) if np.std(final_model) > 1e-12 else 0.0

    best_overall["int1"] = final_int1
    best_overall["offset"] = final_off
    best_overall["rms"] = final_rms
    best_overall["corr"] = final_corr

    return {
        "best": best_overall,
        "levels": level_history,
        "axis1_label": axis1_label,
        "axis2_label": axis2_label,
        "total_grid_nodes": total_grid_nodes,
        "unique_hamiltonians": len(node_cache),
    }

