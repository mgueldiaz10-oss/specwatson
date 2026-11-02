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
