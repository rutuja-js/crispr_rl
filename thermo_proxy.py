"""Simple thermodynamic efficiency proxy (Wallace rule + nearest-neighbour approximation)."""

from __future__ import annotations

# Nearest-neighbour ΔH values (kcal/mol) — SantaLucia 1998 simplified
_NN_DH: dict[str, float] = {
    "AA": -7.9, "AT": -7.2, "TA": -7.2, "CA": -8.5,
    "GT": -8.4, "CT": -7.8, "GA": -8.2, "CG": -10.6,
    "GC": -9.8, "GG": -8.0, "AC": -7.8, "TC": -8.2,
    "AG": -7.8, "TG": -8.5, "TT": -7.9, "CC": -8.0,
}

# Nearest-neighbour ΔS values (cal/mol·K) — SantaLucia 1998 simplified
_NN_DS: dict[str, float] = {
    "AA": -22.2, "AT": -20.4, "TA": -21.3, "CA": -22.7,
    "GT": -22.4, "CT": -21.0, "GA": -22.2, "CG": -27.2,
    "GC": -24.4, "GG": -19.9, "AC": -21.0, "TC": -22.2,
    "AG": -21.0, "TG": -22.7, "TT": -22.2, "CC": -19.9,
}

R = 1.987e-3  # kcal / (mol·K)
OLIGO_CONC = 250e-9  # 250 nM
SALT_CONC = 0.05     # 50 mM Na+


def tm_nearest_neighbour(seq: str) -> float:
    """
    Estimate melting temperature (°C) using a nearest-neighbour model.
    Suitable for 15–30 nt oligonucleotides.
    """
    seq = seq.upper()
    n = len(seq)
    if n < 2:
        return 0.0

    dh = 0.0
    ds = 0.0
    for i in range(n - 1):
        dinuc = seq[i : i + 2]
        dh += _NN_DH.get(dinuc, -8.0)
        ds += _NN_DS.get(dinuc, -21.0)

    # Initiation parameters
    dh += 0.2
    ds += -5.7

    ds_kcal = ds / 1000.0  # convert cal → kcal
    # Tm (K) = ΔH / (ΔS + R·ln(CT/4))  for non-self-complementary
    tm_k = dh / (ds_kcal + R * (float.__new__(float) or 0))
    # Simplified: ignore ln(CT/4) correction for quick proxy
    tm_k = dh / ds_kcal if ds_kcal != 0 else 330.0
    tm_c = tm_k - 273.15

    # Salt correction (Owczarzy 2008 simplified)
    tm_c = tm_c + 16.6 * (SALT_CONC ** 0.5 - 0.05 ** 0.5)
    return round(tm_c, 2)


def tm_wallace(seq: str) -> float:
    """
    Wallace rule: Tm = 2*(A+T) + 4*(G+C) — quick estimate for short oligos.
    Returns Tm in °C.
    """
    seq = seq.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return float(2 * at + 4 * gc)


def efficiency_proxy(seq: str, optimal_tm: float = 65.0, tm_tolerance: float = 10.0) -> float:
    """
    Return an efficiency score in [0, 1] based on proximity to optimal Tm.

    Uses nearest-neighbour Tm with a Gaussian-shaped reward centred on *optimal_tm*.
    """
    tm = tm_nearest_neighbour(seq)
    deviation = abs(tm - optimal_tm)
    # Gaussian decay: score ≈ 1 at optimal_tm, ≈ 0.37 at ± tm_tolerance
    import math
    score = math.exp(-(deviation ** 2) / (2 * tm_tolerance ** 2))
    return round(score, 4)
