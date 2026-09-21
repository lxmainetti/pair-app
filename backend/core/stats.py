"""Pure-math psychometric helpers — no model dependency."""

import math


def get_r(pairs: dict, a: str, b: str) -> float:
    """Symmetric lookup; diagonal = 1."""
    if a == b:
        return 1.0
    return pairs.get((min(a, b), max(a, b)), pairs.get((max(a, b), min(a, b)), 0.0))


def build_matrix(items: list[str], pairs: dict) -> list[list[float]]:
    return [[get_r(pairs, a, b) for b in items] for a in items]


def build_cross_matrix(items_a: list[str], items_b: list[str], pairs: dict) -> list[list[float]]:
    return [[get_r(pairs, a, b) for b in items_b] for a in items_a]


def cronbach_alpha(items: list[str], pairs: dict) -> float:
    """Standardised-items Cronbach's α = (n/(n-1)) * (1 - n / (n + 2*Σr_ij))"""
    n = len(items)
    if n < 2:
        return float("nan")
    sum_r = sum(
        get_r(pairs, items[i], items[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    return round((n / (n - 1)) * (1 - n / (n + 2 * sum_r)), 4)


def inter_scale_r(items_a: list[str], items_b: list[str], pairs: dict) -> float:
    """
    Analytical inter-scale correlation from inter-item rs.
    Var(ΣA) = n_A + 2·Σ_{i<j ∈ A} r_ij
    Cov(ΣA, ΣB) = Σ_{i∈A, j∈B} r_ij
    r(A,B) = Cov / sqrt(Var_A · Var_B)
    """
    n_a, n_b = len(items_a), len(items_b)
    var_a = n_a + 2 * sum(
        get_r(pairs, items_a[i], items_a[j])
        for i in range(n_a) for j in range(i + 1, n_a)
    )
    var_b = n_b + 2 * sum(
        get_r(pairs, items_b[i], items_b[j])
        for i in range(n_b) for j in range(i + 1, n_b)
    )
    cov = sum(get_r(pairs, a, b) for a in items_a for b in items_b)
    denom = math.sqrt(var_a * var_b)
    return round(cov / denom, 4) if denom else 0.0


def alpha_interpretation(alpha: float) -> str:
    if alpha >= 0.90: return "Excellent"
    if alpha >= 0.80: return "Good"
    if alpha >= 0.70: return "Acceptable"
    if alpha >= 0.60: return "Questionable"
    if alpha >= 0.50: return "Poor"
    return "Unacceptable"
