from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CKKSParameters:
    poly_modulus_degree: int
    coeff_modulus_bits: list[int]
    scale_bits: int
    security_bits: int

    @property
    def total_modulus_bits(self) -> int:
        return sum(self.coeff_modulus_bits)


class CKKSParameterSelector:
    """Heurisztikus paraméterválasztás 128-bit klasszikus biztonságra."""

    # Konzervatív felső korlátok 128-bit biztonságra (SEAL/OpenFHE jellegű)
    MAX_LOGQ_BY_N = {
        4096: 109,
        8192: 218,
        16384: 438,
        32768: 881,
    }

    def __init__(self, security_bits: int = 128, scale_bits: int = 40):
        self.security_bits = security_bits
        self.scale_bits = scale_bits

    def select(self, multiplicative_depth: int, margin_levels: int = 2) -> CKKSParameters:
        levels = multiplicative_depth + margin_levels
        # első prime + levels darab scale prime + záró special prime
        coeff_modulus_bits = [60] + [self.scale_bits] * levels + [60]
        needed_logq = sum(coeff_modulus_bits)

        for n, max_logq in self.MAX_LOGQ_BY_N.items():
            if needed_logq <= max_logq:
                return CKKSParameters(
                    poly_modulus_degree=n,
                    coeff_modulus_bits=coeff_modulus_bits,
                    scale_bits=self.scale_bits,
                    security_bits=self.security_bits,
                )

        raise ValueError(
            "A szükséges modulus túl nagy bootstrapping nélkül ezen biztonsági célhoz. "
            "Csökkentsd a mélységet vagy használj kisebb skálát."
        )
