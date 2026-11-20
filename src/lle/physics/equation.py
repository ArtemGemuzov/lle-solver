"""Физические параметры нормированной LLE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLEParameters:
    """dpsi/dt = -(1 + i*delta)*psi - i*beta2/2*d2psi/dtheta2 + i*|psi|^2*psi + F."""

    detuning: float
    pump_amplitude: float
    beta2: float
