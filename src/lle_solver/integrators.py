from lle_solver.types import Derivative, State, StepSize


def rk2_step(
        derivative: Derivative,
        state: State,
        step_size: StepSize,
) -> State:
    k1 = derivative(state)
    state_mid = state + step_size * k1 / 2.0
    k2 = derivative(state_mid)
    return state + step_size * k2


def rk4_step(
        derivative: Derivative,
        state: State,
        step_size: StepSize,
) -> State:
    k1 = derivative(state)
    k2 = derivative(state + step_size * k1 / 2.0)
    k3 = derivative(state + step_size * k2 / 2.0)
    k4 = derivative(state + step_size * k3)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0