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