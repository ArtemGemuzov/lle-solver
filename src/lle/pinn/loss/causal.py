"""Взвешивание невязки по временным полосам."""

import deepxde as dde
import torch


class CausalTimePDE(dde.data.TimePDE):
    """Взвешивает невязку LLE по последовательным временным полосам."""

    def __init__(self, *args, bands: int, epsilon: float, time_end: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.bands = bands
        self.epsilon = epsilon
        self.time_end = time_end
        self._weight_history = []
        self.update_pending = False

    def state_dict(self):
        return {"epsilon": self.epsilon, "weight_history": self._weight_history}

    def load_state_dict(self, state):
        self.epsilon = state["epsilon"]
        self._weight_history = state["weight_history"]
        self.update_pending = False

    def before_optimizer_step(self, optimizer, args, kwargs):
        self.update_pending = True

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        # DeepXDE добавляет копии точек IC перед точками невязки уравнения.
        times = inputs[sum(self.num_bcs) :, 1]
        band_count = self.bands
        bands = (times / self.time_end * band_count).long().clamp_max(band_count - 1)
        zeros = times.new_zeros(band_count)
        counts = zeros.index_add(0, bands, torch.ones_like(times))

        def band_mse(_, residual):
            sums = zeros.index_add(0, bands, residual[:, 0].square())
            return sums / counts.clamp_min(1)

        pde_u, pde_v, ic_u, ic_v = super().losses_train(
            targets,
            outputs,
            [band_mse, band_mse, loss_fn, loss_fn],
            inputs,
            model,
            aux=aux,
        )
        residual_by_band = (pde_u + pde_v).detach()
        previous = torch.cat((zeros[:1], residual_by_band[:-1].cumsum(dim=0)))
        weights = torch.exp(-self.epsilon * previous)
        if self.update_pending:
            self.update_pending = False
            self._weight_history.append(torch.cat((residual_by_band, weights)))

        return [(weights * pde_u).mean(), (weights * pde_v).mean(), ic_u, ic_v]

    @property
    def weight_history(self) -> tuple[torch.Tensor, ...]:
        """Невзвешенные потери и веса временных полос по шагам."""
        return tuple(self._weight_history)

    @property
    def last_min_weight(self) -> float:
        return self._weight_history[-1][-self.bands :].min().item()
