"""Адаптивное взвешивание потерь ReLoBRaLo."""

import deepxde as dde
import numpy as np
import torch


class ReLoBRaLoTimePDE(dde.data.TimePDE):
    """Обновляет общий вес каждой пары компонент LLE и начального условия."""

    def __init__(
        self,
        *args,
        alpha: float,
        temperature: float,
        lookback_probability: float,
        seed: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.lookback_probability = lookback_probability
        self.rng = np.random.default_rng(seed)
        self._weight_history = []
        self.initial = None
        self.previous = None
        self.weights = None
        self.update_pending = False

    def state_dict(self):
        return {
            "initial": self.initial,
            "previous": self.previous,
            "weights": self.weights,
            "weight_history": self._weight_history,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self.initial = state["initial"]
        self.previous = state["previous"]
        self.weights = state["weights"]
        self._weight_history = state["weight_history"]
        self.rng.bit_generator.state = state["rng"]
        self.update_pending = False

    def before_optimizer_step(self, optimizer, args, kwargs):
        # Adam вызывает closure один раз. Вычисления для вывода loss проходят вне optimizer.step.
        self.update_pending = True

    def relative_weights(self, current, reference):
        ratios = current / reference.clamp_min(1e-12)
        logits = (ratios - ratios.max()) / self.temperature
        return current.numel() * torch.softmax(logits, dim=0)

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        losses = super().losses_train(targets, outputs, loss_fn, inputs, model, aux=aux)
        if not self.update_pending:
            return losses

        self.update_pending = False
        pde_u, pde_v, ic_u, ic_v = losses
        current = torch.stack((pde_u + pde_v, ic_u + ic_v)).detach()
        if self.initial is None:
            self.initial = current
            self.weights = torch.ones_like(current)
        else:
            historical = self.weights
            if self.rng.random() < self.lookback_probability:
                historical = self.relative_weights(current, self.initial)
            recent = self.relative_weights(current, self.previous)
            alpha = self.alpha
            self.weights = alpha * historical + (1 - alpha) * recent

        self.previous = current
        model.loss_weights = self.weights.repeat_interleave(2)
        self._weight_history.append(torch.cat((current, self.weights)))
        return losses

    @property
    def weight_history(self) -> tuple[torch.Tensor, ...]:
        """Невзвешенные групповые потери и использованные веса по шагам."""
        return tuple(self._weight_history)
