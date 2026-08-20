from dataclasses import dataclass

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from lle_solver.evolution import SplitScheme, evolve
from lle_solver.grid import PeriodicGrid
from lle_solver.types import State, StepSize


@dataclass
class History:
    times: np.ndarray
    intensity: np.ndarray
    spectrum: np.ndarray
    energy: np.ndarray


def run(
        scheme: SplitScheme,
        grid: PeriodicGrid,
        state: State,
        step_size: StepSize,
        num_steps: int,
        title: str = "",
        steps_per_frame: int = 400,
        save_every: int = 100,
        live: bool = True,
) -> tuple[State, History]:
    times, intensities, spectra, energies = [], [], [], []

    if live:
        plt.ion()
        figure, axes = plt.subplots()
        line, = axes.plot(grid.theta, np.abs(state) ** 2, "b")
        axes.set_xlim(-grid.length / 2, grid.length / 2)
        axes.set_xlabel(r"$\theta$")
        axes.set_ylabel(r"$|u|^2$")

    for step, time, state in evolve(scheme, state, num_steps, step_size):
        if live and (step == 1 or step % steps_per_frame == 0):
            intensity = np.abs(state) ** 2

            line.set_ydata(intensity)
            axes.set_ylim(intensity.min() - 0.2, intensity.max() + 0.2)
            axes.set_title(f"{title} t = {time:.2f}")
            figure.canvas.draw_idle()
            plt.pause(0.001)

        if step % save_every == 0:
            if not np.isfinite(state).all():
                raise RuntimeError(f"расходимость на шаге {step}, t = {time:.3f}")
            intensity = np.abs(state) ** 2
            intensities.append(intensity)
            spectra.append(np.abs(fft(state)) ** 2)
            energies.append(float(intensity.sum() * grid.spacing))
            times.append(time)

    if live:
        plt.ioff()

    return state, History(
        np.array(times), np.array(intensities), np.array(spectra), np.array(energies)
    )


def plot_maps(grid: PeriodicGrid, history: History) -> None:
    intensity = history.intensity / history.intensity.max()
    spectrum = np.fft.fftshift(history.spectrum / history.spectrum.max(), axes=1)
    modes = np.fft.fftshift(grid.wavenumbers)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    image = axes[0].imshow(
        intensity, aspect="auto", origin="lower",
        extent=[grid.theta[0], grid.theta[-1], history.times[0], history.times[-1]],
    )
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$t$")
    figure.colorbar(image, ax=axes[0], orientation="horizontal", location="top")

    image = axes[1].imshow(
        spectrum, aspect="auto", origin="lower", norm=LogNorm(vmin=1e-6, vmax=1.0),
        extent=[modes[0], modes[-1], history.times[0], history.times[-1]],
    )
    axes[1].set_xlabel(r"$m$")
    figure.colorbar(image, ax=axes[1], orientation="horizontal", location="top")

    figure.tight_layout()
    plt.show()