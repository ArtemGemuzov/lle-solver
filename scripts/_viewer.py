import json
from pathlib import Path

import numpy as np

from lle_solver.grid import PeriodicGrid
from lle_solver.types import State, Time

from _common import count_peaks

PHASE_COLORS = ("#4f9dde", "#52c785", "#e0654f", "#b57edc", "#8a93a6")


class FrameRecorder:
    def __init__(
            self,
            grid: PeriodicGrid,
            interval: Time,
            field_downsample: int = 4,
            spectrum_downsample: int = 4,
    ) -> None:
        self.interval = interval
        self.field_downsample = field_downsample
        self.spectrum_downsample = spectrum_downsample
        self.theta = grid.theta[::field_downsample]
        self.modes = np.fft.fftshift(grid.wavenumbers)[::spectrum_downsample]
        self.frames: list[dict] = []
        self._last_time = -np.inf

    def record(self, state: State, time: Time, detuning: float, phase: str) -> None:
        if time - self._last_time < self.interval:
            return
        self._last_time = time

        intensity = np.abs(state) ** 2
        spectrum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(state))) + 1e-12)
        spectrum -= spectrum.max()

        self.frames.append({
            "tau": round(float(time), 4),
            "zeta0": round(float(detuning), 4),
            "P": round(float(intensity.mean()), 5),
            "N": count_peaks(state),
            "phase": phase,
            "field": [round(float(x), 4) for x in intensity[::self.field_downsample]],
            "spec": [round(float(x), 2) for x in spectrum[::self.spectrum_downsample]],
        })


def write_viewer(
        name: str,
        recorder: FrameRecorder,
        title: str,
        subtitle: str,
        phases: dict[str, str],
        **parameters: float | int,
) -> None:
    meta = {
        "title": title,
        "subtitle": subtitle,
        "phases": {
            key: {"label": label, "color": PHASE_COLORS[index % len(PHASE_COLORS)]}
            for index, (key, label) in enumerate(phases.items())
        },
        "theta_ds": [round(float(x), 4) for x in recorder.theta],
        "modes_ds": [round(float(x), 3) for x in recorder.modes],
        "parameters": parameters,
    }
    data = json.dumps(
        {"meta": meta, "frames": recorder.frames},
        separators=(",", ":"), ensure_ascii=False,
    )

    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    Path(f"{name}.json").write_text(data, encoding="utf-8")
    Path(f"{name}.html").write_text(
        template.replace("__DATA_JSON__", data), encoding="utf-8"
    )
    print(f"{name}.html: {len(recorder.frames)} кадров, "
          f"{Path(f'{name}.html').stat().st_size / 1024:.0f} КБ")