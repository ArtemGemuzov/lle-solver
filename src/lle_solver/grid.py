import numpy as np


class PeriodicGrid:
    def __init__(self, num_points: int, length: float = 2*np.pi) -> None:
        self.num_points = num_points
        self.length = length
        self.spacing = length / num_points
        
        self.theta = np.linspace(-length / 2, length / 2, num_points, endpoint=False)
        self.wavenumbers = (
            2 * np.pi / length * np.fft.fftfreq(num_points, d=1.0 / num_points)
        )