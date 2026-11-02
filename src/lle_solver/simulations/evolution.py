from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray


from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class FieldEvolution:
    theta: NDArray[np.float64]
    times: NDArray[np.float64]
    fields: NDArray[np.complex128]

    FILENAME: ClassVar[str] = "field_evolution.npz"

    @property
    def intensity(self) -> NDArray[np.float64]:
        return np.abs(self.fields) ** 2

    def save(self, directory: Path) -> None:
        with (directory / self.FILENAME).open("xb") as target:
            np.savez_compressed(
                target,
                **{
                    field.name: getattr(self, field.name)
                    for field in fields(self)
                },
            )

    @classmethod
    def load(cls, directory: Path) -> Self:
        with np.load(directory / cls.FILENAME, allow_pickle=False) as data:
            return cls(
                **{
                    field.name: data[field.name]
                    for field in fields(cls)
                }
            )
