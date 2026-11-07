from lle_analysis.fields import relative_field_l2_error, spatial_contrast
from lle_solver.simulations.evolution import FieldEvolution, Intensity


DEFAULT_STATIONARY_TOLERANCE = 1e-3
DEFAULT_HOMOGENEOUS_CONTRAST = 0.01


def is_stationary(
    evolution: FieldEvolution, *, tolerance: float = DEFAULT_STATIONARY_TOLERANCE,
) -> bool:
    """Проверяет стационарность на переданных снимках комплексного поля (оси: время, пространство).

    Для каждого снимка вычисляется L2-норма разности с последним снимком,
    делённая на L2-норму последнего. Максимум должен быть строго меньше tolerance.
    При нулевом эталоне ошибка равна нулю для нулевого снимка и бесконечности для остальных.
    """
    if evolution.times.size < 2:
        raise ValueError("Для проверки стационарности нужны хотя бы два снимка")
    return bool(relative_field_l2_error(evolution, evolution[-1]).max() < tolerance)


def is_homogeneous(
    intensity: Intensity, *, tolerance: float = DEFAULT_HOMOGENEOUS_CONTRAST,
) -> bool:
    """Проверяет пространственную однородность интенсивности одного снимка.

    Размах интенсивности делится на её пространственное среднее.
    Отношение должно быть строго меньше tolerance.
    У нулевого профиля контраст равен нулю.
    """
    return spatial_contrast(intensity) < tolerance
