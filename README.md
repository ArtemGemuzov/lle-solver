# LLE-Solver

## Решение уравнения Лугиато–Лефевера

Решение уравнения Лугиато–Лефевера (LLE) методом Split-Step Fourier.

Постановка задачи:

$$
\begin{cases}
\displaystyle \frac{\partial \psi}{\partial t}
= F - (1 + i\delta)\psi
- i\frac{\beta_2}{2}\frac{\partial^2\psi}{\partial\theta^2}
+ i|\psi|^2\psi,
& \theta \in [-\pi, \pi), \quad t > 0, \\[0.5em]
\psi(\theta + 2\pi, t) = \psi(\theta, t),
& t \geq 0, \\
\psi(\theta, 0) = \psi_0(\theta),
& \theta \in [-\pi, \pi).
\end{cases}
$$

где:

- $\psi(\theta, t)$ — комплексное поле;
- $t$ — время;
- $\theta$ — угловая пространственная координата;
- $F$ — амплитуда внешней накачки;
- $\Delta$ — расстройка;
- $\beta_2$ — коэффициент дисперсии второго порядка.

## Установка и запуск

Все команды выполняются из корня проекта. `pyproject.toml` описывает зависимости,
`uv.lock` фиксирует их версии и источники, а `.venv` содержит установленные пакеты.

### Lock-файл

После изменения зависимостей в `pyproject.toml` обновите lock-файл командой `uv lock`.
Проверить его актуальность можно командой `uv lock --check`.

### Окружение

`uv sync --locked` создаёт или обновляет `.venv` по актуальному lock-файлу.
Для спектрального решателя и блокнотов выполните следующую команду.

```bash
uv sync --locked --extra notebooks
```

Для PINN на CPU и блокнотов используйте профиль `pinn`.

```bash
uv sync --locked --extra pinn --extra notebooks
```

Для PINN на NVIDIA с CUDA 12.8 и блокнотов используйте профиль `pinn-cuda`.

```bash
uv sync --locked --extra pinn-cuda --extra notebooks
```

Профили `pinn` и `pinn-cuda` взаимоисключающие. Для CUDA нужен совместимый
драйвер NVIDIA. `uv sync` без `--extra` устанавливает базовые зависимости
и удаляет пакеты дополнительных профилей из окружения.

### Запуск скриптов

`uv run --locked` синхронизирует окружение и запускает скрипт.

```bash
uv run --locked scripts/gaussian.py
```

Для PINN сначала укажите бэкенд PyTorch в PowerShell.

```powershell
$env:DDE_BACKEND = 'pytorch'
```

Для обучения на CPU используйте профиль `pinn`.

```powershell
uv run --locked --extra pinn scripts/pinn_gaussian.py
```

Для обучения на NVIDIA используйте профиль `pinn-cuda`.

```powershell
uv run --locked --extra pinn-cuda scripts/pinn_gaussian.py
```

Если нужный профиль уже установлен, `--no-sync` запускает скрипт в текущем
окружении проекта без синхронизации.

```powershell
uv run --no-sync scripts/pinn_gaussian.py
```
