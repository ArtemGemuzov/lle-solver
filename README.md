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

## Модель Паскуази

Команда решает уравнения (S16)-(S17) для поля микрокольца и супермод
усиливающей петли методом расщепления Стрэнга. Быстрая координата
`x` имеет период 1, время измеряется в оборотах петли.

```bash
lle pasquazi solve equation.n=5
```

Для расчета с незакоммиченными изменениями добавьте `--allow-dirty`
к `lle ssfm solve` или `lle pasquazi solve`. Команда выполнит расчет и
сохранит Run. Поле `git_dirty` в `metadata.json` отражает фактическое
состояние рабочего дерева при создании запуска.

`equation.n=N` задает индексы супермод от `-N` до `N` включительно.
По умолчанию `N=0`, коэффициенты уравнения взяты из примера на рис. S3.
Начальное поле кольца задано гауссовым импульсом (`initial.amplitude`,
`initial.width`), все поля петли в начале равны нулю. Число точек задает
`num_points`, шаг и длительность расчета задают `solver.time_step` и
`solver.time_end`, интервал записи задает `recording.interval`.
`equation.detuning` задает частотный сдвиг Delta, `equation.fsr_mismatch`
задает рассогласование delta. [Статья Bao и соавторов](https://doi.org/10.1038/s41566-019-0379-5).

`lle.storage.load_pasquazi(run.path)` возвращает поле кольца `ring` и
комплексный массив `loop` с осями (время, супермода, координата).
Поле супермоды `q` доступно как `loop[:, q + N, :]`.
`lle.storage.load_evolution(run.path)` загружает только поле кольца.

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

### Команды lle

`uv run --locked` синхронизирует окружение и запускает команду.

```bash
uv run --locked lle ssfm solve
```

Для PINN сначала укажите бэкенд PyTorch в PowerShell.

```powershell
$env:DDE_BACKEND = 'pytorch'
```

Для обучения на CPU используйте профиль `pinn`.

```powershell
uv run --locked --extra pinn lle pinn train
```

Для обучения на NVIDIA используйте профиль `pinn-cuda`.

```powershell
uv run --locked --extra pinn-cuda lle pinn train
```

Если нужный профиль уже установлен, `--no-sync` запускает команду в текущем
окружении проекта без синхронизации.

```powershell
uv run --no-sync lle pinn train
```

Запуск с заданным YAML, продолжение и перенос весов из активного окружения:

```bash
lle pinn train --config experiments/06_pinn-rad-sampling/rad_config.yaml
lle pinn resume RUN_ID
lle pinn extend RUN_ID --steps 80000
lle pinn transfer RUN_ID domain.time_end=4.5 training.iterations=20000
```

В conda-окружении GPU-сервера доступны те же команды. Для конкретной серии
расчетов используйте ее скрипт, например
`python scripts/_run_pinn_sweep.py --dry-run`.
