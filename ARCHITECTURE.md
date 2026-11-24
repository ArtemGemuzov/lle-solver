# Архитектура исследований

Цель — воспроизводимые численные расчёты, обучение нейросетей и анализ результатов.

## Понятия

- **Run** — отдельный вычислительный запуск с уникальным ID, параметрами,
  метаданными и результатами.
- **Experiment** — исследовательская директория. Не является программной сущностью.
- **Config** — типизированная конфигурация. Модели могут
  переиспользоваться между скриптами и включать другие модели.

Run существует независимо от Experiment. Один Run может использоваться
в нескольких исследованиях. Ссылки на выбранные запуски задаются в анализе.

## Ответственность

- `lle.physics` — параметры LLE, начальные поля, поле и интенсивность, общая геометрия.
  Не зависит от вычислительных методов, Run, конфигов и PyTorch.
- `lle.storage` — запись/чтение полей в NPZ.
- `lle.numerics` — уравнения, сетки, интеграторы и схемы численного решения.
- `lle.pinn` — построение модели, потери, обучение, предсказание и состояние обучения.
  Не зависит от Run и модулей сценариев.
- `lle.workflows` — сборка вычислений из конфигурации и сопровождение запусков.
- `lle.config` — базовый Config, загрузка YAML и переопределений.
- `lle.runs` — создание, поиск, статусы и блокировка запусков, сведения Git, хранение чекпоинтов и RNG.
- `lle.analysis`, `lle.viz` — анализ и визуализация общих типов поля.
- `lle.cli` разбирает команды `lle` и передает их публичным операциям workflow.
- `scripts/` содержит конкретные сценарии и серии расчетов.
- `experiments/` — исследовательские постановки и анализ результатов.

## Граф зависимостей

В таблице указаны разрешенные прямые импорты между частями пакета `lle`.
Импорты внутри каждой части разрешены.

| Кто импортирует | Разрешенные зависимости внутри `lle` |
| --- | --- |
| `physics`, `config`, `random_state` | Нет |
| `numerics`, `storage`, `analysis`, `viz` | `physics` |
| `pinn` | `physics`, `random_state` |
| `runs` | `config`, `random_state` |
| `cli` | `config`, `runs`, `workflows` |
| `workflows` | `config`, `physics`, `numerics`, `pinn`, `runs`, `storage` |

Сценарии SSFM и PINN независимы друг от друга и используют общие параметры
из `workflows.parameters`. `physics`, `numerics` и сценарий SSFM не зависят
от `torch` и `deepxde` даже через промежуточные модули. `runs.checkpoint`
использует `torch` для сериализации и сохранения состояния генераторов.

Import Linter проверяет эти правила по контрактам в `pyproject.toml`.
Отдельный контракт проверяет независимость операций обучения в памяти
(`lle.pinn`) от `runs`, `storage`, `config` и `workflows`.
Скрипты и блокноты используют библиотеку извне, в том числе для сборки
расчетов из `numerics` и анализа сохраненных результатов.

## Конфигурация

Порядок применения значений: defaults модели → YAML → переопределения
командной строки вида `initial.amplitude=2.0` или `solver.time_step=0.001`.
Файл задаётся через `--config`; без него используются defaults модели.

Общие части конфигураций находятся в `lle.workflows.parameters`.
Полный конфиг и проверки связей между его частями принадлежат сценарию:
`SsfmGaussianConfig` в `lle.workflows.ssfm_gaussian.config`,
`PinnGaussianConfig` в `lle.workflows.pinn_gaussian.config`.
В обоих пакетах операции сценария находятся в `workflow.py`.

## API сценариев

```python
from lle.config import load_config
from lle.workflows import pinn_gaussian
from lle.workflows.ssfm_gaussian.config import SsfmGaussianConfig
from lle.workflows.ssfm_gaussian.workflow import run as run_ssfm
from lle.storage import load_evolution

config = load_config(SsfmGaussianConfig, overrides=[
    "solver.time_end=1.0", "recording.interval=0.1",
])
ssfm_run = run_ssfm(config)
field = load_evolution(ssfm_run.path)

pinn_config = load_config(pinn_gaussian.PinnGaussianConfig, overrides=[
    "domain.time_end=1.0", "recording.interval=0.1",
])
run = pinn_gaussian.train_run(pinn_config)
pinn_field = load_evolution(run.path)
```

`run_ssfm(config)` создает Run, сохраняет эволюцию поля и возвращает
завершенный запуск. CLI `lle ssfm solve` вызывает этот же сценарий.
Для расчета в памяти `simulate(config)` возвращает `FieldEvolution`.

`lle.numerics.ssfm.solve(scheme, grid, state, num_steps=...,
step_size=..., record_every=...)` принимает готовую схему расщепления, сетку
и начальный вектор поля. Результат содержит начальный кадр, кадры через
`record_every` шагов и конечный кадр. Для продолжения можно передать
`result[-1].values` на той же сетке. Времена нового участка начинаются с нуля.
Стандартный `solve_strang_rk4` собирает схему Стрэнга с RK4 и вызывает `solve`.

Операции PINN возвращают Run и сохраняют поле, модель, историю потерь и чекпоинт:

- `train_run(config)` создает новый Run для обучения с нуля. Имя `train(config)` также поддерживается.
- `resume(run_or_id)` — продолжение незавершённого Run в том же каталоге.
- `extend(run_or_id, steps)` — новый дочерний Run, дополнительные шаги
  от последнего чекпоинта с сохранением Adam, точек, RNG и счётчиков.
  Для causal schedule расширение запрещено, как и ранее.
- `transfer(run_or_id, config=None)` — новый Run с весами родителя,
  новым Adam и счётчиком с нуля. Без конфига используются параметры родителя.
  При изменении временного горизонта пересчитывается первый слой, чтобы
  начальное предсказание комплексного поля сохранялось. Архитектура сети
  и пространственная длина должны совпадать.

Для анализа завершенного Run функция
`lle.workflows.pinn_gaussian.workflow.load_network(run_or_id)` восстанавливает
периодические признаки и загружает веса из `model.pt`. Она возвращает сеть
PyTorch на CPU в режиме `eval()` для `predict_evolution` и `evaluate_residual`.

Оба гауссовых конфига предоставляют `to_problem()`. Результат
`lle.physics.problem.LLEProblem` объединяет параметры уравнения, функцию
начального комплексного поля от вектора theta, длину кольца и конец интервала.
Для сравнения методов можно использовать постановку из примера выше.

```python
from lle.numerics.ssfm import solve_strang_rk4
from lle.pinn.features import PeriodicFeatures
from lle.pinn.model import build_periodic_network
from lle.pinn.solver import train, resume, extend, transfer

problem = config.to_problem()
reference = solve_strang_rk4(problem, num_points=128, time_step=0.001, recording_interval=0.1)
equation, domain, initial = problem.equation, problem.domain, problem.initial
features = PeriodicFeatures(domain.length, domain.time_end, harmonics=4)
net = build_periodic_network(features, hidden_layers=4, hidden_width=64)
snapshots = []
solution = train(net, equation, domain, initial, steps=2000,
                 on_checkpoint=snapshots.append, checkpoint_every=1000)
prediction = solution.predict(theta=reference.theta, times=reference.times)

continued = resume(net, equation, domain, initial, snapshots[0], steps=2000)
longer = extend(net, equation, domain, initial, continued.state, steps=100)

next_net = build_periodic_network(features, hidden_layers=4, hidden_width=64)
child = transfer(next_net, equation, domain, initial, longer.state, steps=100)
```

Все четыре операции работают с переданной сетью PyTorch. Она принимает
координаты (theta, t) и возвращает действительную и мнимую компоненты поля.
Периодичность в примере обеспечивает преобразование входных координат.
Обучение меняет веса этой сети на месте. Для независимого продолжения можно
передать новый экземпляр той же архитектуры. Аргумент seed управляет выборкой
и обучением; начальные веса определены при создании сети.

В resume число steps задает общий бюджет, в extend число дополнительных шагов.
Снимок содержит веса, Adam, точки, историю, счетчики и состояние генераторов.
При переносе между разными нормировками времени периодической FNN передают
исходные признаки через source_features. Workflow создает сеть и подставляет
аргументы из конфигурации запуска.

`resume` возвращает результат сразу, если снимок уже завершен.
`extend` сохраняет состояние Adam и допускает фиксированный бюджет без расписания
causal weighting. `transfer` начинает новый отсчет шагов с новым Adam и выборкой.

`predict_evolution` возвращает комплексное поле на декартовом произведении
заданных осей, сохраняя порядок координат и времен. Workflow строит регулярные
оси по настройкам записи. Фурье-операторы SSFM и производные autograd в PINN
используют параметры одной постановки.

`Field`, `FieldEvolution`, `Intensity` и `IntensityEvolution` находятся в
`lle.physics.fields`; SSFM и PINN строят периодические координаты через
`physics.geometry.periodic_coordinates`.
Расчётная сетка с фурье-модами остаётся в `numerics.grid`.

`build_network(input_size=..., hidden_layers=..., hidden_width=...)` создаёт FNN
без преобразования координат. Сценарий явно создаёт `PeriodicFeatures` и
подключает его через `net.apply_feature_transform(features)`. Можно использовать
исходные координаты или другое преобразование. `PeriodicFeatures.transfer_weights`
знает расположение временного признака и устройство первого слоя FNN; операция
не применяется к произвольной сети и проверяет совместимость пространственных признаков.

`predict_field(net, points)` и `evaluate_residual(net, points, residual)`
не требуют обучающей выборки или оптимизатора. `pinn.model.build_model` собирает модель, а
`pinn.training.train_model` настраивает Adam, обучает переданную модель
и возвращает историю потерь. Настройки метода определены в `pinn.parameters`. Для адаптивной выборки передается
`rad`, для смены стадий causal weighting передается `schedule`.

Опциональный `on_checkpoint(state)` получает независимый снимок каждые
`checkpoint_every` шагов, при смене стадии и после завершения обучения.
`train_run` вызывает `lle.pinn.solver.train`, передавая функции сохранения
снимков и полей стадий. При продолжении `lle.pinn.solver.resume` восстанавливает
состояние между созданием Adam и началом оптимизации.

`PinnTraining.state_dict()` возвращает независимый снимок состояния сети,
Adam, точек, истории, адаптивных потерь, стадий и счётчиков.
`load_state_dict()` восстанавливает это состояние, включая особенности DeepXDE.
Код PINN не знает путей файлов и полного конфига запуска.

`pinn.sampling.RADSampler` обновляет внутренние PDE-точки по величине двух
компонент невязки. Сценарий включает его через `sampling.rad`; без этого блока
сохраняется равномерное пересэмплирование. Первая адаптация происходит на
`warmup_steps` (при нуле перед первым шагом), следующие через
`training.resample_every`. До прогрева действует обычное пересэмплирование.
RAD использует отдельный NumPy RNG, состояние которого вместе с историей
адаптаций восстанавливается при `resume` и `extend`. `transfer` начинает
семплирование заново. Старые чекпоинты без состояния RAD остаются читаемыми.

Кандидаты каждый раз генерируются равномерно, невязка вычисляется порциями
`batch_size`. Вероятности: `p_i = a/M + (1-a)*r_i**k/sum(r**k)`, где
`a = uniform_fraction`, `k = power`, `M = candidate_points`.
Из кандидатов выбираются `domain.interior_points` точек. По умолчанию отбор
идет с возвращением. При `sampling.rad.replace=false` каждый индекс кандидата
может попасть в набор только один раз за обновление.
При нулевой невязке используется равномерное распределение. Поправки весов
loss на вероятности нет. IC-выборка и тестовая сетка сохраняются; дополнительные
PDE-точки на начальном срезе обновляются равномерно. Первая версия RAD
поддерживает фиксированные веса loss.

`sampling.csv` содержит шаг, среднее, максимум и 90-й перцентиль невязки
кандидатов, число уникальных выбранных кандидатов и время обновления в секундах.
`sampling_histograms.npz` содержит `steps`, `counts` с осями
`(update, theta_bin, time_bin)`, `theta_edges` и `time_edges` для сетки 16x16.
Эти результаты записываются при чекпоинтах и завершении запуска.

`runs.checkpoint` упаковывает состояние, конфиг, версии библиотек и глобальные
генераторы Python, NumPy и Torch, атомарно записывает файл и читает его обратно.
Сценарий передает `save_training_checkpoint` из `outputs.py` в обучение, проверяет
конфиги при `resume`. Ядро восстанавливает RNG после создания модели и загрузки состояния.
При `extend` сценарий формирует новый конфиг, сохраняя состояние оптимизатора.
Поле `script` остаётся метаданными; инфраструктура не проверяет его при возобновлении.

Импорт физики и конфигов не загружает DeepXDE/PyTorch. Для обучения используется
окружение с `DDE_BACKEND=pytorch`. Публичные операции сценариев сохраняют прежние
сигнатуры; низкоуровневые импорты обновлены на новую структуру.

Форматы существующих NPZ, YAML и checkpoint.pt сохранены. Для совместимости
выборок Run сценарии сохраняют прежние идентификаторы `scripts/gaussian.py`
и `scripts/pinn_gaussian.py`. Python-импорты перенесены в `lle`; старые пакеты
не предоставляют совместимых псевдонимов.

## Хранение запусков

```text
runs/<run_id>/
    parameters.yaml
    metadata.json
    ...
```

ID формируется автоматически из времени UTC с микросекундами:
`2026-11-02_08-30-15_123456`. Каждый запуск получает новый каталог.

Общий код определяет, **где** находится запуск. Скрипт и блокнот получают
каталог через общий интерфейс. Корень проекта определяется поиском
`pyproject.toml` вверх по дереву каталогов.

Код, постановки и исходные конфигурации хранятся в Git; `runs/` исключён
из Git.


## Эксперименты

Каждый каталог `experiments/NN_description/` представляет один эксперимент.
Он содержит один блокнот с предметным названием без номера, собственные CSV
выбранных запусков и каталог `results/` для графиков и таблиц.
Порядок чтения приведен в `experiments/index.md`.

Скрипты из `scripts/` или API сценариев из блокнота создают запуски.
Блокнот также загружает исходные Run, формулирует вопрос, выполняет анализ
и содержит выводы.

Каталоги results содержат воспроизводимые производные данные и исключены из Git.
