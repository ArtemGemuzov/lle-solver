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

```bash
uv sync --locked --extra notebooks
```
```bash
uv run scripts/<script-name>.py
```