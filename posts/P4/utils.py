import random
import numpy as np
import pandas as pd
import numpy.typing as npt

from params import SimulationParameters


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def synthesize_complex_wave(amplitudes: list[int], frequencies: list[int], times: npt.NDArray) -> npt.NDArray:
    args = np.outer(times, frequencies)
    M = np.cos(np.pi * 2 * args)
    ys = np.dot(M, amplitudes)
    return ys


def create_multiple_gaussian_noise_signals(
    base_signal: npt.NDArray,
    noise_level: float,
    noise_stdev: float,
    n_signals: int,
    frame_rate: int
) -> npt.NDArray:
    noise = np.random.normal(noise_level, noise_stdev, size=(n_signals, frame_rate))
    return base_signal + noise


def create_random_interval(len_signal: int, a: int, b: int) -> tuple[int, int]:
    begin = random.randint(0, int(len_signal / 2))
    multiplier = random.betavariate(a, b)
    end = int(begin + (1 + multiplier) * begin)
    return begin, end


def insert_outliers_into_signals(
    gaussian_noise_signals: npt.NDArray,
    noise_level: float,
    noise_stdev: float,
    contamination: float,
    a: int,
    b: int
) -> tuple[npt.NDArray, list[int]]:

    ground_truth = []
    n_signals, len_signals = gaussian_noise_signals.shape

    for i in range(n_signals):

        is_outlier = random.random() <= contamination
        ground_truth.append(-1 if is_outlier else 1)

        if is_outlier:
            begin, end = create_random_interval(len_signal=len_signals, a=a, b=b)
            constant = np.random.normal(noise_level, noise_stdev)
            gaussian_noise_signals[i, begin:end] = constant
    
    return gaussian_noise_signals, ground_truth


def simulate_signals(params: SimulationParameters) -> tuple[npt.NDArray, npt.NDArray, list[int]]:

    t = np.linspace(0, 1, params.frame_rate, endpoint=False)

    base_signal = synthesize_complex_wave(params.amplitudes, params.frequencies, t).reshape(1, -1)

    gaussian_noise_signals = create_multiple_gaussian_noise_signals(
        base_signal=base_signal,
        noise_level=params.noise_level,
        noise_stdev=params.noise_stdev,
        n_signals=params.n_signals,
        frame_rate=params.frame_rate
    )

    noisy_signals, ground_truth = insert_outliers_into_signals(
        gaussian_noise_signals=gaussian_noise_signals,
        noise_level=params.noise_level,
        noise_stdev=params.noise_stdev,
        contamination=params.contamination,
        a=params.a,
        b=params.b
    )

    return base_signal, noisy_signals, ground_truth