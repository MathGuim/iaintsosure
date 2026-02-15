
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulationParameters:
    amplitudes: list[int]
    frequencies: list[int]
    frame_rate: int
    noise_level: float
    noise_stdev: float
    n_signals: int
    contamination: float
    a: int
    b: int
    n_components: int
    n_simulations: int

