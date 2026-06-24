import pandas as pd
import numpy as np
import sys

from mesa import batch_run
from model import LynxHare

from tslearn.metrics import dtw
from ax import Client, RangeParameterConfig, ChoiceParameterConfig


N_TICKS         = 52
MAX_STEPS       = 20 * N_TICKS
NUM_SIMULATIONS = 10


def objective_fn(parameters, real_data, rng_seeds, number_processes=1):
    
    results = batch_run(
        model_cls=LynxHare,          
        parameters=parameters,
        rng=rng_seeds,           
        max_steps=MAX_STEPS,
        number_processes=number_processes,
        data_collection_period=N_TICKS,
    )

    results = (
        pd.DataFrame(results)
        .assign(Step=(pd.col("Step") / N_TICKS) + 1900)
        .merge(real_data, on="Step", how="inner")
    )

    survival = (
        results.groupby("RunId")
        .agg({"Lynxes": "last", "Hare": "last"})
        .assign(survived=(pd.col("Lynxes") > 0) & (pd.col("Hare") > 0))
        ["survived"]
        .mean()
    )

    dtw_hare = (
        results.groupby("RunId")
        .apply(lambda d: dtw(d["Hare"], d["hare"]))
        .mean()
        .item()
    )

    dtw_lynx = (
        results.groupby("RunId")
        .apply(lambda d: dtw(d["Lynxes"], d["lynx"]))
        .mean()
        .item()
    )

    return dict(rmse_hare=dtw_hare, rmse_lynx=dtw_lynx, survival=survival)


def main():

    rng_generator = np.random.default_rng(20260603)
    rng_seeds = rng_generator.integers(0, sys.maxsize, size=NUM_SIMULATIONS).tolist()

    lynx_hare_df = pd.read_csv("data/input/Howard2009.csv", skiprows=2).rename(columns=lambda x: x.strip().lower())
    lynx_hare_df = lynx_hare_df.assign(Step=pd.col("year"))

    client = Client(random_seed=20260603)

    client.configure_experiment(
        parameters=[
            ChoiceParameterConfig(
                name="width",
                parameter_type="int",
                values=[25],
            ),
            ChoiceParameterConfig(
                name="height",
                parameter_type="int",
                values=[25],
            ),
            ChoiceParameterConfig(
                name="initial_hare",
                parameter_type="int",
                values=[lynx_hare_df.head(1).hare.item()],
            ),
            ChoiceParameterConfig(
                name="initial_lynxes",
                parameter_type="int",
                values=[lynx_hare_df.head(1).lynx.item()],
            ),
            RangeParameterConfig(
                name="hare_reproduce",
                parameter_type="float",
                bounds=(0.001, 0.2),
            ),
            RangeParameterConfig(
                name="lynx_reproduce",
                parameter_type="float",
                bounds=(0.001, 0.05),
            ),
            RangeParameterConfig(
                name="hare_gain_from_food",
                parameter_type="int",
                bounds=(1, 10),
            ),
            RangeParameterConfig(
                name="lynx_gain_from_food",
                parameter_type="int",
                bounds=(5, 20)
            ),
            RangeParameterConfig(
                name="grass_regrowth_time",
                parameter_type="int",
                bounds=(1, 30)
            ),
        ]
    )

    client.configure_optimization(
        objective="survival, -rmse_lynx, -rmse_hare",
        outcome_constraints=[
            "survival >= 0.95",
            "rmse_lynx <= 100",
            "rmse_hare <= 50",
        ]
    )

    for _ in range(100):

        for trial_index, parameters in client.get_next_trials(max_trials=1).items():

            parameters = parameters | {"grass_regrowth": True}

            result = objective_fn(parameters, lynx_hare_df, rng_seeds)

            print(
                f"Trial {trial_index}: ",
                f"survival={result.get('survival', None)}",
                f"distance_hare={result.get('rmse_hare', None)}",
                f"distance_lynx={result.get('rmse_lynx', None)}",
                "\n",
                sep="\n"
            )

            client.complete_trial(trial_index=trial_index, raw_data=result)
        
    return client.get_pareto_frontier()


if __name__ == "__main__":
    print(main())

