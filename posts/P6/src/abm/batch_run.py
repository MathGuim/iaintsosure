import pandas as pd
import numpy as np
import sys
import plotnine as p9
import matplotlib

from mesa import batch_run
from model import LynxHare


matplotlib.use("Agg")

N_TICKS         = 52
MAX_STEPS       = 20 * N_TICKS
NUM_SIMULATIONS = 10
 
rng_generator = np.random.default_rng(20260603)
rng_seeds = rng_generator.integers(0, sys.maxsize, size=NUM_SIMULATIONS).tolist()


parameters = {'hare_reproduce': 0.2, 'lynx_reproduce': 0.001, 'hare_gain_from_food': 1, 'lynx_gain_from_food': 17, 'grass_regrowth_time': 13, 'width': 25, 'height': 25, 'initial_hare': 30, 'initial_lynxes': 4} #using GenerationNode MBM.

parameters = parameters | {"grass_regrowth": True}


if __name__ == "__main__":

    results = batch_run(
        model_cls=LynxHare,          
        parameters=parameters,
        rng=rng_seeds,           
        max_steps=MAX_STEPS,
        number_processes=6,
        data_collection_period=N_TICKS,
    )

    if results:

        df = pd.DataFrame(results).assign(Step=(pd.col("Step") / N_TICKS) + 1900)

        df.to_parquet("data/output/large_simulation.parquet", index=False)

        df = df.melt(id_vars=["RunId", "Step"], value_vars=["Lynxes", "Hare"])

        p = (
            p9.ggplot(df, p9.aes("Step", "value", color="variable"))
            + p9.geom_line()
            + p9.geom_point()
            + p9.scale_x_continuous(breaks=range(1900, 1930, 10))
            + p9.facet_wrap("RunId", ncol=2, scales="free_x")
        )

        p.save("img/04_simulation.png", width=20, height=15, verbose=False)

