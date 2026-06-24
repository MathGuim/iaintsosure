import math
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid

from agents import GrassPatch, Hare, Lynx


class LynxHare(Model):
    """Lynx-Hare predator-prey model with parameters assigned directly to the model."""

    def __init__(
        self,
        width: int,
        height: int,
        initial_hare: int,
        initial_lynxes: int,
        hare_reproduce: float,
        lynx_reproduce: float,
        lynx_gain_from_food: float,
        grass_regrowth: bool,
        grass_regrowth_time: int,
        hare_gain_from_food: float,
        rng=None
    ):
        # Handle seeds properly from both standalone creation and batch_run
        super().__init__(rng=rng)

        self.width = width
        self.height = height
        self.grass_regrowth = grass_regrowth

        # Create grid using experimental cell space
        self.grid = OrthogonalMooreGrid(
            [self.height, self.width],
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

        # Set up data collection
        model_reporters = {
            "Lynxes": lambda m: len(m.agents_by_type[Lynx]),
            "Hare": lambda m: len(m.agents_by_type[Hare]),
        }
        if self.grass_regrowth:
            model_reporters["Grass"] = lambda m: len(
                m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
            )

        self.datacollector = DataCollector(model_reporters)

        # Create hare:
        Hare.create_agents(
            self,
            initial_hare,
            energy=self.rng.random((initial_hare,)) * 2 * hare_gain_from_food,
            p_reproduce=hare_reproduce,
            energy_from_food=hare_gain_from_food,
            cell=self.random.choices(
                self.grid.all_cells.cells, k=initial_hare
            ),
        )
        
        # Create Lynxes:
        Lynx.create_agents(
            self,
            initial_lynxes,
            energy=self.rng.random((initial_lynxes,)) * 2 * lynx_gain_from_food,
            p_reproduce=lynx_reproduce,
            energy_from_food=lynx_gain_from_food,
            cell=self.random.choices(
                self.grid.all_cells.cells, k=initial_lynxes
            ),
        )

        # Create grass patches if enabled
        if self.grass_regrowth:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0
                    if fully_grown
                    else self.random.randrange(0, grass_regrowth_time)
                )
                GrassPatch(self, countdown, grass_regrowth_time, cell)

        # Collect initial data
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Execute one step of the model."""
        # First activate all hare, then all lynxes, both in random order
        self.agents_by_type[Hare].shuffle_do("step")
        self.agents_by_type[Lynx].shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)
