import altair as alt
import pandas as pd
import solara
import sys
import numpy as np

from mesa.visualization import (
    CommandConsole,
    Slider,
    SolaraViz,
    make_plot_component,
)

from agents import GrassPatch, Hare, Lynx
from model import LynxHare


alt.data_transformers.disable_max_rows()


def custom_emoji_grid(model):
    if not hasattr(model, "agents"):
        return solara.Markdown("Waiting for initialization...")
        
    # Track cells that contain an animal to prevent rendering background elements there
    occupied_cells = set()
    animal_data = []
    grass_data = []
    
    for agent in list(model.agents):
        if isinstance(agent, (Lynx, Hare, GrassPatch)):
            if hasattr(agent, "cell") and agent.cell is not None:
                x, y = agent.cell.position
                
                if isinstance(agent, Lynx):
                    animal_data.append({"x_coord": x, "y_coord": y, "emoji_char": "🐈"})
                    occupied_cells.add((x, y))  # Block this cell
                elif isinstance(agent, Hare):
                    animal_data.append({"x_coord": x, "y_coord": y, "emoji_char": "🐇"})
                    occupied_cells.add((x, y))  # Block this cell
                elif isinstance(agent, GrassPatch):
                    emoji = "🌿" if getattr(agent, "fully_grown", True) else "🟫"
                    grass_data.append({"x_coord": x, "y_coord": y, "emoji_char": emoji})

    # Only keep grass tiles if their coordinates are NOT in occupied_cells
    visible_grass = [tile for tile in grass_data if (tile["x_coord"], tile["y_coord"]) not in occupied_cells]
    
    # Combine the filtered lists
    combined_data = animal_data + visible_grass
                
    if not combined_data:
        df = pd.DataFrame(columns=["x_coord", "y_coord", "emoji_char"])
    else:
        df = pd.DataFrame(combined_data)
    
    chart = (
        alt.Chart(df)
        .configure_axis(grid=False)
        .mark_text(size=22, baseline="middle", align="center")
        .encode(
            x=alt.X("x_coord:Q", scale=alt.Scale(domain=[-0.5, model.grid.width - 0.5])).title(None),
            y=alt.Y("y_coord:Q", scale=alt.Scale(domain=[-0.5, model.grid.height - 0.5])).title(None),
            text=alt.Text("emoji_char:N")
        )
        .properties(width=600, height=600)
    )
    return solara.FigureAltair(chart)

rng_generator = np.random.default_rng(20260603)
rng_seeds = rng_generator.integers(0, sys.maxsize, size=50).tolist()

model_params = {
    "rng": {
        "type": "Select",
        "value": rng_seeds[0],
        "values": rng_seeds,
        "label": "Random Seed",
    },
    "grass_regrowth": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "grass regrowth enabled?",
    },
    "initial_hare": Slider("Initial Hare Population", 30, 1, 100),
    "initial_lynxes": Slider("Initial Lynx Population", 4, 1, 100),
    "hare_reproduce": Slider("Hare Reproduction Rate", 0.2, 0, 1, 0.05),
    "lynx_reproduce": Slider(
        "Lynx Reproduction Rate",
        0.001,
        0,
        1,
        0.01,
    ),
    "hare_gain_from_food": Slider("Hare Gain From Food", 1, 1, 100),
    "lynx_gain_from_food": Slider("Lynx Gain From Food Rate", 17, 1, 100),
    "grass_regrowth_time": Slider("Grass Regrowth Time", 13, 1, 100),
    "width": 25,
    "height": 25
}

def post_process_altair(chart):
    return chart.properties(width=550, height=350)

plot_component = make_plot_component(
    {"Lynxes": "gold", "Hare": "cyan"},
    backend="altair", 
    post_process=post_process_altair
)

parameters = {"rng": rng_seeds[0], 'hare_reproduce': 0.2, 'lynx_reproduce': 0.001, 'hare_gain_from_food': 1, 'lynx_gain_from_food': 17, 'grass_regrowth_time': 13, 'width': 25, 'height': 25, 'initial_hare': 30, 'initial_lynxes': 4}

parameters = parameters | {"grass_regrowth": True}

model = LynxHare(**parameters)

page = SolaraViz(
    model,
    components=[plot_component, CommandConsole, custom_emoji_grid],
    model_params=model_params,
    name="Lynx Hare",
)

page
