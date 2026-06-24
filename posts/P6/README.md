
## Dependencies

You will need uv installed, Then,

```bash
uv sync
```

Then, inside a python terminal install cmdstanpy:

```python
cmdstanpy.install_cmdstan(version="2.38.0", overwrite=True, cores=4)
```

## ABM

To run the app and visualizate the simulations:

```bash
uv run solara run src/abm/app.py
```

To run a specific simulation change parameters inside the file and run:

```bash
uv run src/abm/batch_run.py
```

## Estimation

The analysis are inside the lotka_volterra_model.ipynb notebook.
