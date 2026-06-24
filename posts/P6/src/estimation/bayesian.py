import cmdstanpy
import arviz_base as az


def fit(data, stan_file):

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    posterior = model.sample(
        data=data, show_console=False, seed=42,
        parallel_chains=4, show_progress=False
    )

    # Model summary and diagnostcs
    print(posterior.summary())
    print(posterior.diagnose())

    idata = az.from_cmdstanpy(
        posterior=posterior,
        posterior_predictive=["y_rep"],
        observed_data={"y": data["y"]}
    )

    return idata
