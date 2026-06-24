functions {
  vector dz_dt(real t, vector z, array[] real theta,
               array[] real x_r, array[] int x_i) {

    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta  = theta[2];
    real gamma = theta[3];
    real delta = theta[4];

    vector[2] dz;

    dz[1] = (alpha - beta * v) * u;
    dz[2] = (-gamma + delta * u) * v;

    return dz;
  }
}

data {
  int<lower=0> N;                 // number of measurement times
  array[N] real ts;               // measurement times
  vector[2] y_init;               // initial measured populations
  array[N] vector<lower=0>[2] y;  // measured populations
}

parameters {
  array[4] real<lower=0> theta;   // {alpha, beta, gamma, delta}
  vector<lower=0>[2] z_init;      // initial population
  vector<lower=0>[2] sigma;       // measurement errors
}

transformed parameters {
  array[N] vector[2] z;

  z = ode_rk45_tol(
    dz_dt,
    z_init,
    0.0,
    ts,
    1e-6,   // relative tolerance
    1e-6,   // absolute tolerance
    1000000,    // max_num_steps
    theta,
    rep_array(0.0, 0),
    rep_array(0, 0)
  );
}

model {
  theta[1] ~ normal(1, 0.5);
  theta[3] ~ normal(1, 0.5);

  theta[2] ~ normal(0.05, 0.05);
  theta[4] ~ normal(0.05, 0.05);

  sigma ~ lognormal(-1, 1);
  z_init ~ lognormal(log(10), 1);

  y_init ~ lognormal(log(z_init), sigma);

  for (n in 1:N) {
    y[n] ~ lognormal(log(z[n]), sigma);
  }
}

generated quantities {
  vector[2] y_init_rep;
  array[N] vector[2] y_rep;

  for (k in 1:2)
    y_init_rep[k] = lognormal_rng(log(z_init[k]), sigma[k]);

  for (n in 1:N)
    for (k in 1:2)
      y_rep[n][k] = lognormal_rng(log(z[n][k]), sigma[k]);
}
