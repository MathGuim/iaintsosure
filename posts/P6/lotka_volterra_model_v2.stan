functions {
  vector dz_dt(real t, vector z, array[] real theta, array[] real x_r, array[] int x_i) {
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
  int<lower=0> N;
  array[N] real ts;
  vector[2] y_init;
  array[N] vector<lower=0>[2] y;
  
  // New data for projection
  int<lower=0> N_future;
  array[N_future] real ts_future;
}

parameters {
  array[4] real<lower=0> theta;
  vector<lower=0>[2] z_init;
  vector<lower=0>[2] sigma;
}

transformed parameters {
  array[N] vector[2] z = ode_rk45_tol(dz_dt, z_init, 0.0, ts, 1e-6, 1e-6, 1000000, theta, rep_array(0.0, 0), rep_array(0, 0));
}

model {

  theta[1] ~ normal(1, 0.5);
  theta[3] ~ normal(1, 0.5);
  theta[2] ~ normal(0.05, 0.05);
  theta[4] ~ normal(0.05, 0.05);
  sigma ~ lognormal(-1, 1);
  z_init ~ lognormal(log(10), 1);
  
  y_init ~ lognormal(log(z_init), sigma);
  for (n in 1:N) y[n] ~ lognormal(log(z[n]), sigma);
}

generated quantities {
  array[N + N_future] vector[2] z_all;
  array[N + N_future] vector[2] y_rep;
  
  // Combine time points
  array[N + N_future] real ts_all;
  for(i in 1:N) ts_all[i] = ts[i];
  for(i in 1:N_future) ts_all[N + i] = ts_future[i];
  
  // Solve ODE for the extended timeline
  z_all = ode_rk45_tol(dz_dt, z_init, 0.0, ts_all, 1e-6, 1e-6, 1000000, theta, rep_array(0.0, 0), rep_array(0, 0));
  
  // Generate predictive distribution for the entire period
  for (n in 1:(N + N_future)) {
    for (k in 1:2) {
      y_rep[n][k] = lognormal_rng(log(z_all[n][k]), sigma[k]);
    }
  }
}