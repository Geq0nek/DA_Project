// Model 2: season-specific team strength — points (Student-t likelihood)
// Skill is on the **points scale** (tau_skill in pts) to avoid beta_pts × sigma_skill
// non-identifiability. Covariates z-scored in Python.

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  real<lower=1> nu;  // fixed df (default 5 in Python)
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=T> team;
  array[N] real pts;
  array[N] real sot_diff_pg;
  array[N] real pts_lag1;
  array[N] real ppg_last10;
}

parameters {
  real intercept;
  real beta_sot;
  real beta_lag;
  real beta_form;
  real log_sigma_pts;
  real log_tau_skill;
  matrix[S, T] skill_z;
}

transformed parameters {
  real sigma_pts = exp(log_sigma_pts);
  real tau_skill = exp(log_tau_skill);
  matrix[S, T] skill;
  for (s in 1:S) {
    row_vector[T] row = tau_skill * skill_z[s];
    skill[s] = row - mean(row);
  }
}

model {
  intercept ~ normal(52, 10);
  beta_sot ~ normal(0, 8);
  beta_lag ~ normal(0, 0.5);
  beta_form ~ normal(0, 8);
  log_sigma_pts ~ normal(log(10), 0.25);
  log_tau_skill ~ normal(log(12), 0.25);
  to_vector(skill_z) ~ std_normal();

  for (n in 1:N) {
    real mu = intercept
              + skill[season[n], team[n]]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n];
    pts[n] ~ student_t(nu, mu, sigma_pts);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real mu = intercept
              + skill[season[n], team[n]]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
