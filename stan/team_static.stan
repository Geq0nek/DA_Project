// Model 1 (team_static): static team strength — predict points for one team-season
// Covariates (z-scored in Python): sot_diff_pg, pts_lag1, ppg_last10

data {
  int<lower=1> N;
  int<lower=1> T;
  real<lower=1> nu;  // fixed df (default 5 in Python)
  array[N] int<lower=1, upper=T> team;
  array[N] real pts;
  array[N] real sot_diff_pg;
  array[N] real pts_lag1;
  array[N] real ppg_last10;
  array[N] int<lower=0, upper=1> is_promoted;
}

parameters {
  real intercept;
  real<lower=0> beta_pts;
  real beta_sot;
  real beta_lag;
  real beta_form;
  real beta_promoted;
  real<lower=log(0.5), upper=log(30)> log_sigma_pts;
  sum_to_zero_vector[T] skill;
}

transformed parameters {
  real sigma_pts = exp(log_sigma_pts);
}

model {
  intercept ~ normal(52, 15);
  beta_pts ~ normal(10, 10);
  beta_sot ~ normal(0, 10);
  beta_lag ~ normal(0, 1);
  beta_form ~ normal(0, 10);
  beta_promoted ~ normal(-8, 8);
  log_sigma_pts ~ normal(log(15), 0.6);
  skill ~ std_normal();

  for (n in 1:N) {
    real mu = intercept
              + beta_pts * skill[team[n]]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n]
              + beta_promoted * is_promoted[n];
    pts[n] ~ student_t(nu, mu, sigma_pts);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real mu = intercept
              + beta_pts * skill[team[n]]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n]
              + beta_promoted * is_promoted[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
