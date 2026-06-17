// Model 2: long-run team strength + season-specific deviations — points
// (Student-t likelihood). Skills are on the points scale. Covariates z-scored
// in Python.

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  int<lower=1> K;
  real<lower=1> nu;  // fixed df (default 5 in Python)
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=K> obs_pos;
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
  real<lower=log(0.5), upper=log(20)> log_sigma_pts;
  real<lower=log(1), upper=log(25)> log_tau_team;
  real<lower=log(0.25), upper=log(15)> log_tau_season;
  sum_to_zero_vector[T] team_skill_z;
  array[S] sum_to_zero_vector[K] season_dev_z;
}

transformed parameters {
  real sigma_pts = exp(log_sigma_pts);
  real tau_team = exp(log_tau_team);
  real tau_season = exp(log_tau_season);
  vector[T] team_skill = tau_team * team_skill_z;
  vector[N] skill_obs;

  for (n in 1:N) {
    skill_obs[n] = team_skill[team[n]]
                   + tau_season * season_dev_z[season[n], obs_pos[n]];
  }
}

model {
  intercept ~ normal(52, 10);
  beta_sot ~ normal(0, 8);
  beta_lag ~ normal(0, 0.5);
  beta_form ~ normal(0, 8);
  // Expert priors (points scale, Student-t nu=5 fixed in data):
  // sigma_pts: unpredictable season noise after skill + covariates (~4-6 pts SD equiv.).
  // tau_team: long-run residual team spread beyond process metrics (~6-10 pts SD).
  // tau_season: season-to-season deviations around long-run strength (~1.5-3 pts SD).
  log_sigma_pts ~ normal(log(4.5), 0.12);
  log_tau_team ~ normal(log(7), 0.20);
  log_tau_season ~ normal(log(2.25), 0.18);
  team_skill_z ~ std_normal();
  for (s in 1:S) {
    season_dev_z[s] ~ std_normal();
  }

  for (n in 1:N) {
    real mu = intercept
              + skill_obs[n]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n];
    pts[n] ~ student_t(nu, mu, sigma_pts);
  }
}

generated quantities {
  matrix[S, T] skill;
  vector[N] log_lik;

  skill = rep_matrix(0, S, T);

  for (n in 1:N) {
    real mu = intercept
              + skill_obs[n]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n];
    skill[season[n], team[n]] = skill_obs[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
