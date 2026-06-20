// Model 2 (team_hierarchical): long-run team strength + season deviations — one team-season row

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
  array[N] int<lower=0, upper=1> is_promoted;
}

parameters {
  real intercept;
  real beta_sot;
  real beta_lag;
  real beta_form;
  real beta_promoted;
  real<lower=log(0.5), upper=log(20)> log_sigma_pts;
  real<lower=log(1), upper=log(25)> log_tau_team;
  real<lower=log(0.25), upper=log(15)> log_tau_season;
  sum_to_zero_vector[T] team_skill_z;
  array[S] sum_to_zero_vector[T] season_dev_z;
}

transformed parameters {
  real sigma_pts = exp(log_sigma_pts);
  real tau_team = exp(log_tau_team);
  real tau_season = exp(log_tau_season);
  vector[T] team_skill = tau_team * team_skill_z;
  vector[N] skill_obs;

  for (n in 1:N) {
    skill_obs[n] = team_skill[team[n]]
                   + tau_season * season_dev_z[season[n], team[n]];
  }
}

model {
  intercept ~ normal(52, 10);
  beta_sot ~ normal(0, 8);
  beta_lag ~ normal(0, 0.5);
  beta_form ~ normal(0, 8);
  beta_promoted ~ normal(-10, 5);
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
              + beta_form * ppg_last10[n]
              + beta_promoted * is_promoted[n];
    pts[n] ~ student_t(nu, mu, sigma_pts);
  }
}

generated quantities {
  matrix[S, T] skill;
  vector[N] log_lik;

  for (t in 1:T) {
    for (s in 1:S) {
      skill[s, t] = team_skill[t] + tau_season * season_dev_z[s, t];
    }
  }

  for (n in 1:N) {
    real mu = intercept
              + skill_obs[n]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n]
              + beta_promoted * is_promoted[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
