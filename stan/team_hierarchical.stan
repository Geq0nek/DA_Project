// Model 2 (team_hierarchical): long-run team strength + season-level effects

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
  sum_to_zero_vector[S] season_effect_z;
}

transformed parameters {
  real sigma_pts = exp(log_sigma_pts);
  real tau_team = exp(log_tau_team);
  real tau_season = exp(log_tau_season);
  vector[T] team_skill = tau_team * team_skill_z;
  vector[S] season_effect = tau_season * season_effect_z;
}

model {
  intercept ~ normal(52, 15);
  beta_sot ~ normal(0, 10);
  beta_lag ~ normal(0, 1);
  beta_form ~ normal(0, 10);
  beta_promoted ~ normal(-8, 8);
  log_sigma_pts ~ normal(log(8), 0.5);
  log_tau_team ~ normal(log(8), 0.5);
  log_tau_season ~ normal(log(3), 0.5);
  team_skill_z ~ std_normal();
  season_effect_z ~ std_normal();

  for (n in 1:N) {
    real mu = intercept
              + team_skill[team[n]]
              + season_effect[season[n]]
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
      skill[s, t] = team_skill[t] + season_effect[s];
    }
  }

  for (n in 1:N) {
    real mu = intercept
              + team_skill[team[n]]
              + season_effect[season[n]]
              + beta_sot * sot_diff_pg[n]
              + beta_lag * pts_lag1[n]
              + beta_form * ppg_last10[n]
              + beta_promoted * is_promoted[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
