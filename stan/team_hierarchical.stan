// Model 2 (team_hierarchical): long-run team strength + season-level effects

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  real<lower=1> nu;  // fixed df (default 2 in Python)
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=T> team;
  array[N] real pts;
  array[N] int<lower=0, upper=1> is_promoted;
}

parameters {
  real intercept;
  real beta_promoted;
  real<lower=log(1), upper=log(35)> log_sigma_pts;
  real<lower=log(0.5), upper=log(45)> log_tau_team;
  real<lower=log(0.25), upper=log(25)> log_tau_season;
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
  intercept ~ normal(50, 25);
  beta_promoted ~ normal(-5, 15);
  log_sigma_pts ~ normal(log(15), 0.8);
  log_tau_team ~ normal(log(12), 0.8);
  log_tau_season ~ normal(log(5), 0.8);
  team_skill_z ~ std_normal();
  season_effect_z ~ std_normal();

  for (n in 1:N) {
    real mu = intercept
              + team_skill[team[n]]
              + season_effect[season[n]]
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
              + beta_promoted * is_promoted[n];
    log_lik[n] = student_t_lpdf(pts[n] | nu, mu, sigma_pts);
  }
}
