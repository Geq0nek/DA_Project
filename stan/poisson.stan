// Poisson goals model: team attack/defense + home advantage
// One row per match: home team, away team, full-time goals

data {
  int<lower=1> N;              // number of matches
  int<lower=1> T;              // number of teams
  array[N] int<lower=1, upper=T> home;
  array[N] int<lower=1, upper=T> away;
  array[N] int<lower=0> goals_h;
  array[N] int<lower=0> goals_a;
}

parameters {
  real home_adv;
  vector[T] att_raw;
  vector[T] def_raw;
}

transformed parameters {
  vector[T] att = att_raw - mean(att_raw);
  vector[T] def = def_raw - mean(def_raw);
}

model {
  home_adv ~ normal(0.3, 0.3);
  att_raw ~ std_normal();
  def_raw ~ std_normal();

  for (n in 1:N) {
    real log_lambda_h = home_adv + att[home[n]] - def[away[n]];
    real log_lambda_a = att[away[n]] - def[home[n]];
    goals_h[n] ~ poisson(exp(log_lambda_h));
    goals_a[n] ~ poisson(exp(log_lambda_a));
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real log_lambda_h = home_adv + att[home[n]] - def[away[n]];
    real log_lambda_a = att[away[n]] - def[home[n]];
    log_lik[n] = poisson_lpmf(goals_h[n] | exp(log_lambda_h))
               + poisson_lpmf(goals_a[n] | exp(log_lambda_a));
  }
}
