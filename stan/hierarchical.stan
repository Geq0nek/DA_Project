// Hierarchical Poisson: season-specific att/def (non-centered)
// att[s,*] and def[s,*] are centered per season for identifiability.
// sigma_att / sigma_def = spread of team strengths within a season.

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=1, upper=T> home;
  array[N] int<lower=1, upper=T> away;
  array[N] int<lower=0> goals_h;
  array[N] int<lower=0> goals_a;
}

parameters {
  real home_adv;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
  matrix[S, T] att_z;
  matrix[S, T] def_z;
}

transformed parameters {
  matrix[S, T] att;
  matrix[S, T] def;
  for (s in 1:S) {
    row_vector[T] a = sigma_att * att_z[s];
    row_vector[T] d = sigma_def * def_z[s];
    att[s] = a - mean(a);
    def[s] = d - mean(d);
  }
}

model {
  home_adv ~ normal(0.3, 0.3);
  sigma_att ~ exponential(1);
  sigma_def ~ exponential(1);
  to_vector(att_z) ~ std_normal();
  to_vector(def_z) ~ std_normal();

  for (n in 1:N) {
    int s = season[n];
    real log_lambda_h = home_adv + att[s, home[n]] - def[s, away[n]];
    real log_lambda_a = att[s, away[n]] - def[s, home[n]];
    goals_h[n] ~ poisson(exp(log_lambda_h));
    goals_a[n] ~ poisson(exp(log_lambda_a));
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    int s = season[n];
    real log_lambda_h = home_adv + att[s, home[n]] - def[s, away[n]];
    real log_lambda_a = att[s, away[n]] - def[s, home[n]];
    log_lik[n] = poisson_lpmf(goals_h[n] | exp(log_lambda_h))
               + poisson_lpmf(goals_a[n] | exp(log_lambda_a));
  }
}
