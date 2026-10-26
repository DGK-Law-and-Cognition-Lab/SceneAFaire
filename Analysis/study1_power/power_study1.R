# Study 1: power and sensitivity by simulation.
# Described in the Study 1 preregistration (Section 3.4). Seed 20260219.
#
# No pilot data exist, so every variance component is an assumption. Each
# scenario is run under a smaller and a larger item variance, because item
# variability, not participant variability, limits precision for item-level
# predictors. The design is simulated exactly as registered: six genres, 48
# stratified items per genre (12 per cell) plus 8 foils, 90 participants each
# completing two genres under the twelve-cell schedule, so 30 per genre.
#
# Usage: Rscript power_study1.R [n_sims_rating] [n_sims_generation]
# Output: power_results.csv and power_summary.txt in this folder.

suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(parallel) })
args <- commandArgs(trailingOnly = TRUE)
N_SIM_R <- if (length(args) >= 1) as.integer(args[1]) else 200
N_SIM_G <- if (length(args) >= 2) as.integer(args[2]) else 200
SEED <- 20260219
CORES <- max(1, detectCores() - 2)
RNGkind("L'Ecuyer-CMRG"); set.seed(SEED)
# Separate worker processes (PSOCK): forked lme4 fits crash with the macOS math library.
cl <- makeCluster(CORES, type = "PSOCK")
invisible(clusterEvalQ(cl, suppressPackageStartupMessages({ library(lme4); library(lmerTest) })))
clusterSetRNGStream(cl, SEED)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()

genres <- c("western", "horror", "science_fiction", "crime", "comedy", "fantasy")
pairs <- list(c("western","horror"), c("crime","science_fiction"), c("comedy","fantasy"),
              c("western","crime"), c("horror","fantasy"), c("science_fiction","comedy"))
# 90 participants: 12 ordered cells, 7 or 8 each, so every genre gets 30.
cells <- rep(1:12, length.out = 90)
participants <- data.frame(pid = 1:90, cell = cells)
participants$g1 <- sapply(participants$cell, function(c) { p <- pairs[[(c - 1) %/% 2 + 1]]; if (c %% 2 == 1) p[1] else p[2] })
participants$g2 <- sapply(participants$cell, function(c) { p <- pairs[[(c - 1) %/% 2 + 1]]; if (c %% 2 == 1) p[2] else p[1] })

# ---------------------------------------------------------------- ratings
items <- do.call(rbind, lapply(genres, function(g) {
  data.frame(genre = g,
             itemclass = c(rep(c("HH","HL","LH","LL"), each = 12), rep("FOIL", 8)),
             stringsAsFactors = FALSE)
}))
items$item <- seq_len(nrow(items))
items$P <- ifelse(items$itemclass %in% c("HH","HL"), 0.5, ifelse(items$itemclass %in% c("LH","LL"), -0.5, NA))
items$D <- ifelse(items$itemclass %in% c("HH","LH"), 0.5, ifelse(items$itemclass %in% c("HL","LL"), -0.5, NA))
long <- do.call(rbind, lapply(seq_len(nrow(participants)), function(i) {
  p <- participants[i, ]
  rbind(cbind(pid = p$pid, items[items$genre == p$g1, ]), cbind(pid = p$pid, items[items$genre == p$g2, ]))
}))

sim_rating <- function(bP, bD, bPD, foil_gap, sd_item, sd_pid = 0.7, sd_slope = 0.3, sd_res = 1.3) {
  it <- rnorm(nrow(items), 0, sd_item)
  pint <- rnorm(90, 0, sd_pid); psP <- rnorm(90, 0, sd_slope); psD <- rnorm(90, 0, sd_slope)
  d <- long
  strat <- !is.na(d$P)
  mu <- 4 + ifelse(strat, bP * d$P + bD * d$D + bPD * d$P * d$D, -foil_gap)
  # foil_gap is the foil mean below the LL cell mean
  mu[!strat] <- 4 + (-0.5 * bP - 0.5 * bD + 0.25 * bPD) - foil_gap
  y <- mu + it[d$item] + pint[d$pid] + ifelse(strat, psP[d$pid] * d$P + psD[d$pid] * d$D, 0) + rnorm(nrow(d), 0, sd_res)
  d$rating <- pmin(7, pmax(1, round(y)))
  d
}

fit_h2 <- function(d) {
  s <- d[!is.na(d$P), ]
  # Uncorrelated by-participant slopes, the registered structure after its first
  # reduction step; the full reduction chain is not simulated (it changes little
  # for item-level predictors and multiplies run time).
  m <- suppressMessages(suppressWarnings(lmer(rating ~ P * D + genre + (1 + P + D || pid) + (1 | item), data = s, REML = TRUE,
                                              control = lmerControl(calc.derivs = FALSE))))
  cf <- summary(m)$coefficients
  L <- setNames(rep(0, length(fixef(m))), names(fixef(m))); L["P"] <- 1; L["D"] <- -1
  ct <- contest(m, L, joint = FALSE)
  c(pP = cf["P", "Pr(>|t|)"], estP = cf["P", "Estimate"], pD = cf["D", "Pr(>|t|)"],
    pPD = ct[["Pr(>|t|)"]], estPD = ct[["Estimate"]])
}

fit_h3 <- function(d) {
  d$itemclass <- relevel(factor(d$itemclass), ref = "FOIL")
  m <- suppressMessages(suppressWarnings(lmer(rating ~ itemclass + genre + (1 | pid) + (1 | item), data = d, REML = TRUE)))
  cf <- summary(m)$coefficients
  # foil minus each cell = -(cell coefficient); all four must be negative and significant
  cells <- paste0("itemclass", c("HH","HL","LH","LL"))
  all(cf[cells, "Estimate"] > 0 & cf[cells, "Pr(>|t|)"] < .05)
}

rating_scenarios <- rbind(
  expand.grid(sd_item = c(0.8, 1.2), bD = c(0.3, 0.5), delta = 0),
  expand.grid(sd_item = c(0.8, 1.2), bD = 0.5, delta = c(0.3, 0.5, 0.75)))
rating_scenarios$bP <- rating_scenarios$bD + rating_scenarios$delta
rating_scenarios$bPD <- 0
rating_scenarios$foil_gap <- 0.75   # foils 0.75 points below the LL cell, a conservative gap

run_rating <- function(k) {
  sc <- rating_scenarios[k, ]
  clusterExport(cl, c("sc"), envir = environment())
  res <- parLapply(cl, seq_len(N_SIM_R), function(i) {
    d <- sim_rating(sc$bP, sc$bD, sc$bPD, sc$foil_gap, sc$sd_item)
    c(fit_h2(d), h3 = fit_h3(d))
  })
  r <- do.call(rbind, res)
  data.frame(outcome = "rating", sd_item = sc$sd_item, bP = sc$bP, bD = sc$bD, delta = sc$delta,
             power_P = mean(r[, "pP"] < .05 & r[, "estP"] > 0),
             power_D = mean(r[, "pD"] < .05),
             power_contrast = mean(r[, "pPD"] < .05),
             mean_est_contrast = mean(r[, "estPD"]),
             power_H3_all_four = mean(r[, "h3"]),
             n_sims = N_SIM_R)
}

# ---------------------------------------------------------------- generation
pool <- read.csv(file.path(here, "..", "v4_rebuild", "10_human_study_stimuli", "eligible_pool_predictors.csv"))
pool <- pool[pool$in_generation_analysis_set %in% c(TRUE, "True"), ]
pool$genre_key <- tolower(gsub(" ", "_", pool$genre))
pool$prev <- pool$prevalence_pct - 0.5
pool$dist <- pool$distinctiveness_pct - 0.5
N_PER_GENRE <- 30
MEAN_PRODUCED <- 6   # eligible-pool elements produced per participant per genre (assumed)

sim_generation <- function(b_prev, b_dist, sd_item) {
  u <- rnorm(nrow(pool), 0, sd_item)
  eta0 <- b_prev * pool$prev + b_dist * pool$dist + u
  # calibrate each genre's intercept so a participant produces MEAN_PRODUCED pool elements on average
  alpha <- tapply(seq_len(nrow(pool)), pool$genre_key, function(ix) {
    f <- function(a) sum(plogis(a + eta0[ix])) - MEAN_PRODUCED
    uniroot(f, c(-30, 10))$root
  })
  p <- plogis(alpha[pool$genre_key] + eta0)
  k <- rbinom(nrow(pool), N_PER_GENRE, p)
  data.frame(k = k, n = N_PER_GENRE, prev = pool$prev, dist = pool$dist, genre = pool$genre_key, item = seq_len(nrow(pool)))
}

fit_h1 <- function(d) {
  m <- suppressMessages(suppressWarnings(glmer(cbind(k, n - k) ~ prev + dist + genre + (1 | item),
                                               data = d, family = binomial, nAGQ = 0)))
  cf <- summary(m)$coefficients
  c(p = cf["prev", "Pr(>|z|)"], est = cf["prev", "Estimate"])
}

gen_scenarios <- expand.grid(b_prev = c(0.5, 1.0, 2.0), b_dist = 1.0, sd_item = c(1.0, 2.0))
run_generation <- function(k) {
  sc <- gen_scenarios[k, ]
  clusterExport(cl, c("sc"), envir = environment())
  res <- parLapply(cl, seq_len(N_SIM_G), function(i) fit_h1(sim_generation(sc$b_prev, sc$b_dist, sc$sd_item)))
  r <- do.call(rbind, res)
  data.frame(outcome = "generation", sd_item = sc$sd_item, b_prev = sc$b_prev, b_dist = sc$b_dist,
             odds_ratio_full_range = round(exp(sc$b_prev), 2),
             power_H1 = mean(r[, "p"] < .05 & r[, "est"] > 0),
             mean_est = mean(r[, "est"]), n_sims = N_SIM_G)
}

clusterExport(cl, c("items", "long", "pool", "N_PER_GENRE", "MEAN_PRODUCED", "sim_rating", "fit_h2", "fit_h3", "sim_generation", "fit_h1"))
t0 <- Sys.time()
rating_res <- do.call(rbind, lapply(seq_len(nrow(rating_scenarios)), function(k) { cat("rating scenario", k, "\n"); run_rating(k) }))
gen_res <- do.call(rbind, lapply(seq_len(nrow(gen_scenarios)), function(k) { cat("generation scenario", k, "\n"); run_generation(k) }))
write.csv(rating_res, file.path(here, "power_results_rating.csv"), row.names = FALSE)
write.csv(gen_res, file.path(here, "power_results_generation.csv"), row.names = FALSE)

sink(file.path(here, "power_summary.txt"))
cat("Study 1 power simulations. Seed", SEED, "| rating sims per scenario", N_SIM_R, "| generation sims per scenario", N_SIM_G, "\n")
cat("Assumed SDs (rating scale points): participant intercept 0.7, participant slopes 0.3, residual 1.3; item SD as listed.\n")
cat("Ratings rounded to the 1-7 scale. Generation: 30 participants per genre, 6 pool elements produced per participant per genre.\n\n")
print(rating_res, row.names = FALSE); cat("\n"); print(gen_res, row.names = FALSE)
cat("\nElapsed:", format(Sys.time() - t0), "\n"); sessionInfo()
sink()
stopCluster(cl)
cat("done\n")
