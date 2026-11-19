# Study 1: the registered analysis (preregistration Sections 7, 8 and 9.1 to 9.2).
#
# Every confirmatory, secondary and robustness model of the preregistration is fit
# here in its registered form, with the registered random-effects reduction rule.
# Exploratory analyses (Section 9.4) are not in this script.
#
# Usage, from this folder:
#   Rscript study1_analysis.R simulate   # simulated data in the registered format; checks the code
#   Rscript study1_analysis.R real       # the real data (after notebook 11 has run)
#
# Real-mode inputs: the session files downloaded from the Pavlovia project's data/
# folder into Data/Study1_raw/, and the outputs of notebook 11 in
# Analysis/v4_rebuild/11_study1_coding/ (coding_decision.json, quality flags,
# production matrices, s1_genre_discrimination.csv). Outputs go to
# Analysis/v4_rebuild/12_study1_analysis/ (real) or ./simulated_run/ (simulate).

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(ordinal); library(jsonlite); library(arrow)
})
args <- commandArgs(trailingOnly = TRUE)
MODE <- if (length(args)) args[1] else "simulate"
stopifnot(MODE %in% c("simulate", "real"))
SEED <- 20260219
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
HERE <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()
AN <- normalizePath(file.path(HERE, ".."))
ROOT <- normalizePath(file.path(AN, ".."))
STIM <- file.path(AN, "v4_rebuild", "10_human_study_stimuli")
OUT <- if (MODE == "real") file.path(AN, "v4_rebuild", "12_study1_analysis") else file.path(HERE, "simulated_run")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
RAW <- if (MODE == "real") file.path(ROOT, "Data", "Study1_raw") else file.path(OUT, "raw")
CODED <- if (MODE == "real") file.path(AN, "v4_rebuild", "11_study1_coding") else file.path(OUT, "coded")

GENRE_KEY <- c(Western = "western", Horror = "horror", "Science Fiction" = "science_fiction",
               Crime = "crime", Comedy = "comedy", Fantasy = "fantasy")
CELLS <- c("HH", "HL", "LH", "LL")
CTRL_L <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
CTRL_G <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
SINGULAR_TOL <- 1e-4

# ------------------------------------------------------------------ registered materials
stim <- read.csv(file.path(STIM, "stimuli_long.csv"), stringsAsFactors = FALSE)
stim$is_foil <- as.logical(stim$is_foil)
pool <- read.csv(file.path(STIM, "eligible_pool_predictors.csv"), stringsAsFactors = FALSE)
pool <- pool[as.logical(pool$in_generation_analysis_set), ]
stim <- merge(stim, pool[, c("genre", "key", "ensemble_pct")], by = c("genre", "key"), all.x = TRUE)

# ------------------------------------------------------------------ simulated inputs
if (MODE == "simulate") source(file.path(HERE, "simulate_study1_inputs.R"))

# ------------------------------------------------------------------ sessions and exclusions (9.2)
read_sessions <- function(dir) {
  files <- list.files(dir, pattern = "SAF_STUDY1_.*\\.csv$", full.names = TRUE)   # Pavlovia names them data_SAF_STUDY1_...
  rows <- lapply(files, function(f) read.csv(f, colClasses = "character", check.names = FALSE))
  cols <- unique(unlist(lapply(rows, names)))
  do.call(rbind, lapply(rows, function(d) { for (c in setdiff(cols, names(d))) d[[c]] <- NA_character_; d[cols] }))
}
is_true <- function(x) tolower(trimws(as.character(x))) %in% "true"
blank <- function(x) is.na(x) | trimws(x) == ""

decision <- fromJSON(file.path(CODED, "coding_decision.json"))
PRIMARY <- decision$primary                          # "automated" or "human"
suffix <- if (PRIMARY == "human") "_human" else ""
flags <- read.csv(file.path(CODED, paste0("quality_flags", suffix, ".csv")), colClasses = c(participant_id = "character"))
flagged <- unique(flags$participant_id[is_true(flags$flag)])

sessions <- read_sessions(RAW)
test <- grepl("^(pilot|test)", sessions$participant_id, ignore.case = TRUE) | is_true(sessions$is_test)
sessions <- sessions[!test, ]
sessions <- sessions[order(sessions$start_time), ]
rules <- data.frame(
  r1_hidden_trap = !is_true(sessions$captcha_pass),
  r2_reading_speed = is_true(sessions$spr_fast_any),
  r3_task_check = is_true(sessions$task_check_failed_both),
  r4_attention = suppressWarnings(as.integer(sessions$attention_pass_total)) %in% NA |
                 suppressWarnings(as.integer(sessions$attention_pass_total)) < 4,
  r5_incomplete = blank(sessions$end_time) | blank(sessions$genre1_ratings) | blank(sessions$genre2_ratings),
  r6_generation_quality = sessions$participant_id %in% flagged,
  r7_ai_use = sessions$ai_usage %in% "Yes",
  r8_duplicate_id = duplicated(sessions$participant_id),
  r9_not_serious = sessions$seriousness %in% "1")
first_rule <- apply(rules, 1, function(r) if (any(r)) names(rules)[which(r)[1]] else NA_character_)
excl_report <- data.frame(rule = names(rules),
                          excluded_in_order = sapply(names(rules), function(r) sum(first_rule %in% r)),
                          meeting_rule_any_order = colSums(rules))
usable <- sessions[is.na(first_rule), ]
cat("sessions:", nrow(sessions), "| usable:", nrow(usable), "\n")
print(excl_report, row.names = FALSE)

# ------------------------------------------------------------------ analysis tables
ratings_table <- function(s) {
  out <- vector("list", 2 * nrow(s))
  k <- 0
  for (i in seq_len(nrow(s))) for (slot in 1:2) {
    r <- fromJSON(s[[paste0("genre", slot, "_ratings")]][i])
    k <- k + 1
    out[[k]] <- data.frame(participant_id = s$participant_id[i], slot = slot, item_id = r$item_id,
                           rating = as.numeric(r$response), stringsAsFactors = FALSE)
  }
  d <- do.call(rbind, out)
  d <- merge(d, stim[, c("item_id", "genre", "key", "cell_code", "is_foil", "display_text", "element",
                         "prevalence_pct", "distinctiveness_pct", "ensemble_pct", "Lg10CD")], by = "item_id")
  d$pid <- factor(d$participant_id); d$item <- factor(d$item_id)
  d$P <- ifelse(d$cell_code %in% c("HH", "HL"), 0.5, ifelse(d$cell_code %in% c("LH", "LL"), -0.5, NA))
  d$D <- ifelse(d$cell_code %in% c("HH", "LH"), 0.5, ifelse(d$cell_code %in% c("HL", "LL"), -0.5, NA))
  d$prevalence <- d$prevalence_pct - 0.5
  d$distinctiveness <- d$distinctiveness_pct - 0.5
  d$composite <- d$ensemble_pct - 0.5
  d
}
production_table <- function(path, ids) {
  m <- as.data.frame(read_parquet(path))
  m <- m[m$participant_id %in% ids, ]
  m <- merge(m, pool[, c("genre", "key", "prevalence_pct", "distinctiveness_pct", "ensemble_pct",
                         "Lg10CD", "rel_freq_in_genre")], by = c("genre", "key"))
  m$pid <- factor(m$participant_id); m$item <- factor(paste(m$genre, m$key, sep = "|"))
  m$prevalence <- m$prevalence_pct - 0.5
  m$distinctiveness <- m$distinctiveness_pct - 0.5
  m$composite <- m$ensemble_pct - 0.5
  m
}
# Centering and contrasts are set on the rows of each model.
prep <- function(d) {
  d <- d[!is.na(d$Lg10CD), ]
  d$lg <- d$Lg10CD - mean(d$Lg10CD)
  d$genre <- factor(d$genre)
  contrasts(d$genre) <- contr.sum(nlevels(d$genre))
  d
}
prep_nolg <- function(d) { d$genre <- factor(d$genre); contrasts(d$genre) <- contr.sum(nlevels(d$genre)); d }

R_all <- ratings_table(usable)
R48 <- R_all[!R_all$is_foil, ]
# The primary production matrix follows the coding decision; the liberal and strict sensitivity
# codings (and, when the human coding is primary, the automated coding) come from notebook 11.
prod_file <- function(tag, primary = TRUE) file.path(CODED, paste0("production_matrix", tag, if (primary) suffix else "", ".parquet"))
G <- production_table(prod_file(""), usable$participant_id)

# ------------------------------------------------------------------ model fitting with the registered reduction rule (7.1)
re_part <- function(slopes, corr) {
  if (!length(slopes)) return("(1 | pid) + (1 | item)")
  if (corr) return(sprintf("(1 + %s | pid) + (1 | item)", paste(slopes, collapse = " + ")))
  paste0("(1 | pid) + ", paste(sprintf("(0 + %s | pid)", slopes), collapse = " + "), " + (1 | item)")
}
clmm_failed <- function(m) {
  if (inherits(m, "try-error")) return(TRUE)
  se <- suppressWarnings(sqrt(diag(vcov(m))))
  sds <- unlist(lapply(VarCorr(m), function(v) attr(v, "stddev")))
  any(!is.finite(se)) || any(sds < SINGULAR_TOL)
}
fit_once <- function(fixed, slopes, corr, data, kind, REML) {
  f <- as.formula(paste(fixed, "+", re_part(slopes, corr)))
  msgs <- character()
  m <- withCallingHandlers(
    try(switch(kind,
               lmer = lmer(f, data = data, REML = REML, control = CTRL_L),
               glmer = glmer(f, data = data, family = binomial, control = CTRL_G),
               clmm = clmm(f, data = data, link = "logit")), silent = TRUE),
    warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleMessage") })
  failed <- if (inherits(m, "try-error")) TRUE else if (kind == "clmm") clmm_failed(m) else
    length(m@optinfo$conv$lme4$messages) > 0 || isSingular(m, tol = SINGULAR_TOL)
  list(model = m, failed = failed, formula = paste(deparse(f, width.cutoff = 500), collapse = ""),
       messages = unique(msgs), slopes = slopes, corr = corr)
}
slope_variances <- function(m, slopes) {
  vc <- VarCorr(m)
  sapply(slopes, function(s) {
    v <- NA
    for (g in seq_along(vc)) { M <- as.matrix(vc[[g]]); if (s %in% rownames(M)) v <- M[s, s] }
    v
  })
}
fit_reduced <- function(fixed, slopes, data, kind = "lmer", REML = TRUE, structure = NULL) {
  if (!is.null(structure)) {                          # refit with a given structure (cross-validation)
    fit <- fit_once(fixed, structure$slopes, structure$corr, data, kind, REML)
    fit$steps <- fit$formula; return(fit)
  }
  fit <- fit_once(fixed, slopes, TRUE, data, kind, REML); steps <- list(fit)
  if (fit$failed && length(slopes)) {
    fit <- fit_once(fixed, slopes, FALSE, data, kind, REML); steps <- c(steps, list(fit))
    while (fit$failed && length(slopes)) {
      if (length(slopes) == 2 && !inherits(fit$model, "try-error")) {
        slopes <- slopes[-which.min(slope_variances(fit$model, slopes))]
      } else if (length(slopes) == 2) slopes <- slopes[1] else slopes <- character()
      fit <- fit_once(fixed, slopes, FALSE, data, kind, REML); steps <- c(steps, list(fit))
    }
  }
  fit$steps <- vapply(steps, function(x) paste0(x$formula, if (x$failed) "   [failed: reduced]" else "   [used]"), "")
  if (fit$failed) fit$steps[length(fit$steps)] <- sub("\\[failed: reduced\\]", "[failed: no further reduction]", fit$steps[length(fit$steps)])
  fit
}

# ------------------------------------------------------------------ estimates
est_rows <- list()
add <- function(analysis, term, est, se, lo, hi, p, df = NA, note = "") {
  est_rows[[length(est_rows) + 1]] <<- data.frame(analysis = analysis, term = term, estimate = est, se = se,
                                                 df = df, lower = lo, upper = hi, p = p, note = note)
}
coef_row <- function(fit, term, analysis, note = "") {
  m <- fit$model
  cf <- summary(m)$coefficients
  if (inherits(m, "glmerMod")) {
    e <- cf[term, "Estimate"]; s <- cf[term, "Std. Error"]
    add(analysis, term, e, s, e - qnorm(.975) * s, e + qnorm(.975) * s, cf[term, "Pr(>|z|)"], NA, note)
  } else if (inherits(m, "clmm")) {
    e <- cf[term, "Estimate"]; s <- cf[term, "Std. Error"]
    add(analysis, term, e, s, e - qnorm(.975) * s, e + qnorm(.975) * s, cf[term, "Pr(>|z|)"], NA, note)
  } else {
    e <- cf[term, "Estimate"]; s <- cf[term, "Std. Error"]; df <- cf[term, "df"]
    add(analysis, term, e, s, e - qt(.975, df) * s, e + qt(.975, df) * s, cf[term, "Pr(>|t|)"], df, note)
  }
}
contrast_row <- function(fit, weights, label, analysis) {
  m <- fit$model
  L <- setNames(rep(0, length(fixef_any(m))), names(fixef_any(m)))
  L[names(weights)] <- weights
  if (inherits(m, "lmerModLmerTest")) {
    ct <- contest(m, L, joint = FALSE, confint = TRUE)
    add(analysis, label, ct$Estimate, ct$`Std. Error`, ct$lower, ct$upper, ct$`Pr(>|t|)`, ct$df)
  } else {
    b <- fixef_any(m); V <- as.matrix(vcov_fixed(m))
    e <- sum(L * b); s <- sqrt(as.numeric(t(L) %*% V %*% L))
    add(analysis, label, e, s, e - qnorm(.975) * s, e + qnorm(.975) * s, 2 * pnorm(-abs(e / s)))
  }
}
fixef_any <- function(m) if (inherits(m, "clmm")) m$beta else fixef(m)
vcov_fixed <- function(m) if (inherits(m, "clmm")) vcov(m)[names(m$beta), names(m$beta)] else vcov(m)
structures <- list()
note_structure <- function(name, fit) structures[[name]] <<- fit$steps

# ------------------------------------------------------------------ H1 (7.2)
H1_FIXED <- "produced ~ prevalence + distinctiveness + lg + genre"
G1 <- prep(G)
h1 <- fit_reduced(H1_FIXED, c("prevalence", "distinctiveness"), G1, "glmer"); note_structure("H1", h1)
for (t in c("prevalence", "distinctiveness", "lg")) coef_row(h1, t, "H1")
h1n <- fit_reduced("produced ~ prevalence + distinctiveness + genre", c("prevalence", "distinctiveness"), prep_nolg(G), "glmer")
note_structure("H1 without Lg10CD", h1n)
for (t in c("prevalence", "distinctiveness")) coef_row(h1n, t, "H1 without Lg10CD")
for (tag in c("liberal", "strict")) {
  Gs <- prep(production_table(prod_file(paste0("_", tag), primary = FALSE), usable$participant_id))
  f <- fit_reduced(H1_FIXED, c("prevalence", "distinctiveness"), Gs, "glmer"); note_structure(paste("H1", tag), f)
  coef_row(f, "prevalence", paste("H1", tag, "coding"))
}

if (PRIMARY == "human") {                               # Section 7.4.6: H1 is reported on both codings
  Ga <- prep(production_table(prod_file("", primary = FALSE), usable$participant_id))
  f <- fit_reduced(H1_FIXED, c("prevalence", "distinctiveness"), Ga, "glmer"); note_structure("H1 automated coding", f)
  coef_row(f, "prevalence", "H1 automated coding")
}

# production norm shares and the descriptive replication (Spearman, participant bootstrap)
norms <- do.call(rbind, lapply(split(G, G$genre), function(d) {
  n_p <- length(unique(d$participant_id))
  cnt <- tapply(d$produced, d$key, sum)
  data.frame(genre = d$genre[1], n_participants = n_p, n_elements = length(cnt),
             share_produced_by_any = mean(cnt >= 1), share_by_2 = mean(cnt >= 2),
             share_by_one_sixth = mean(cnt >= n_p / 6))         # 5 of 30, the McRae et al. rule
}))
set.seed(SEED)
spearman <- do.call(rbind, lapply(split(G, G$genre), function(d) {
  M <- tapply(d$produced, list(d$participant_id, d$key), sum); M[is.na(M)] <- 0   # participants x elements
  rf <- tapply(d$rel_freq_in_genre, d$key, function(x) x[1])[colnames(M)]
  boot <- replicate(2000, suppressWarnings(cor(colMeans(M[sample(nrow(M), replace = TRUE), , drop = FALSE]), rf, method = "spearman")))
  data.frame(genre = d$genre[1], spearman = cor(colMeans(M), rf, method = "spearman"),
             lower = unname(quantile(boot, .025, na.rm = TRUE)), upper = unname(quantile(boot, .975, na.rm = TRUE)))
}))

# ------------------------------------------------------------------ H2a, H2b, H2c (7.3)
H2_FIXED <- "rating ~ P + D + P:D + lg + genre"
R1 <- prep(R48)
h2 <- fit_reduced(H2_FIXED, c("P", "D"), R1, "lmer"); note_structure("H2", h2)
for (t in c("P", "D", "P:D", "lg")) coef_row(h2, t, "H2")
contrast_row(h2, c(P = 1, D = -1), "P minus D (H2c)", "H2")
for (cc in CELLS) {
  p <- if (cc %in% c("HH", "HL")) .5 else -.5; dd <- if (cc %in% c("HH", "LH")) .5 else -.5
  contrast_row(h2, c("(Intercept)" = 1, P = p, D = dd, "P:D" = p * dd), paste("adjusted mean", cc), "H2 cell means")
}
vc <- as.data.frame(VarCorr(h2$model))
sd_std <- sqrt(sum(vc$vcov[(vc$grp == "pid" & vc$var1 == "(Intercept)" & is.na(vc$var2)) |
                          (vc$grp == "item") | (vc$grp == "Residual")]))
h2n <- fit_reduced("rating ~ P + D + P:D + genre", c("P", "D"), prep_nolg(R48), "lmer"); note_structure("H2 without Lg10CD", h2n)
for (t in c("P", "D")) coef_row(h2n, t, "H2 without Lg10CD")
contrast_row(h2n, c(P = 1, D = -1), "P minus D", "H2 without Lg10CD")
R1o <- R1; R1o$rating_f <- factor(R1o$rating, levels = 1:7, ordered = TRUE)
h2o <- fit_reduced("rating_f ~ P + D + P:D + lg + genre", c("P", "D"), R1o, "clmm"); note_structure("H2 ordinal", h2o)
for (t in c("P", "D")) coef_row(h2o, t, "H2 ordinal (latent scale)")
contrast_row(h2o, c(P = 1, D = -1), "P minus D", "H2 ordinal (latent scale)")
h2c <- fit_reduced("rating ~ prevalence + distinctiveness + lg + genre", c("prevalence", "distinctiveness"), R1, "lmer")
note_structure("H2 continuous", h2c)
for (t in c("prevalence", "distinctiveness")) coef_row(h2c, t, "H2 continuous")
contrast_row(h2c, c(prevalence = 1, distinctiveness = -1), "prevalence minus distinctiveness", "H2 continuous")

# ------------------------------------------------------------------ M1 (8.1)
RM <- prep_nolg(R_all)
RM$itemclass <- relevel(factor(ifelse(RM$is_foil, "FOIL", RM$cell_code)), ref = "FOIL")
m1 <- fit_reduced("rating ~ itemclass + genre", character(), RM, "lmer"); note_structure("M1", m1)
for (cc in CELLS) contrast_row(m1, setNames(-1, paste0("itemclass", cc)), paste("foil minus", cc), "M1")

# ------------------------------------------------------------------ S1 (8.2)
s1 <- read.csv(file.path(CODED, "s1_genre_discrimination.csv"), colClasses = c(participant_id = "character"))
s1 <- s1[s1$participant_id %in% usable$participant_id, ]
s1$diff <- s1$cos_cued - s1$cos_other
s1p <- aggregate(diff ~ participant_id, data = s1, FUN = mean)
s1t <- t.test(s1p$diff)
add("S1", "mean within-participant cosine difference", unname(s1t$estimate), unname(s1t$stderr),
    s1t$conf.int[1], s1t$conf.int[2], s1t$p.value, unname(s1t$parameter))
s1w <- wilcox.test(s1p$diff)
s1_by_genre <- aggregate(cbind(cos_cued, cos_other, diff) ~ cued_genre, data = s1, FUN = mean)
s1_share_positive <- mean(s1$diff > 0)

# ------------------------------------------------------------------ H3 (7.5)
cv_folds <- function(ids) { set.seed(SEED); ids <- sort(unique(as.character(ids))); setNames(sample(rep(1:10, length.out = length(ids))), ids) }
# Held-out participants are scored by their marginal log-likelihood: item effects from the
# refit model, participant effects integrated over their estimated distribution (exactly for
# ratings, by Monte Carlo with 1,000 draws for generation). Setting them to zero instead
# would penalize the model whose richer random-effects structure leaves less residual variance.
pid_cov <- function(m) {
  vc <- VarCorr(m)
  blocks <- vc[grepl("^pid(\\.[0-9]+)?$", names(vc))]
  nm <- unlist(lapply(blocks, function(b) rownames(as.matrix(b))))
  G <- matrix(0, length(nm), length(nm), dimnames = list(nm, nm))
  for (b in blocks) { M <- as.matrix(b); G[rownames(M), colnames(M)] <- M }
  G
}
z_matrix <- function(d, nm) matrix(vapply(nm, function(n) if (n == "(Intercept)") rep(1, nrow(d)) else d[[n]],
                                          numeric(nrow(d))), nrow = nrow(d))
ll_gauss <- function(y, mu, Z, G, s2) {
  R <- chol(Z %*% G %*% t(Z) + diag(s2, length(y)))
  r <- backsolve(R, y - mu, transpose = TRUE)
  -0.5 * (length(y) * log(2 * pi) + 2 * sum(log(diag(R))) + sum(r^2))
}
ll_logit <- function(y, eta, Z, G, S = 1000) {
  ev <- eigen(G, symmetric = TRUE)
  L <- ev$vectors %*% diag(sqrt(pmax(ev$values, 0)), ncol(G))
  E <- eta + Z %*% (L %*% matrix(rnorm(S * ncol(G)), ncol(G), S))
  ll_s <- colSums(y * plogis(E, log.p = TRUE) + (1 - y) * plogis(-E, log.p = TRUE))
  mx <- max(ll_s); mx + log(mean(exp(ll_s - mx)))
}
heldout_ll <- function(fixed, structure, data, kind, outcome) {
  folds <- cv_folds(data$pid)
  per_fold <- numeric(10)
  for (k in 1:10) {
    set.seed(SEED + k)
    test_ids <- names(folds)[folds == k]
    train <- data[!(as.character(data$pid) %in% test_ids), ]; test <- data[as.character(data$pid) %in% test_ids, ]
    f <- fit_reduced(fixed, NULL, train, kind, REML = FALSE, structure = structure)
    G <- pid_cov(f$model)
    eta <- predict(f$model, newdata = test, re.form = ~ (1 | item), type = "link", allow.new.levels = TRUE)
    per_fold[k] <- sum(vapply(split(seq_len(nrow(test)), as.character(test$pid)), function(ix) {
      Z <- z_matrix(test[ix, ], rownames(G))
      if (kind == "glmer") ll_logit(test[[outcome]][ix], eta[ix], Z, G)
      else ll_gauss(test[[outcome]][ix], eta[ix], Z, G, sigma(f$model)^2)
    }, 0))
  }
  per_fold
}
h3 <- list()
for (o in list(list(name = "generation", data = G1, kind = "glmer", outcome = "produced"),
               list(name = "rating", data = R1, kind = "lmer", outcome = "rating"))) {
  two_f <- paste(o$outcome, "~ prevalence + distinctiveness + lg + genre")
  comp_f <- paste(o$outcome, "~ composite + lg + genre")
  two <- fit_reduced(two_f, c("prevalence", "distinctiveness"), o$data, o$kind, REML = FALSE)
  comp <- fit_reduced(comp_f, "composite", o$data, o$kind, REML = FALSE)
  note_structure(paste("H3", o$name, "two-dimension"), two); note_structure(paste("H3", o$name, "composite"), comp)
  cv_two <- heldout_ll(two_f, list(slopes = two$slopes, corr = two$corr), o$data, o$kind, o$outcome)
  cv_comp <- heldout_ll(comp_f, list(slopes = comp$slopes, corr = comp$corr), o$data, o$kind, o$outcome)
  h3[[o$name]] <- data.frame(outcome = o$name, AIC_two = AIC(two$model), AIC_composite = AIC(comp$model),
                             heldout_ll_two = sum(cv_two), heldout_ll_composite = sum(cv_comp),
                             supported = AIC(two$model) < AIC(comp$model) && sum(cv_two) > sum(cv_comp),
                             folds_favoring_two = sum(cv_two > cv_comp))
}
h3 <- do.call(rbind, h3)

# ------------------------------------------------------------------ robustness (9.1)
for (g in sort(unique(G1$genre))) {
  f <- fit_reduced(H1_FIXED, c("prevalence", "distinctiveness"), prep(G[G$genre != g, ]), "glmer")
  note_structure(paste("H1 without", g), f); coef_row(f, "prevalence", paste("H1 leaving out", g))
  f <- fit_reduced(H2_FIXED, c("P", "D"), prep(R48[R48$genre != g, ]), "lmer")
  note_structure(paste("H2 without", g), f)
  for (t in c("P", "D")) coef_row(f, t, paste("H2 leaving out", g))
  contrast_row(f, c(P = 1, D = -1), "P minus D", paste("H2 leaving out", g))
}
robust <- list(
  "H2 first genre block only" = R48[R48$slot == 1, ],
  "H2 without display-edited items" = R48[R48$display_text == R48$element, ],
  "H2 without significantly distracted participants" =
    R48[!(R48$participant_id %in% usable$participant_id[usable$distractions %in% "1"]), ])
for (nm in names(robust)) {
  f <- fit_reduced(H2_FIXED, c("P", "D"), prep(robust[[nm]]), "lmer"); note_structure(nm, f)
  for (t in c("P", "D")) coef_row(f, t, nm)
  contrast_row(f, c(P = 1, D = -1), "P minus D", nm)
}
f <- fit_reduced(H1_FIXED, c("prevalence", "distinctiveness"),
                 prep(G[!(G$participant_id %in% usable$participant_id[usable$distractions %in% "1"]), ]), "glmer")
note_structure("H1 without significantly distracted participants", f)
coef_row(f, "prevalence", "H1 without significantly distracted participants")

# ------------------------------------------------------------------ output
est <- do.call(rbind, est_rows)
est$odds_ratio <- ifelse(grepl("^H1", est$analysis), exp(est$estimate), NA)
est$standardized <- ifelse(grepl("^H2", est$analysis) & !grepl("ordinal", est$analysis), est$estimate / sd_std, NA)
write.csv(est, file.path(OUT, "estimates.csv"), row.names = FALSE)
write.csv(excl_report, file.path(OUT, "exclusions.csv"), row.names = FALSE)
write.csv(h3, file.path(OUT, "h3_model_comparison.csv"), row.names = FALSE)
write.csv(spearman, file.path(OUT, "descriptive_spearman.csv"), row.names = FALSE)
write.csv(norms, file.path(OUT, "production_norm_shares.csv"), row.names = FALSE)
write.csv(s1_by_genre, file.path(OUT, "s1_by_genre.csv"), row.names = FALSE)

decide <- function(a, t) {
  r <- est[est$analysis == a & est$term == t, ]
  if (r$lower > 0) "supported (interval above zero)" else if (r$upper < 0) "contradicted (interval below zero)" else "not supported (interval includes zero)"
}
h2c <- est[est$analysis == "H2" & est$term == "P minus D (H2c)", ]
m1r <- est[est$analysis == "M1", ]
sink(file.path(OUT, "results.txt"))
cat("Study 1 registered analysis |", MODE, "mode |", format(Sys.time()), "\n")
cat("Primary coding:", PRIMARY, "| usable participants:", nrow(usable), "\n")
cat("Usable participants per genre:\n"); print(table(c(usable$g1, usable$g2)))
cat("\nExclusions (rules applied in order):\n"); print(excl_report, row.names = FALSE)
cat("\nDecisions\n")
cat("H1  prevalence:", decide("H1", "prevalence"), "\n")
cat("H2a P:", decide("H2", "P"), "\n")
cat("H2b D:", decide("H2", "D"), "\n")
cat(sprintf("H2c P minus D: %.3f [%.3f, %.3f] -> %s\n", h2c$estimate, h2c$lower, h2c$upper,
            if (h2c$lower > 0) "expectation follows prevalence more closely" else if (h2c$upper < 0)
              "expectation follows distinctiveness more closely" else "no detectable difference"))
cat("M1  all four foil contrasts below zero:", all(m1r$upper < 0), "\n")
cat("H3  supported by outcome:\n"); print(h3, row.names = FALSE)
cat(sprintf("\nS1  mean cosine difference %.4f, t test p = %.4g, Wilcoxon p = %.4g, share of genre blocks positive %.3f\n",
            s1t$estimate, s1t$p.value, s1w$p.value, s1_share_positive))
cat("\nStandardizing SD for rating effects:", round(sd_std, 3), "\n")
cat("\nAll estimates:\n"); print(est, row.names = FALSE, digits = 4)
cat("\nRandom-effects structures (reduction steps):\n")
for (nm in names(structures)) { cat("-", nm, "\n"); cat(paste0("    ", structures[[nm]], "\n"), sep = "") }
cat("\nDescriptive replication (Spearman, participant bootstrap):\n"); print(spearman, row.names = FALSE)
cat("\nProduction norm shares:\n"); print(norms, row.names = FALSE)
cat("\n"); print(sessionInfo())
sink()
cat("wrote", OUT, "\n")
