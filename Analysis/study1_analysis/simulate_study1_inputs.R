# Simulated inputs for study1_analysis.R in simulate mode (sourced by that script).
# Writes session files in the experiment's data format to OUT/raw/ and stand-ins for
# the outputs of notebook 11 to OUT/coded/, so that every step of the registered
# analysis can be run and checked before real data exist. The generating effects
# are known, so the estimates can be compared with them:
#   ratings: P = 0.5, D = 0.4 scale points, foils 1 point below the LL cell;
#   production: prevalence 1.0 and distinctiveness 1.0 on the logit scale, with participant
#   intercept SD 0.5 and slope SDs 0.4.
# Each generation pool is subsampled to SIM_POOL_N elements per genre for speed.

set.seed(SEED)
SIM_N <- 100
SIM_POOL_N <- 150
dir.create(RAW, recursive = TRUE, showWarnings = FALSE); unlink(list.files(RAW, full.names = TRUE))
dir.create(CODED, recursive = TRUE, showWarnings = FALSE)

pairs <- list(c("western", "horror"), c("crime", "science_fiction"), c("comedy", "fantasy"),
              c("western", "crime"), c("horror", "fantasy"), c("science_fiction", "comedy"))
cell_genres <- function(cell) { p <- pairs[[cell %/% 2 + 1]]; if (cell %% 2 == 0) p else rev(p) }
key_to_display <- setNames(names(GENRE_KEY), GENRE_KEY)

ids <- as.character(1000 + seq_len(SIM_N))
p_int <- setNames(rnorm(SIM_N, 0, 0.7), ids)
p_sP <- setNames(rnorm(SIM_N, 0, 0.3), ids); p_sD <- setNames(rnorm(SIM_N, 0, 0.3), ids)
item_eff <- setNames(rnorm(nrow(stim), 0, 0.8), stim$item_id)

rate_block <- function(pid, gkey) {
  it <- stim[stim$genre == key_to_display[[gkey]], ]
  P <- ifelse(it$cell_code %in% c("HH", "HL"), .5, -.5); D <- ifelse(it$cell_code %in% c("HH", "LH"), .5, -.5)
  mu <- 4 + 0.5 * P + 0.4 * D + p_sP[pid] * P + p_sD[pid] * D
  mu[it$is_foil] <- 4 - 0.25 - 0.2 - 1
  it$response <- pmin(7, pmax(1, round(mu + p_int[pid] + item_eff[it$item_id] + rnorm(nrow(it), 0, 1.3))))
  it <- it[sample(nrow(it)), ]
  as.character(toJSON(data.frame(pos = seq_len(nrow(it)), item_id = it$item_id, text = it$display_text,
                                 cell_code = it$cell_code, is_foil = it$is_foil, response = it$response),
                      auto_unbox = TRUE))
}

session <- function(pid, start, overrides = list()) {
  cell <- suppressWarnings(as.integer(pid)) %% 12
  if (is.na(cell)) cell <- 0                       # test identifiers run cell 0 here
  g <- cell_genres(cell)
  s <- list(participant_id = pid, is_test = "false", platform = "sona", platform_source = "numeric_id",
            cell_index = cell, g1 = g[1], g2 = g[2], start_time = start, end_time = "2027-01-15T10:30:00Z",
            captcha_pass = "true", spr_fast_any = "false", task_check_failed_both = "false",
            attention_pass_total = 4, attention_total = 4, ai_usage = "No", seriousness = "2",
            distractions = sample(c("1", "2", "3"), 1, prob = c(.1, .4, .5)),
            genre1_familiarity = sample(1:7, 1), genre2_familiarity = sample(1:7, 1),
            genre1_ratings = rate_block(pid, g[1]), genre2_ratings = rate_block(pid, g[2]))
  s[names(overrides)] <- overrides
  as.data.frame(s, stringsAsFactors = FALSE)
}
write_session <- function(s, tag) write.csv(s, file.path(RAW, sprintf("SAF_STUDY1_%s_SESSION_%s.csv", s$participant_id, tag)), row.names = FALSE)

# One planted case per exclusion rule (rule 6 is planted in the quality flags below).
planted <- list("1091" = list(captcha_pass = "false"), "1092" = list(spr_fast_any = "true"),
                "1093" = list(task_check_failed_both = "true"), "1094" = list(attention_pass_total = 3),
                "1095" = list(end_time = ""), "1097" = list(ai_usage = "Yes"), "1099" = list(seriousness = "1"))
for (i in seq_along(ids)) {
  pid <- ids[i]
  write_session(session(pid, sprintf("2027-01-%02dT09:%02d:00Z", 10 + i %/% 60, i %% 60), planted[[pid]] %||% list()), "a")
}
write_session(session("1098", "2027-02-01T09:00:00Z"), "b")              # a second session from 1098: rule 8
write_session(session("PILOT_C0", "2027-01-01T09:00:00Z", list(is_test = "true")), "pilot")

# Stand-ins for notebook 11 outputs
sub_pool <- do.call(rbind, lapply(split(pool, pool$genre), function(d) d[sample(nrow(d), min(SIM_POOL_N, nrow(d))), ]))
u_item <- setNames(rnorm(nrow(sub_pool), 0, 1), paste(sub_pool$genre, sub_pool$key))
pg <- do.call(rbind, lapply(ids, function(pid) { g <- cell_genres(as.integer(pid) %% 12)
  data.frame(participant_id = pid, genre = key_to_display[g], stringsAsFactors = FALSE) }))
prod <- merge(pg, sub_pool[, c("genre", "key", "prevalence_pct", "distinctiveness_pct")], by = "genre")
g_int <- setNames(rnorm(SIM_N, 0, 0.5), ids); g_sP <- setNames(rnorm(SIM_N, 0, 0.4), ids); g_sD <- setNames(rnorm(SIM_N, 0, 0.4), ids)
eta <- -4 + (1.0 + g_sP[prod$participant_id]) * (prod$prevalence_pct - .5) +
  (1.0 + g_sD[prod$participant_id]) * (prod$distinctiveness_pct - .5) +
  g_int[prod$participant_id] + u_item[paste(prod$genre, prod$key)]
prod$produced <- rbinom(nrow(prod), 1, plogis(eta))
base <- prod[, c("participant_id", "genre", "key", "produced")]
write_parquet(base, file.path(CODED, "production_matrix.parquet"))
write_parquet(transform(base, produced = pmax(produced, rbinom(nrow(base), 1, .01))), file.path(CODED, "production_matrix_liberal.parquet"))
write_parquet(transform(base, produced = produced * rbinom(nrow(base), 1, .7)), file.path(CODED, "production_matrix_strict.parquet"))

flags <- do.call(rbind, lapply(ids, function(pid) data.frame(participant_id = pid, slot = 1:2,
  genre = cell_genres(as.integer(pid) %% 12), n_entries = 14, share_uninterpretable = 0, share_instruction_echo = 0, flag = "False")))
flags[flags$participant_id == "1096", c("share_uninterpretable", "flag")] <- list(0.6, "True")
write.csv(flags, file.path(CODED, "quality_flags.csv"), row.names = FALSE)

s1 <- do.call(rbind, lapply(ids, function(pid) { g <- cell_genres(as.integer(pid) %% 12)
  data.frame(participant_id = pid, slot = 1:2, cued_genre = g, other_genre = rev(g), n_entries = 14,
             cos_cued = 0.30 + rnorm(2, 0, 0.02), cos_other = 0.28 + rnorm(2, 0, 0.02)) }))
write.csv(s1, file.path(CODED, "s1_genre_discrimination.csv"), row.names = FALSE)
write_json(list(primary = "automated", kappa_human_human = 0.8, kappa_human_automated_mean = 0.75),
           file.path(CODED, "coding_decision.json"), auto_unbox = TRUE)
cat("simulated inputs written to", OUT, "\n")
