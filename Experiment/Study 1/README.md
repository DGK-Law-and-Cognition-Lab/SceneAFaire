# Study 1 (SAF_STUDY1): Genre Element Generation and Expectedness Ratings

Plain-JavaScript Pavlovia experiment. No build step, no PsychoJS. After consent,
a two-sentence reading task (moving-window self-paced reading) flags implausibly
fast or automated responding. Each participant then completes two genres; for each genre they (1) rate familiarity, (2) generate
typical narrative elements in five cued categories, and (3) rate the expectedness
of 56 pre-selected words or phrases (48 target items from a 2 x 2
prevalence x distinctiveness design plus 8 foils from other genres), with two
instructed-response attention checks mixed in.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page shell (loading screen, progress bar, page container). Loads `experiment.js`. |
| `experiment.js` | The whole experiment: constants, Pavlovia plumbing, page system, tasks, data record. |
| `style.css` | Styles. |
| `captcha.png` | Image used by the hidden-instruction captcha trap (unchanged from SA_GEN1). |
| `config.json` | Record of the Pavlovia project identity (`genres-and-themes/experiment-1`, GitLab project id 510161). Not pushed: that repository has no config.json, so Pavlovia generates the correct one. (A config.json committed to a repository overrides Pavlovia's, which is what broke `characters/study-2`.) |
| `deploy_to_pavlovia.sh` | Pushes the page files and stimuli to the root of `genres-and-themes/experiment-1` only. The address is fixed and re-checked after cloning, and you must type experiment-1 to push. Not itself uploaded. |
| `stimuli/stimuli_<genre>.json` | Six stimulus files (western, horror, science_fiction, crime, comedy, fantasy), copied from `Analysis/v4_rebuild/10_human_study_stimuli/`. |
| `stimuli/stimuli_all.js` | Generated inline fallback that sets `window.STIMULI = {...}` from the six JSON files. Only used when `fetch` fails (for example when `index.html` is opened directly over `file://`). |
| `README.md` | This file. |

## What changed from SA_GEN1 (Experiment/Study 1)

Kept as is: Pavlovia init/save via the REST API, desktop-only device gate,
full-screen page, hidden-instruction captcha trap, consent page (same UCLA Box
link), demographics, familiarity scale, generation UI (chips, duplicate check,
Enter to add), seriousness/full-screen/distraction questions, feedback boxes,
debrief text with the UCLA contact, CSV download fallback when not on Pavlovia.

Changed:

1. **Genres.** Six genres (Western, Horror, Science Fiction, Crime, Comedy,
   Fantasy) instead of four (Comedy, War, Adventure, Animation). Keys match the
   stimulus file names.
2. **Assignment.** Two rounds of three disjoint pairs ("waves" A and B), matching the
   preregistration's assignment schedule, so every genre has two partners. The study runs from a single SONA listing, so the experiment assigns the cell
   itself: the numeric SONA survey code modulo 12 picks one of the twelve
   ordered-pair cells (near-even, reproducible); a non-numeric code gets a random
   cell. URL parameters (`wave`, `pair`, `order`, or `g1`/`g2`) override this and
   are used for the pilot and for testing. `wave`, `pair_index`, `order`, `g1`, `g2` are recorded.
3. **Per-genre sequence.** Familiarity first, then generation, then a rating
   block. SA_GEN1 ran generation, ranking, then familiarity.
4. **Ranking task removed** entirely (drag-and-drop list and its CSS are gone).
5. **Rating block added.** All 56 items from the genre's JSON, shuffled per
   participant, plus two attention checks ("Attention check: please select 2
   for this item." and "... select 6 ...") at random positions between item 10
   and item 50. The two checks show only their own instruction, without the
   genre question under them. Question for every other item: "How expected is this in a typical
   <Genre> film?" on a 7-point scale, 1 = Not at all expected, 7 = Completely
   expected. Shown 8 items per page (`RATING_ITEMS_PER_PAGE`; set to 1 for one
   item per screen). Every item on a page must be answered. Per-item response
   time is recorded (ms from page render to first answer).
6. **Generation rules.** At least 2 in every one of the five boxes (Settings, Characters,
   Objects, Events, Themes) and at most 20 in total, so 10 to 20 entries. All five boxes
   are required since October 22, 2026; earlier drafts made Themes optional.
7. **Serial-position bug fixed.** Each entry receives its 1-based generation
   `order` when it is added and is never renumbered. Removing an entry marks it
   `removed: true` (with `removed_ms`) and keeps it in the data. Each entry also
   records `added_ms` (ms since the generation page was shown).
8. **Stimuli are loaded at runtime**, never hard-coded. `fetch` of
   `stimuli/stimuli_<genre>.json` relative to the page, with
   `stimuli/stimuli_all.js` as a fallback.
9. **Participant identity.** `ST.pid` = `PROLIFIC_PID`, then `survey_code`,
   `participant`, `id`. `STUDY_ID` and `SESSION_ID` are recorded when present.
10. **Completion routing.** Data is saved first (`await PAVLOVIA.save`), then:
    Prolific redirect if `platform` is `prolific` and `PROLIFIC_COMPLETION_URL`
    is non-empty; SONA credit redirect if `?platform=sona`; otherwise a
    completion page showing `COMPLETION_CODE`.
11. **No hard-coded "Genre N of 2".** Labels use `GENRES_PER_PARTICIPANT`.
12. **No em-dashes** in UI text; scale labels use the "1 = Not at all familiar"
    style.
13. Data file prefix is `SAF_STUDY1` (constant `STUDY_CODE`).

## Constants (top of `experiment.js`)

| Constant | Default | Meaning |
| --- | --- | --- |
| `STUDY_CODE` | `'SAF_STUDY1'` | Prefix of the data file name. |
| `GENRES` | six entries | `key` and `display` for each genre; keys match `stimuli/stimuli_<key>.json`. |
| `GENRES_PER_PARTICIPANT` | `2` | Genres per participant. Each `PAIR_SETS` entry must list this many keys. |
| `PAIR_SETS` | A, B | Wave A: western-horror, crime-science_fiction, comedy-fantasy. Wave B: western-crime, horror-fantasy, science_fiction-comedy. Matches `Docs/Preregistration/Study1_assignment_schedule.csv`. |
| `DEFAULT_WAVE` | `'A'` | Unused by the SONA assignment path (kept for compatibility). |
| `CATEGORIES` | 5 entries | Generation categories; `required: true` entries need `MIN_PER_CAT`. |
| `MIN_PER_CAT`, `MIN_TOTAL`, `MAX_TOTAL` | `2`, `10`, `20` | Generation limits. |
| `FAMILIARITY_LABELS` | 7 labels | "N = text" style; unlabeled points show only the number. |
| `RATING_SCALE_MIN/MAX` | `1`, `7` | Rating scale. |
| `RATING_ANCHOR_LOW/HIGH` | `'1 = Not at all expected'`, `'7 = Completely expected'` | Rating anchors. |
| `RATING_ITEMS_PER_PAGE` | `8` | Items per rating page (1 = one per screen). |
| `ATTENTION_CHECKS` | 2 entries | Text and expected response for each check. |
| `ATTENTION_MIN_POS`, `ATTENTION_MAX_POS` | `10`, `50` | 1-based position window for attention checks in the final sequence. |
| `PROLIFIC_COMPLETION_URL` | `''` | Prolific completion link. Empty means no redirect (completion code shown instead). |
| `COMPLETION_CODE` | `'SAFSTUDY1'` | Code shown on the completion page. |
| `SONA_URL_BASE` | credit URL of SONA experiment 3298 (set Oct 22, 2026) | The credit-granting URL from the SONA listing, ending in `survey_code=`; the participant id is appended. If emptied, `platform=sona` sessions end on the completion page instead of redirecting. The copy in the GitHub release has the credit token blanked. |
| `CONSENT_URL`, `CONTACT_NAME`, `CONTACT_EMAIL` | UCLA Box link, David G. Kamper, davidgkamper@ucla.edu | Consent and debrief contact. |
| `ESTIMATED_MINUTES` | `'25 to 35'` | Shown on the instructions page. |
| `MIN_SECONDS` | instructions 15, readingIntro 5, familiarity 7, ratingIntro 7, ratingPage 8, taskCheck 5 | Seconds before Next is enabled on those pages (0 = no wait). Answers are still required. Consent, generation, demographics and final pages have no wait. Total forced wait is about 3 minutes. |
| `SPR_SENTENCES` | 2 sentences | Reading-task sentences (same as the Second Amendment study). |
| `SPR_FAST_MEDIAN_MS` | `150` | Preregistered exclusion: median per-word time below this on either sentence. Recorded only; nothing is shown to the participant. |
| `TASK_CHECK` | 5 options, `'3'` correct | "What is your task in this study?" after the instructions. A wrong answer shows a notice, the instructions again, and one retry. Failing both is a preregistered exclusion; the participant still finishes and gets credit. |

## URL parameters

| Parameter | Values | Effect |
| --- | --- | --- |
| `g1`, `g2` | genre keys (`western`, `horror`, `science_fiction`, `crime`, `comedy`, `fantasy`) | Highest priority. Both must be valid and different. Presented in this order. If the pair appears in `PAIR_SETS`, `wave`, `pair_index` and `order` are filled in from it; otherwise `wave = custom`, `pair_index = -1`, `order = 1`. |
| `wave` | `A`, `B` | Forces the round. Only for test links. If absent, the cell comes from the survey code modulo 12 (random for a non-numeric id). Never put `wave` in the SONA link. |
| `pair` | `0`, `1`, `2` | With `wave`: index into the round's pair list. Random if missing or invalid. |
| `order` | `1`, `2` | With `wave`: `1` = pair as listed, `2` = reversed. Random if missing or invalid. |
| `PROLIFIC_PID` | string | Participant id (first choice). Also sets `platform = prolific` unless `platform` is given. |
| `STUDY_ID`, `SESSION_ID` | string | Recorded as `study_id`, `session_id`. |
| `survey_code` | string | Participant id (used when `PROLIFIC_PID` is absent). SONA's own link format uses `participant=` instead; either works. |
| `participant`, `id` | string | Further fallbacks for the participant id. An id beginning `PILOT` or `TEST` is recorded with `is_test = true` and is dropped before analysis. |
| `platform` | `prolific`, `sona`, or any string | Controls completion routing. `sona` redirects to the SONA credit URL. If absent, a numeric participant id (as SONA sends) is treated as `sona`; a non-numeric id (PILOT..., TEST...) is not. Recorded with `platform_source`. |
| `__pilotToken` | string | Added automatically by Pavlovia when piloting. |

Examples:

```
index.html?participant=PILOT_DK1&wave=A&pair=0&order=1     (your own test run, forced cell 0)
index.html?participant=12345                               (what a SONA participant gets; cell = 12345 mod 12)
```

## Deploy on Pavlovia

The study lives at the root of the Pavlovia project `genres-and-themes/experiment-1`
(GitLab project id 510161). This is the address the SONA listing uses:
`https://run.pavlovia.org/genres-and-themes/experiment-1`. Results go to that
project's `data/` folder as `SAF_STUDY1_<id>_SESSION_<date>.csv`. (From October 19
to 22, 2026 a copy was hosted in the `genres/` folder of `characters/study-2`; that
copy is no longer used.)

1. Upload. From this folder run `./deploy_to_pavlovia.sh`. It clones
   `genres-and-themes/experiment-1` (and stops if the clone points anywhere else or
   lacks the `data/` folder), replaces `index.html`, `experiment.js`, `style.css` and
   `captcha.png`, updates `stimuli/`, shows what changed, and pushes only after you
   type experiment-1. It does not push `config.json` and touches nothing else. Without the terminal:
   upload the same files in GitLab, with the stimuli in a `stimuli` folder.
2. Status. On the Pavlovia dashboard the project must be RUNNING and covered by
   the UCLA license (or have credits), with results saved as CSV. RUNNING means
   anyone with the link can start it; until SONA opens, only the investigators
   have the link.
3. Your own test run (no SONA, no credit):
   `https://run.pavlovia.org/genres-and-themes/experiment-1/index.html?participant=PILOT_DK1&wave=A&pair=0&order=1`.
   Use a new id each run (PILOT_DK2, ...). Change `wave`, `pair`, `order` to see
   other genre pairs; the twelve combinations are in
   `Docs/Preregistration/Study1_assignment_schedule.csv`. If the page says the
   study could not connect to its data server, Pavlovia refused the session and
   nothing would have been saved.
4. Check the data. After the completion page, a file
   `data/SAF_STUDY1_PILOT_DK1_SESSION_<date>.csv` should appear in the repository
   (or under Download results on the Pavlovia dashboard).
5. SONA. The listing's study URL is
   `https://run.pavlovia.org/genres-and-themes/experiment-1?participant=%SURVEY_CODE%`,
   which SONA fills with each participant's numeric survey code. A numeric id with
   no `platform` parameter is treated as a SONA session (`platform_source =
   numeric_id`), so at the end the participant is sent to SONA experiment 3298's
   credit URL (`SONA_URL_BASE`) with their code. Adding `&platform=sona` is
   harmless but not needed. Never add `wave`, `pair` or `order` to the SONA link:
   they override the automatic assignment.

Session handling: `PAVLOVIA.init` opens a session on load. On the live site the
study refuses to start if no session opens, so a participant can never finish
with data that only exists on their own computer. `PAVLOVIA.save` uploads the CSV
and closes the session as completed. If the tab closes after the upload has
started, an `unload` beacon retries it. Local runs (file://, localhost) never
contact Pavlovia, even though `config.json` is present, so testing here cannot
write into the live `data/` folder.

## Replacing the stimuli

1. Re-run `Analysis/notebooks/10_human_study_stimuli.ipynb`. It writes
   `Analysis/v4_rebuild/10_human_study_stimuli/stimuli_<genre>.json` (plus the
   long CSV/parquet, review sheet and report).
2. Copy the six JSON files into `Experiment/Study 1/stimuli/`, overwriting the
   old ones.
3. Regenerate the inline fallback from inside the `stimuli/` folder:

   ```
   cd "Experiment/Study 1/stimuli"
   python3 - <<'PY'
   import json
   genres = ['western','horror','science_fiction','crime','comedy','fantasy']
   out = {g: json.load(open('stimuli_%s.json' % g, encoding='utf-8')) for g in genres}
   with open('stimuli_all.js', 'w', encoding='utf-8') as f:
       f.write('// stimuli_all.js (GENERATED FILE; do not edit by hand)\n')
       f.write('// Inline fallback for the six stimuli_<genre>.json files in this folder.\n')
       f.write('window.STIMULI = ' + json.dumps(out, ensure_ascii=True, indent=1) + ';\n')
   PY
   ```

4. Run `./deploy_to_pavlovia.sh`.

The experiment reads these fields from each item: `id`, `text`, `cell_code`,
`is_foil`, `home_genre`. The number of items per genre is whatever the file
contains (currently 56); the rating block adapts. If you rename a genre key,
change the `GENRES` table and `PAIR_SETS` to match.

## Data columns

One row per participant. JSON-valued columns are strings; parse them in analysis.
Arrays of simple values (checklists) are joined with `;`.

Identity and session:
`participant_id`, `is_test` (id begins PILOT or TEST), `pavlovia_session` (`open`, `local`, `failed`), `prolific_pid`, `study_id`, `session_id`, `platform`,
`start_time`, `end_time`, `total_duration_ms`, `n_pages`, `page_timestamps`
(JSON list of `{p, t}`), `user_agent`, `screen_width`, `screen_height`,
`device_type`, `device_self_report`, `fullscreen_entered`, `atlas_detected`,
`captcha_response`, `captcha_pass`, `consent`, `consent_time`.

Assignment:
`wave` (A, B or custom), `pair_index` (0 to 2, or -1 for a custom URL pair),
`order` (1 or 2), `cell_index` (0 to 11, -1 for a custom pair), `assignment_source`
(`url_genres`, `url_wave`, `survey_code_mod`, `random`),
`genres_per_participant`, `g1`, `g2`, `g1_display`, `g2_display`, `genre_order`,
`stimuli_source` (`fetch`, `inline`, `mixed`), `stimuli_meta` (JSON: per genre
`n_items`, `seed`, `foil_rule`).

Per genre slot N (1 and 2), where `gN` tells you which genre the slot held:

- `genreN_familiarity` (1 to 7)
- `genreN_gen_time_ms`, `genreN_gen_count` (active entries),
  `genreN_gen_removed_count`, `genreN_gen_<category>_count` for settings,
  characters, objects, events, themes
- `genreN_gen_entries`: JSON list of `{text, category, order, added_ms, removed,
  removed_ms}`. `order` is the 1-based serial position at the time of adding and
  is never renumbered; removed entries stay in the list with `removed: true`.
- `genreN_rating_order`: JSON list of item ids in presentation order (attention
  check ids `ATT_SELECT_2`, `ATT_SELECT_6` included).
- `genreN_ratings`: JSON list of `{pos, item_id, text, cell_code, is_foil,
  home_genre, response, rt_ms, rt_last_ms, n_changes, page}` for the 56 stimulus
  items. `rt_ms` is ms from page render to the first answer on that item;
  `rt_last_ms` to the last change.
- `genreN_attention`: JSON list of `{id, pos, expected, response, pass, rt_ms,
  page}`; `genreN_attention_pass`, `genreN_attention_total`.
- `genreN_n_rated`, `genreN_rating_time_ms` (first rating page shown to last
  page submitted), `genreN_time_ms` (familiarity page shown to last rating page
  submitted).

Totals: `attention_pass_total`, `attention_total`.

Reading task, for each of `spr1` and `spr2`: `sprN_rts` (`;`-separated ms each word
stayed on screen, one per word), `sprN_mean_rt`, `sprN_median_rt`, `sprN_min_rt`,
`sprN_repeat_presses` (held-spacebar repeats, which do not advance the window),
`sprN_understood` (`yes` or `no`), `sprN_fast` (median below `SPR_FAST_MEDIAN_MS`).
`spr_fast_any` (either sentence fast; exclusion).

Task check: `task_check`, `task_check_pass`, `task_check_retry`, `task_check_retry_pass`
(empty when the first answer was right), `task_check_failed_both` (exclusion).

Demographics: `age`, `gender`, `race`, `race_other`, `native_language`,
`english_proficiency`, `education`, `country`, `employment`, `occupation`,
`films_per_month`, `tv_shows`, `genre_familiarity_checklist`, `film_knowledge`,
`viewing_platforms`, `film_reviews`, `professional_training`, `ai_usage`.

Final questions: `seriousness`, `fullscreen_maintained`, `distractions`,
`feedback_procedure`, `feedback_other`.

## Local test

1. In a terminal:

   ```
   cd "Experiment/Study 1"
   python3 -m http.server 8000
   ```

2. Open `http://localhost:8000/index.html?g1=western&g2=comedy&order=1` in a
   desktop browser. On localhost `PAVLOVIA.init` does not contact Pavlovia, so
   the experiment runs in local mode.
3. On the captcha page press `q` (the on-screen image instruction); pressing `z`
   (the hidden trap instruction) ends the session.
4. Work through both genres. The console shows `=== STUDY DATA ===` with the full
   record on the final page, and a CSV named `SAF_STUDY1_<pid>_<timestamp>.csv`
   downloads instead of uploading.
5. Other useful test URLs: `?wave=B&pair=1&order=2`, `?platform=sona&survey_code=12347`
   (cell 11; redirects to SONA only once `SONA_URL_BASE` is set), `?PROLIFIC_PID=test123`
   (shows the Prolific completion-code page while `PROLIFIC_COMPLETION_URL` is
   empty).
6. Opening `index.html` directly from Finder (`file://`) also works: fetch is
   blocked, so the inline `stimuli/stimuli_all.js` fallback is used and
   `stimuli_source` is recorded as `inline`.

Syntax check without a browser: `node --check experiment.js` and
`node --check stimuli/stimuli_all.js`.
