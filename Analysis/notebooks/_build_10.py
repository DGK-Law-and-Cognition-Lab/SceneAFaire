#!/usr/bin/env python3
# Builder for notebooks/10_human_study_stimuli.ipynb
# Run:  python3 notebooks/_build_10.py
# Same pattern as notebooks/_build_07b.py: every cell is authored here as a
# string and written out with nbformat, so the notebook is reproducible from
# source and can be diffed.
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------
md(r"""
# Notebook 10: Human-study stimulus export

This notebook exports the stimulus set for Study 1, a genre-cued generation task with an expectedness-rating block, from the classified stockiness table produced by Notebooks 03 and 03b. It writes nothing back into any upstream directory; every output goes to a new directory, `v4_rebuild/10_human_study_stimuli/`.

The design follows the two-dimension framing of stock-ness. Prevalence is how common an element is within a genre (of all Westerns, how many contain a sheriff), measured here by `rel_freq_in_genre`. Distinctiveness is how specific an element is to a genre (of all films with a sheriff, how many are Westerns), measured here by the smoothed posterior `posterior_smoothed`. The two come apart in the data, and the stimulus set is stratified on both so the human study can ask which of them an audience is responding to when it calls an element "expected" in a genre. Each genre gets four cells (high and low on each dimension, crossed) plus a set of foils drawn from other genres that are rare in the target genre (fewer than five films) or absent from it, with a size-scaled fallback threshold for any genre where that rule finds too few candidates.

Every constant that controls the export sits in the configuration cell immediately below. Change it there, rerun, and the outputs regenerate deterministically from the seed.
""")

# ----------------------------------------------------------------------
code(r"""
# Configuration. Every editable constant lives here.

GENRES = ['Western', 'Horror', 'Science Fiction', 'Crime', 'Comedy', 'Fantasy']

SEED = 20260219                  # project preregistered seed

MIN_FILMS_IN_GENRE = 20          # reliability floor from the Sept 2026 audit:
                                 # an element must appear in at least this
                                 # many films of a genre to be a candidate there

K_PER_CELL = 12                  # elements sampled per stratum (cell) per genre
N_FOILS = 8                      # foils per genre, drawn from other genres

ALLOWED_TYPES = ['noun', 'chunk']   # verbs excluded by default; entities never

FOIL_MAX_FILMS_IN_TARGET = 5     # the foil rule: a foil has film_count below
                                 # this in the target genre, or is absent from
                                 # it. Applied to every target genre where it
                                 # yields at least N_FOILS candidates
FOIL_MAX_RATE_IN_TARGET = 0.002  # fallback for a target genre where the flat
                                 # rule yields fewer than N_FOILS candidates:
                                 # its threshold becomes max(flat rule, this
                                 # rate x films in the target). In the current
                                 # data only Comedy (14,722 films) needs it,
                                 # because any element common anywhere appears
                                 # in at least five comedies. The report names
                                 # every genre that used the fallback
FOILS_UNIQUE_ACROSS_SET = True   # a foil string is used for at most one target
                                 # genre and never coincides with any selected
                                 # stimulus; target genres are served in order
                                 # of ascending candidate count so the scarcest
                                 # genre picks first

SUBTLEX_PATH = 'Data/SUBTLEX-US/SUBTLEXus74286wordstextversion.txt'  # relative to the project root, or absolute; None to skip.
                                 # If None or missing, the join is skipped and a
                                 # note is printed and written to the report.

# Lowercase strings the review rejected (narrative_ok = n). Three independent
# judges rated all 336 candidates of the first draw on October 14, 2026, with a
# majority vote; sixteen were rejected by majority and three more (indian,
# entity, actor) were rejected by the investigator on review of the split
# decisions. Decisions and reasons are in stimuli_review_sheet_judged.csv and
# stimuli_rejects_for_confirmation.md. Add strings here and rerun rather than
# editing output files by hand.
# Restored October 21, 2026 by PI decision, after earlier exclusion: 'black man',
# 'white man' (race labels kept as character types) and 'orgasm', 'intercourse',
# 'sex' (sexual content kept; judged necessary). They are no longer excluded, so
# each returns to its place in its cell's seeded queue.
MANUAL_EXCLUDE = {
    'testing', 'new light', 'antagonist', 'dismay', 'yorker', 'weight',   # Comedy
    'west', 'parody', 'reality',                                          # Crime
    'horror', 'charge', 'star', 'theme',                                  # Horror
    'production',                                                         # Science Fiction
    'cold blood', 'band',                                                 # Western
    'indian', 'entity', 'actor',                                          # investigator overrules
    'new film', 'new book', 'orientation', 'screen', 'saving', 'antic',   # second-draw review, Oct 15 2026:
    'operation', 'wood', 'gas', 'footage', 'director',                   #   meta or ambiguous
    'rock', 'music', 'seal', 'shell', 'iron', 'con', 'skin',              # third-draw review, Oct 15 2026: ambiguous
    'performance', 'replacement', 'superior', 'ray', 'humor', 'old west',  # fourth-draw review, Oct 15 2026: ambiguous, not an element, or the genre's own name
}

# Display text shown to participants. The pipeline stores lemmatized, lowercased
# strings; a few read badly as stimuli. Only capitalization of proper nouns,
# standard hyphenation, a possessive for the 'X death' phrases, a plural where
# the dictionary singular is unnatural, and US spelling are changed. No content
# word changes. `element` keeps the pipeline string, so every statistic,
# including the SUBTLEX-US join, is unaffected. Investigator decision,
# October 19, 2026 (Experiment/Study 1/wording_review_2026-10-19.md).
DISPLAY_OVERRIDES = {
    'father death': "a father's death",
    'brother death': "a brother's death",
    'fbi agent': 'FBI agent',
    'self defense': 'self-defense',
    'double cross': 'double-cross',
    'right hand man': 'right-hand man',
    'christmas': 'Christmas',
    'christmas eve': 'Christmas Eve',
    'real killer': 'the real killer',
    'north pole': 'North Pole',
    'stair': 'stairs',
    'mysterious circumstance': 'mysterious circumstances',
    'theatre': 'theater',
    'civil war': 'the Civil War',
    'texas ranger': 'Texas Ranger',
    'troop': 'troops',
    'boot': 'boots',
}

# Tercile cut points on within-genre percentile ranks. HIGH is the top tercile,
# LOW the bottom tercile; the middle tercile is never sampled.
HIGH_CUT = 2.0 / 3.0
LOW_CUT  = 1.0 / 3.0

CELLS = ['high_prev_high_dist', 'high_prev_low_dist',
         'low_prev_high_dist',  'low_prev_low_dist']
CELL_CODES = {'high_prev_high_dist': 'HH', 'high_prev_low_dist': 'HL',
              'low_prev_high_dist':  'LH', 'low_prev_low_dist':  'LL',
              'foil': 'FOIL'}

# Map from the 03b taxonomy to the SA_GEN1 experiment's category keys
# (Experiment/Archive/SA_GEN1 Feb 2026 draft/experiment.js, CATEGORIES). Editable.
TAXONOMY_TO_CATEGORY = {
    'setting':        'settings',
    'character_type': 'characters',
    'relationship':   'characters',
    'object':         'objects',
    'event':          'events',
    'theme':          'themes',
}

EXPECTED_NARRATIVE_BEFORE_BLOCKLIST = 17979  # after the Oct 2026 instance-filter correction (was 28066)   # Notebook 03b count
EXPECTED_NARRATIVE_AFTER_BLOCKLIST  = 28049   # the paper's count (Notebook 09)
""")

# ----------------------------------------------------------------------
code(r"""
# Setup, path auto-detection (Colab or local), output directory
import os, re, json, ast, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Mount Google Drive if running in Colab; harmless otherwise.
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
except Exception:
    pass

# Find v4_rebuild whether running in Colab, locally from Drive, or from notebooks/.
_CANDS = [
    Path('/content/drive/MyDrive/DGK Lab/Reasoning Lab/'
         'DGK Lab - Scene A Faire Computational/Analysis/v4_rebuild'),
    Path('/Users/USER/Library/CloudStorage/GoogleDrive-ACCOUNT/'
         'My Drive/DGK Lab/Reasoning Lab/'
         'DGK Lab - Scene A Faire Computational/Analysis/v4_rebuild'),
    Path.cwd() / 'v4_rebuild',
    Path.cwd().parent / 'v4_rebuild',
]
V4 = next((c for c in _CANDS if c.exists()), None)
assert V4 is not None, 'Could not locate v4_rebuild; set V4 manually.'
print('V4 =', V4)

# Inputs (READ ONLY)
PATH_CLASS = V4 / '03b_classification' / 'stockiness_long_classified.parquet'
PATH_NB09  = V4.parent / 'notebooks' / '09_deliverables.ipynb'

# Output (NEW directory)
OUT = V4 / '10_human_study_stimuli'
OUT.mkdir(parents=True, exist_ok=True)
print('OUT =', OUT)

def genre_slug(g):
    return re.sub(r'[^a-z0-9]+', '_', g.lower()).strip('_')

def genre_code(g):
    return genre_slug(g).replace('_', '')[:3].upper()

def genre_rng(g):
    # Per-genre generator seeded from SEED and the genre name, so adding or
    # reordering genres does not change another genre's draw.
    h = int.from_bytes(hashlib.sha256(g.encode('utf-8')).digest()[:4], 'little')
    return np.random.default_rng([SEED, h])

print({g: (genre_slug(g), genre_code(g)) for g in GENRES})
""")

# ----------------------------------------------------------------------
md(r"""
## Load the classified table and apply the franchise blocklist

Notebook 03b classifies 28,066 unique (element, type) pairs as narrative. Notebook 09 then applies a franchise blocklist that forces a short list of confirmed franchise leaks (stooge, panther, coyote, enterprise, and so on) to non-narrative with taxonomy `character_name`, which gives the 28,049 narrative pairs the paper reports. This notebook reproduces that filter exactly. The blocklist is hardcoded below so the notebook runs on its own, and when `09_deliverables.ipynb` is reachable the set literal is parsed out of its setup cell and compared, so any drift between the two notebooks is caught at run time rather than discovered later.
""")
code(r"""
stk = pd.read_parquet(PATH_CLASS)
print(f'classified table: {stk.shape[0]:,} rows x {stk.shape[1]} cols')
if not (stk.shape[0] == 456159): print('WARNING (post filter fix, counts changed): ' + f'unexpected row count {stk.shape[0]:,}')

# Franchise blocklist, copied verbatim from the Notebook 09 setup cell
# (post-canonicalization surface forms, one rationale per entry there).
FRANCHISE_BLOCKLIST = {
    # Three Stooges (Comedy)
    'stooge', 'stooges',
    # Looney Tunes / MGM / Tex Avery cartoons (Animation / Family / Comedy)
    'panther', 'coyote', 'road runner', 'roadrunner', 'tasmanian devil',
    'wabbit', 'daffy', 'felix the cat', 'speedy gonzales', 'mickey mouse',
    'donald duck',
    # Star Trek (Science Fiction)
    'enterprise', 'enterprise crew', 'shuttlecraft',
    # Marvel / DC (Action)
    'spider man', 'spider-man', 'spiderman', 'batman', 'superman',
    'wonder woman',
    # Bond / Indy / Terminator / LOTR (Action / Adventure / Fantasy)
    'james bond', 'indiana jones', 'terminator', 'hobbit',
}

# Cross-check against the live definition in Notebook 09 when it is reachable.
def read_nb09_blocklist(path):
    if not path.exists():
        return None, 'notebook 09 not found at ' + str(path)
    import nbformat
    nb09 = nbformat.read(str(path), as_version=4)
    for c in nb09.cells:
        if c.cell_type != 'code' or 'FRANCHISE_BLOCKLIST' not in c.source:
            continue
        m = re.search(r'FRANCHISE_BLOCKLIST\s*=\s*(\{.*?\n\})', c.source, re.S)
        if m:
            return set(ast.literal_eval(m.group(1))), 'parsed from notebook 09'
    return None, 'no FRANCHISE_BLOCKLIST literal found in notebook 09'

nb09_bl, nb09_status = read_nb09_blocklist(PATH_NB09)
print('blocklist cross-check:', nb09_status)
if nb09_bl is not None:
    if nb09_bl == FRANCHISE_BLOCKLIST:
        print(f'  identical to the hardcoded set ({len(FRANCHISE_BLOCKLIST)} entries)')
    else:
        print('  WARNING: hardcoded blocklist differs from notebook 09.')
        print('    only here :', sorted(FRANCHISE_BLOCKLIST - nb09_bl))
        print('    only in 09:', sorted(nb09_bl - FRANCHISE_BLOCKLIST))

# Count unique narrative (element, type) pairs before the blocklist.
pairs = stk[['element', 'element_type', 'is_narrative_element']].drop_duplicates(
    ['element', 'element_type'])
n_narr_before = int(pairs['is_narrative_element'].sum())

# Apply exactly as Notebook 09 does: force the flagged rows to non-narrative
# and taxonomy 'character_name'.
_bl = {s.lower() for s in FRANCHISE_BLOCKLIST}
_mask = stk['element'].astype(str).str.lower().isin(_bl)
stk.loc[_mask, 'is_narrative_element'] = False
stk.loc[_mask, 'element_taxonomy']     = 'character_name'
n_rows_flipped = int(_mask.sum())

pairs = stk[['element', 'element_type', 'is_narrative_element']].drop_duplicates(
    ['element', 'element_type'])
n_narr_after = int(pairs['is_narrative_element'].sum())

print(f'unique (element, type) pairs            : {len(pairs):,}')
print(f'narrative pairs before blocklist        : {n_narr_before:,} '
      f'(expected {EXPECTED_NARRATIVE_BEFORE_BLOCKLIST:,})')
print(f'narrative pairs after blocklist         : {n_narr_after:,} '
      f'(expected {EXPECTED_NARRATIVE_AFTER_BLOCKLIST:,})')
print(f'per-cell rows forced to non-narrative   : {n_rows_flipped:,}')
if n_narr_after != EXPECTED_NARRATIVE_AFTER_BLOCKLIST:
    print('NOTE: the post-blocklist count differs from the paper. Either 03b was '
          'rerun or the blocklist changed; reconcile before using these stimuli.')

# Films per genre, recovered from film_count / rel_freq_in_genre (the table
# does not carry N(g) directly). The ratio is constant within a genre up to
# float rounding; the mode is taken and the spread is checked.
_ratio = (stk['film_count'] / stk['rel_freq_in_genre']).round()
FILMS_IN_GENRE = _ratio.groupby(stk['genre']).agg(lambda s: int(s.mode().iloc[0])).to_dict()
_spread = _ratio.groupby(stk['genre']).agg(lambda s: s.max() - s.min())
assert (_spread <= 1).all(), 'film_count / rel_freq_in_genre is not constant within genre'
print('\nfilms per genre (recovered):')
for g in GENRES:
    print(f'  {g:<16} {FILMS_IN_GENRE[g]:>7,}')
""")

# ----------------------------------------------------------------------
md(r"""
## Eligible pool per genre

A candidate for a genre is a per-cell row that is narrative after the blocklist, whose element type is in `ALLOWED_TYPES`, and whose film count in that genre is at least `MIN_FILMS_IN_GENRE`. The floor is the reliability threshold from the September 2026 audit: below about twenty films the per-genre rate is too noisy to place an element in a tercile with any confidence. Within each genre, element strings that occur under more than one allowed type are collapsed to the row with the higher ensemble so that a string enters the pool once. In the current data no noun and chunk share a string in these pools, so the collapse is a safeguard rather than an active step.
""")
code(r"""
elig = stk[
    stk['is_narrative_element']
    & stk['element_type'].isin(ALLOWED_TYPES)
    & stk['genre'].isin(GENRES)
    & (stk['film_count'] >= MIN_FILMS_IN_GENRE)
].copy()
elig['key'] = elig['element'].astype(str).str.lower().str.strip()

_excl = {s.lower().strip() for s in MANUAL_EXCLUDE}
n_manual_excluded = int(elig['key'].isin(_excl).sum())
# Excluded strings stay in the pool so that percentiles, terciles and the
# seeded queues are computed on the full candidate list; they are skipped at
# draw time. A rejection therefore replaces one item with the next candidate
# in seeded order, as the preregistration states, instead of reshuffling the cell.
print(f'manual exclusions (skipped at draw time) : {n_manual_excluded} pool rows '
      f'({len(_excl)} strings in MANUAL_EXCLUDE)')

n_before_collapse = len(elig)
elig = (elig.sort_values(['genre', 'key', 'ensemble'], ascending=[True, True, False])
            .drop_duplicates(['genre', 'key'])
            .reset_index(drop=True))
n_collapsed = n_before_collapse - len(elig)

print(f'eligible rows (narrative, allowed types, film_count >= {MIN_FILMS_IN_GENRE}): '
      f'{n_before_collapse:,}')
print(f'collapsed identical strings across types : {n_collapsed}')
print()
pool_by_type = (elig.groupby(['genre', 'element_type']).size()
                    .unstack(fill_value=0).reindex(GENRES))
pool_by_type['total'] = pool_by_type.sum(axis=1)
print('eligible pool per genre by element type:')
print(pool_by_type.to_string())
""")

# ----------------------------------------------------------------------
md(r"""
## Stratification on prevalence and distinctiveness

Both dimensions are converted to percentile ranks within genre on the eligible pool. Prevalence is `rel_freq_in_genre`, the share of the genre's films that contain the element. Distinctiveness is `posterior_smoothed`, the Dirichlet-smoothed share of the element's films that belong to the genre, which is the ensemble's Bayesian member. HIGH is the top tercile of the within-genre percentile (above `HIGH_CUT`) and LOW is the bottom tercile (at or below `LOW_CUT`). Crossing the two gives four cells. The middle tercile on either dimension is excluded, so the cells are separated by a gap rather than adjacent.

The cell sizes below are the candidates available before any sampling. Chunks concentrate in the low-prevalence, high-distinctiveness cell because multi-word phrases are rare but specific; the other three cells are almost entirely single nouns. The Spearman correlation between the two percentile ranks is printed per genre so the reader can see how far the two dimensions come apart in this pool.
""")
code(r"""
elig['prevalence_pct']      = elig.groupby('genre')['rel_freq_in_genre'].rank(pct=True)
elig['distinctiveness_pct'] = elig.groupby('genre')['posterior_smoothed'].rank(pct=True)

def tercile(p):
    return 'high' if p > HIGH_CUT else ('low' if p <= LOW_CUT else 'mid')

elig['prev_tercile'] = elig['prevalence_pct'].map(tercile)
elig['dist_tercile'] = elig['distinctiveness_pct'].map(tercile)
elig['cell'] = elig['prev_tercile'] + '_prev_' + elig['dist_tercile'] + '_dist'

pool = elig[elig['cell'].isin(CELLS)].copy()

print('Spearman rho between prevalence and distinctiveness percentiles (eligible pool):')
for g in GENRES:
    s = elig[elig['genre'] == g]
    rho = stats.spearmanr(s['prevalence_pct'], s['distinctiveness_pct']).statistic
    print(f'  {g:<16} rho = {rho:+.3f}   n = {len(s):,}')

cell_avail = (pool.groupby(['genre', 'cell']).size()
                  .unstack(fill_value=0).reindex(index=GENRES, columns=CELLS))
print('\ncandidates available per cell (before sampling):')
print(cell_avail.to_string())

cell_avail_type = (pool.groupby(['genre', 'cell', 'element_type']).size()
                       .unstack(fill_value=0))
print('\ncandidates available per cell by element type:')
print(cell_avail_type.to_string())

short_cells = [(g, c, int(cell_avail.loc[g, c]))
               for g in GENRES for c in CELLS if cell_avail.loc[g, c] < K_PER_CELL]
if short_cells:
    print(f'\nNOTE: these cells have fewer than K_PER_CELL = {K_PER_CELL} candidates; '
          'the notebook samples what exists:')
    for g, c, n in short_cells:
        print(f'  {g:<16} {c:<22} {n}')
else:
    print(f'\nevery cell has at least K_PER_CELL = {K_PER_CELL} candidates')
""")

# ----------------------------------------------------------------------
md(r"""
## Sampling with overlap control

Sampling is a seeded greedy draw per genre. Candidates in each cell are sorted by string, then permuted with the genre's generator, and walked in that order. A candidate is accepted unless it overlaps a string already accepted for that genre in any cell, where overlap means that one string is a whole-word contiguous sub-sequence of the other ("house" inside "haunted house", but not "art" inside "party"). Checking in both directions guarantees that no two stimuli in a genre contain one another, whichever was drawn first.

The noun and chunk balance works in three passes. The first pass draws chunks in every cell up to half of `K_PER_CELL` or as many as the cell has, whichever is smaller. The second pass fills each cell to `K_PER_CELL` with nouns. The third pass returns to chunks for any cell still short, which only happens when a cell has fewer nouns than it needs. Drawing chunks before nouns is what makes the overlap rule drop "house" when "haunted house" is in the set rather than the reverse. In the current data only the low-prevalence, high-distinctiveness cell has enough chunks for a real balance; the other cells are nouns because that is what the pool contains, and the report records the resulting mix.
""")
code(r"""
def overlaps(a, b):
    # Whole-word containment in either direction. Identical strings overlap.
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    n = len(short)
    return any(long_[i:i + n] == short for i in range(len(long_) - n + 1))

def draw_genre(g):
    rng = genre_rng(g)
    sub = pool[pool['genre'] == g]
    accepted = []          # list of (cell, row index into pool)
    accepted_keys = []
    rejected = []          # (cell, key, blocking key)
    per_cell = {c: 0 for c in CELLS}
    per_cell_type = {c: {'noun': 0, 'chunk': 0} for c in CELLS}

    # Build a shuffled queue per (cell, type), sorted first for determinism.
    queues = {}
    for c in CELLS:
        for t in ALLOWED_TYPES:
            idx = sub[(sub['cell'] == c) & (sub['element_type'] == t)].sort_values('key').index.to_numpy()
            queues[(c, t)] = list(rng.permutation(idx)) if len(idx) else []

    def take(c, t, limit):
        q = queues[(c, t)]
        while q and per_cell[c] < limit:
            i = q.pop(0)
            k = pool.at[i, 'key']
            if k in _excl:
                rejected.append((c, k, 'MANUAL_EXCLUDE'))
                continue
            blocker = next((ak for ak in accepted_keys if overlaps(k, ak)), None)
            if blocker is not None:
                rejected.append((c, k, blocker))
                continue
            accepted.append((c, i)); accepted_keys.append(k)
            per_cell[c] += 1; per_cell_type[c][t] += 1

    chunk_target = {c: min(K_PER_CELL // 2, len(queues[(c, 'chunk')])) if 'chunk' in ALLOWED_TYPES else 0
                    for c in CELLS}
    # Pass 1: chunks up to the balance target.
    if 'chunk' in ALLOWED_TYPES:
        for c in CELLS:
            take(c, 'chunk', chunk_target[c])
    # Pass 2: nouns (and any other allowed non-chunk type) to K_PER_CELL.
    for c in CELLS:
        for t in [t for t in ALLOWED_TYPES if t != 'chunk']:
            take(c, t, K_PER_CELL)
    # Pass 3: remaining chunks for cells still short.
    if 'chunk' in ALLOWED_TYPES:
        for c in CELLS:
            take(c, 'chunk', K_PER_CELL)

    rows = []
    for c, i in accepted:
        r = pool.loc[i].to_dict()
        r['cell'] = c
        rows.append(r)
    sel = pd.DataFrame(rows)
    sel['cell'] = pd.Categorical(sel['cell'], categories=CELLS, ordered=True)
    sel = sel.sort_values(['cell', 'ensemble'], ascending=[True, False]).reset_index(drop=True)
    return sel, rejected, per_cell_type

selected = {}
overlap_log = {}
type_mix = {}
for g in GENRES:
    sel, rej, mix = draw_genre(g)
    selected[g] = sel
    overlap_log[g] = rej
    type_mix[g] = mix
    counts = sel['cell'].value_counts().reindex(CELLS).fillna(0).astype(int)
    print(f'{g:<16} ' + '  '.join(f'{CELL_CODES[c]}={counts[c]:2d}' for c in CELLS)
          + f'   overlap rejections={len(rej)}')

print('\nnoun / chunk mix per cell:')
for g in GENRES:
    print(f'  {g:<16} ' + '  '.join(
        f"{CELL_CODES[c]}={type_mix[g][c]['noun']}n+{type_mix[g][c]['chunk']}c" for c in CELLS))

print('\noverlap rejections (cell, dropped, kept):')
for g in GENRES:
    for c, k, b in overlap_log[g]:
        print(f'  {g:<16} {CELL_CODES[c]}  {k!r} blocked by {b!r}')
""")

# ----------------------------------------------------------------------
md(r"""
## Foils

A foil for a target genre is an element that is firmly expected somewhere else and rare or absent in the target. Candidates are the high-prevalence, high-distinctiveness cell of each other genre in `GENRES`, restricted to strings that are rare in the target genre under any element type, or absent from it. The rule is flat: rare means a film count in the target genre below `FOIL_MAX_FILMS_IN_TARGET`, five films by default, and that rule is applied to every target genre where it yields at least `N_FOILS` candidates. A target genre where it yields fewer falls back to a threshold that scales with genre size, the larger of the flat rule and `FOIL_MAX_RATE_IN_TARGET` times the number of films in the target genre. The fallback exists because an absolute count does not transfer across genre sizes: five films is 0.4 percent of the 1,362 Westerns but 0.03 percent of the 14,722 Comedies, and under the flat rule Comedy has no foil candidates at all, since any element that is common in some genre turns up in at least five comedies. In the current data Comedy is the only genre that uses the fallback; every other genre's foils satisfy the flat rule. The rule each genre used, its threshold, and the candidate counts under both rules are printed below and written to the report, and each foil row carries `foil_rule` so the two kinds can be told apart in analysis. The preregistration should state the flat rule and name the fallback genres with their thresholds as the report lists them.

A candidate is also rejected if it overlaps a selected stimulus of the target genre under the same whole-word rule. With `FOILS_UNIQUE_ACROSS_SET` on, a string that is already a selected stimulus for any genre, or already a foil for another target, is not reused, so no foil repeats anywhere in the set. Target genres are served in order of ascending candidate count so that the genre with the fewest candidates picks first and is not starved by the others. Within a target, foils are taken round-robin over the home genres in `GENRES` order from per-home shuffled queues, so the foils spread across the other genres rather than all coming from one.
""")
code(r"""
foil_rng = np.random.default_rng([SEED, 1])

# Max film_count of each lowercase string in each genre, over every element
# type on the full table, so "absent under noun but present as a chunk" is
# still caught.
fc_lookup = (stk.assign(key=stk['element'].astype(str).str.lower().str.strip())
                .groupby(['genre', 'key'])['film_count'].max())

used_global = set()
for g in GENRES:
    used_global.update(selected[g]['key'].tolist())

# Two thresholds per target genre. The flat rule is the primary one: a foil
# has fewer than FOIL_MAX_FILMS_IN_TARGET films in the target genre, or is
# absent from it. The fallback threshold scales with genre size and is used
# only for a target genre where the flat rule yields fewer than N_FOILS
# candidates.
FOIL_THRESHOLD_FLAT     = {g: FOIL_MAX_FILMS_IN_TARGET for g in GENRES}
FOIL_THRESHOLD_FALLBACK = {g: max(FOIL_MAX_FILMS_IN_TARGET,
                                  int(np.ceil(FOIL_MAX_RATE_IN_TARGET * FILMS_IN_GENRE[g])))
                           for g in GENRES}

# Candidate lists per target, split into two tiers: 'flat' (film_count in the
# target below the flat threshold, or absent) and 'rate_fallback' (at or above
# the flat threshold but below the fallback threshold). Built before any
# assignment so the targets can be served in ascending order of candidate
# count.
def foil_candidates(g):
    target_keys = selected[g]['key'].tolist()
    per_home = {}
    for h in GENRES:
        if h == g:
            continue
        cand = pool[(pool['genre'] == h) & (pool['cell'] == 'high_prev_high_dist')].sort_values('key')
        tiers = {'flat': [], 'rate_fallback': []}
        for i, k in zip(cand.index, cand['key']):
            if k in used_global:          # selected stimulus somewhere (incl. g)
                continue
            fc = int(fc_lookup.get((g, k), 0))
            if fc >= FOIL_THRESHOLD_FALLBACK[g]:
                continue
            if any(overlaps(k, tk) for tk in target_keys):
                continue
            tiers['flat' if fc < FOIL_THRESHOLD_FLAT[g] else 'rate_fallback'].append(i)
        per_home[h] = tiers
    return per_home

cand_lists = {g: foil_candidates(g) for g in GENRES}
n_cand_flat     = {g: sum(len(v['flat']) for v in cand_lists[g].values()) for g in GENRES}
n_cand_fallback = {g: sum(len(v['rate_fallback']) for v in cand_lists[g].values()) for g in GENRES}

# Rule per target genre, decided from the candidate counts before any draw:
# flat wherever it can supply N_FOILS, otherwise the rate fallback.
FOIL_RULE = {g: ('flat' if n_cand_flat[g] >= N_FOILS else 'rate_fallback') for g in GENRES}
FOIL_THRESHOLD = {g: (FOIL_THRESHOLD_FLAT[g] if FOIL_RULE[g] == 'flat' else FOIL_THRESHOLD_FALLBACK[g])
                  for g in GENRES}
n_cand = {g: n_cand_flat[g] + (n_cand_fallback[g] if FOIL_RULE[g] == 'rate_fallback' else 0)
          for g in GENRES}
serve_order = sorted(GENRES, key=lambda g: (n_cand[g], GENRES.index(g)))

print('foil rule per target genre (film_count in target must be below the threshold):')
for g in GENRES:
    print(f'  {g:<16} rule={FOIL_RULE[g]:<13} threshold < {FOIL_THRESHOLD[g]:3d} films of {FILMS_IN_GENRE[g]:,}  '
          f'candidates: flat={n_cand_flat[g]:3d}  fallback tier={n_cand_fallback[g]:3d} '
          f'(fallback threshold < {FOIL_THRESHOLD_FALLBACK[g]})')
print('serve order (ascending candidates under the rule used):', serve_order)

# Draw. Each target first walks its flat tier round-robin over home genres.
# Only if that leaves it short of N_FOILS (its rule is the fallback, or the
# uniqueness constraint consumed flat candidates it was counting on) does it
# continue into the fallback tier, and the rule actually used is recorded.
# Both tiers are permuted up front so the flat draw does not depend on
# whether the fallback is reached.
foils = {}
foil_used = set()
FOIL_RULE_USED = {}
for g in serve_order:
    queues = {tier: [(h, list(foil_rng.permutation(np.array(cand_lists[g][h][tier], dtype=np.int64)))
                          if cand_lists[g][h][tier] else [])
                     for h in GENRES if h != g]
              for tier in ('flat', 'rate_fallback')}
    chosen = []
    chosen_keys = set()
    for tier in ('flat', 'rate_fallback'):
        if len(chosen) >= N_FOILS:
            break
        tq = queues[tier]
        while len(chosen) < N_FOILS and any(q for _, q in tq):
            for h, q in tq:
                if len(chosen) >= N_FOILS:
                    break
                while q:
                    i = q.pop(0)
                    k = pool.at[i, 'key']
                    if k in _excl:
                        continue
                    if k in chosen_keys or (FOILS_UNIQUE_ACROSS_SET and k in foil_used):
                        continue
                    chosen.append((h, i, tier)); chosen_keys.add(k)
                    if FOILS_UNIQUE_ACROSS_SET:
                        foil_used.add(k)
                    break
    FOIL_RULE_USED[g] = 'rate_fallback' if any(t == 'rate_fallback' for _, _, t in chosen) else 'flat'
    if FOIL_RULE_USED[g] != FOIL_RULE[g]:
        FOIL_THRESHOLD[g] = (FOIL_THRESHOLD_FLAT[g] if FOIL_RULE_USED[g] == 'flat'
                             else FOIL_THRESHOLD_FALLBACK[g])

    rows = []
    for h, i, tier in chosen:
        r = pool.loc[i].to_dict()
        r['home_genre'] = h
        r['cell'] = 'foil'
        r['foil_rule'] = tier
        r['film_count_in_target_genre'] = int(fc_lookup.get((g, r['key']), 0))
        rows.append(r)
    foils[g] = pd.DataFrame(rows)

FOIL_FALLBACK_GENRES = [g for g in GENRES if FOIL_RULE_USED[g] == 'rate_fallback']

print()
for g in GENRES:
    f = foils[g]
    homes = f['home_genre'].value_counts().to_dict() if len(f) else {}
    n_fb = int((f['foil_rule'] == 'rate_fallback').sum()) if len(f) else 0
    print(f'{g:<16} foils={len(f)} ({len(f) - n_fb} flat + {n_fb} fallback)  rule used={FOIL_RULE_USED[g]}  homes={homes}')
    if len(f) < N_FOILS:
        print(f'  NOTE: only {len(f)} foil candidates satisfied the rules for {g}')
print('\ngenres that used the rate fallback:', FOIL_FALLBACK_GENRES if FOIL_FALLBACK_GENRES else 'none')

# Every foil must sit below the threshold of the rule its genre used, and in a
# flat-rule genre every foil must satisfy the flat rule.
for g in GENRES:
    f = foils[g]
    if not len(f):
        continue
    assert (f['film_count_in_target_genre'] < FOIL_THRESHOLD[g]).all(), f'foil above threshold in {g}'
    if FOIL_RULE_USED[g] == 'flat':
        assert (f['film_count_in_target_genre'] < FOIL_MAX_FILMS_IN_TARGET).all(), f'flat rule violated in {g}'

""")

# ----------------------------------------------------------------------
md(r"""
## Assemble the long table and the optional SUBTLEX join

The long table has one row per stimulus per target genre. For the four stratified cells the statistics columns (`film_count`, `rel_freq_in_genre`, `posterior_smoothed`, `distinctiveness`, `ensemble`, and the two percentiles) describe the element in the target genre, and `home_genre` equals `genre`. For foils the same columns describe the element in its home genre, where it was selected, `film_count_in_target_genre` carries its count in the target genre, zero when absent, and `foil_rule` records whether the foil satisfied the flat rule or came from the rate fallback tier; `foil_rule` is empty for stratified stimuli. `display_text` is what participants see: the pipeline string with `DISPLAY_OVERRIDES` applied (capitals for proper nouns, standard hyphens, a possessive for the two 'X death' phrases, natural plurals, US spelling). No content word changes, and `element` keeps the pipeline string, so all statistics and the SUBTLEX-US join are unaffected.

If `SUBTLEX_PATH` points at a SUBTLEX-US file, `Lg10WF` and `Lg10CD` are joined on the lowercase word. For a multi-word chunk the head noun, taken as the last token, is used, and `subtlex_token` records which token was matched so the choice is auditable. If the path is unset or missing, the columns are left empty and the report says so.
""")
code(r"""
STAT_COLS = ['element', 'element_type', 'element_taxonomy', 'key',
             'film_count', 'total_film_count', 'instance_count',
             'rel_freq_in_genre', 'distinctiveness', 'posterior_smoothed',
             'ensemble', 'prevalence_pct', 'distinctiveness_pct',
             'prev_tercile', 'dist_tercile']

frames = []
for g in GENRES:
    s = selected[g].copy()
    s['genre'] = g
    s['home_genre'] = g
    s['is_foil'] = False
    s['film_count_in_target_genre'] = s['film_count'].astype(int)
    s['foil_rule'] = None
    s['cell'] = s['cell'].astype(str)
    frames.append(s[['genre', 'cell', 'is_foil', 'home_genre', 'film_count_in_target_genre', 'foil_rule'] + STAT_COLS])
    f = foils[g].copy()
    if len(f):
        f['genre'] = g
        f['is_foil'] = True
        frames.append(f[['genre', 'cell', 'is_foil', 'home_genre', 'film_count_in_target_genre', 'foil_rule'] + STAT_COLS])

long = pd.concat(frames, ignore_index=True)
long['foil_rule'] = long['foil_rule'].astype(object).where(long['is_foil'], None)
long['cell_code'] = long['cell'].map(CELL_CODES)
long['category_key'] = long['element_taxonomy'].map(TAXONOMY_TO_CATEGORY)
long['display_text'] = long['element'].astype(str).map(lambda e: DISPLAY_OVERRIDES.get(e, e))

# Stable item ids: <GENRECODE>_<CELLCODE>_<nn>, numbered in table order.
long['item_n'] = long.groupby(['genre', 'cell_code']).cumcount() + 1
long['item_id'] = [f'{genre_code(g)}_{c}_{n:02d}'
                   for g, c, n in zip(long['genre'], long['cell_code'], long['item_n'])]
long = long.drop(columns='item_n')

# Optional SUBTLEX-US join on the head token.
if SUBTLEX_PATH is not None and not Path(SUBTLEX_PATH).is_absolute():
    SUBTLEX_PATH = str(V4.parent.parent / SUBTLEX_PATH)   # project root = parent of Analysis/
long['subtlex_token'] = long['key'].str.split().str[-1]
long['Lg10WF'] = np.nan
long['Lg10CD'] = np.nan
long['subtlex_matched'] = False
if SUBTLEX_PATH is None:
    SUBTLEX_STATUS = 'skipped: SUBTLEX_PATH is None'
elif not Path(SUBTLEX_PATH).exists():
    SUBTLEX_STATUS = f'skipped: file not found at {SUBTLEX_PATH}'
else:
    sub = pd.read_csv(SUBTLEX_PATH, sep=None, engine='python')
    wcol = next((c for c in sub.columns if c.lower() == 'word'), sub.columns[0])
    sub = sub[[wcol, 'Lg10WF', 'Lg10CD']].rename(columns={wcol: 'subtlex_token'})
    sub['subtlex_token'] = sub['subtlex_token'].astype(str).str.lower()
    sub = sub.drop_duplicates('subtlex_token')
    SUBTLEX_TABLE = sub.copy()   # kept under its own name; `sub` is reused later
    long = long.drop(columns=['Lg10WF', 'Lg10CD']).merge(sub, on='subtlex_token', how='left')
    long['subtlex_matched'] = long['Lg10WF'].notna()
    SUBTLEX_STATUS = (f'joined from {SUBTLEX_PATH}: '
                      f'{int(long["subtlex_matched"].sum())} of {len(long)} items matched on the head token')
print('SUBTLEX:', SUBTLEX_STATUS)

COL_ORDER = ['item_id', 'genre', 'cell', 'cell_code', 'is_foil', 'home_genre',
             'display_text', 'element', 'element_type', 'element_taxonomy', 'category_key',
             'film_count', 'film_count_in_target_genre', 'foil_rule', 'total_film_count', 'instance_count',
             'rel_freq_in_genre', 'distinctiveness', 'posterior_smoothed', 'ensemble',
             'prevalence_pct', 'distinctiveness_pct', 'prev_tercile', 'dist_tercile',
             'subtlex_token', 'Lg10WF', 'Lg10CD', 'subtlex_matched', 'key']
long = long[COL_ORDER].reset_index(drop=True)

print(f'\nlong table: {len(long)} rows')
print(long.groupby(['genre', 'cell_code']).size().unstack(fill_value=0).reindex(GENRES).to_string())

# Integrity checks. A string never appears twice within a genre. Foils never
# coincide with a stimulus and, with FOILS_UNIQUE_ACROSS_SET, never repeat.
# Stratified stimuli may recur across genres: a low-distinctiveness element
# such as 'father' is common in several genres and can be drawn in each, which
# is a feature of the design (same element, different genre context), so those
# repeats are reported rather than forbidden.
assert not long.duplicated(['genre', 'key']).any(), 'duplicate string within a genre'
_stim_keys = set(long.loc[~long['is_foil'], 'key'])
_foil_keys = long.loc[long['is_foil'], 'key']
assert not _foil_keys.isin(_stim_keys).any(), 'a foil coincides with a stimulus'
if FOILS_UNIQUE_ACROSS_SET:
    assert _foil_keys.is_unique, 'a foil string is used for more than one target genre'
# Foil rule check on the assembled table: below the genre's threshold always,
# and below the flat rule wherever the genre did not use the fallback.
for g in GENRES:
    _f = long[(long['genre'] == g) & long['is_foil']]
    assert (_f['film_count_in_target_genre'] < FOIL_THRESHOLD[g]).all(), f'foil above threshold in {g}'
    if FOIL_RULE_USED[g] == 'flat':
        assert (_f['film_count_in_target_genre'] < FOIL_MAX_FILMS_IN_TARGET).all(), f'flat rule violated in {g}'
    assert (_f['foil_rule'].isin(['flat', 'rate_fallback'])).all()
assert long.loc[~long['is_foil'], 'foil_rule'].isna().all()

_rep = (long[~long['is_foil']].groupby('key')['genre'].agg(list))
cross_genre_repeats = {k: v for k, v in _rep.items() if len(v) > 1}
print(f'\nstimulus strings drawn in more than one genre: {len(cross_genre_repeats)}')
for k, gs in sorted(cross_genre_repeats.items()):
    cells_k = long[(long['key'] == k) & (~long['is_foil'])].set_index('genre')['cell_code'].to_dict()
    print(f'  {k!r}: ' + ', '.join(f'{g} ({cells_k[g]})' for g in gs))
""")

# ----------------------------------------------------------------------
md(r"""
## Write the outputs

Five kinds of file go to `v4_rebuild/10_human_study_stimuli/`. The long table is written as parquet and csv. One JSON per genre carries the items in the shape the experiment reads. The review sheet is a csv with the four cells and the foils interleaved round-robin within each genre, sorted by genre, and three empty columns for the reviewer. The report is a short text file with the cell counts, the type mix, the foil sources, the overlap rejections and the SUBTLEX status.
""")
code(r"""
written = []

# 1. Long table.
p = OUT / 'stimuli_long.parquet'; long.to_parquet(p, index=False); written.append(p)
p = OUT / 'stimuli_long.csv';     long.to_csv(p, index=False, encoding='utf-8'); written.append(p)

# 2. One JSON per genre, shaped for the experiment.
def to_py(v):
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):    return bool(v)
    if isinstance(v, float) and np.isnan(v): return None
    return v

JSON_FIELDS = ['item_id', 'display_text', 'cell', 'cell_code', 'is_foil', 'home_genre',
               'element_type', 'element_taxonomy', 'category_key',
               'film_count', 'film_count_in_target_genre', 'foil_rule',
               'prevalence_pct', 'distinctiveness_pct', 'ensemble',
               'subtlex_token', 'Lg10WF', 'Lg10CD']
for g in GENRES:
    sub = long[long['genre'] == g]
    items = []
    for _, r in sub.iterrows():
        d = {'id': r['item_id'], 'text': r['display_text'], 'element_string': r['element']}
        d.update({k: to_py(r[k]) for k in JSON_FIELDS if k not in ('item_id', 'display_text')})
        items.append(d)
    payload = {
        'genre': g,
        'genre_key': genre_slug(g),
        'seed': SEED,
        'min_films_in_genre': MIN_FILMS_IN_GENRE,
        'k_per_cell': K_PER_CELL,
        'n_foils': N_FOILS,
        'foil_rule': FOIL_RULE_USED[g],
        'foil_max_films_in_target': int(FOIL_THRESHOLD[g]),
        'allowed_types': ALLOWED_TYPES,
        'cells': CELLS,
        'n_items': len(items),
        'display_text_note': ('text is what participants see: the pipeline element string, with '
                              'DISPLAY_OVERRIDES applied (capitals, hyphens, possessives, natural plurals, '
                              'US spelling; no content word changed). element_string keeps the pipeline string.'),
        'items': items,
    }
    p = OUT / f'stimuli_{genre_slug(g)}.json'
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    written.append(p)

# 3. Hand-verification sheet: cells and foils interleaved round-robin within
#    genre, genres in alphabetical order.
GROUPS = CELLS + ['foil']
review_rows = []
for g in sorted(GENRES):
    sub = long[long['genre'] == g]
    lists = [sub[sub['cell'] == c].to_dict('records') for c in GROUPS]
    while any(lists):
        for lst in lists:
            if lst:
                review_rows.append(lst.pop(0))
review = pd.DataFrame(review_rows)
review = review[['genre', 'item_id', 'display_text', 'element_type', 'element_taxonomy',
                 'cell_code', 'is_foil', 'home_genre', 'film_count',
                 'film_count_in_target_genre', 'prevalence_pct', 'distinctiveness_pct',
                 'ensemble']].copy()
review['prevalence_pct']      = review['prevalence_pct'].round(3)
review['distinctiveness_pct'] = review['distinctiveness_pct'].round(3)
review['ensemble']            = review['ensemble'].round(3)
review['narrative_ok'] = ''   # reviewer: y / n
review['phrasing']     = ''   # reviewer: natural-language phrasing for participants
review['notes']        = ''   # reviewer: free text
p = OUT / 'stimuli_review_sheet.csv'; review.to_csv(p, index=False, encoding='utf-8'); written.append(p)

# 4. Frozen predictors for the whole eligible pool (the preregistered analysis
#    set for the generation outcome). Percentile ranks are the ones the
#    stratification used, computed on the full pool including MANUAL_EXCLUDE
#    strings; excluded strings are flagged and left out of the analysis set.
pool_out = elig.copy()
pool_out['ensemble_pct'] = pool_out.groupby('genre')['ensemble'].rank(pct=True)
pool_out['manual_exclude'] = pool_out['key'].isin(_excl)
pool_out['in_generation_analysis_set'] = ~pool_out['manual_exclude']
_ids = long[~long['is_foil']].set_index(['genre', 'key'])['item_id']
pool_out['stimulus_item_id'] = [_ids.get((g, k), '') for g, k in zip(pool_out['genre'], pool_out['key'])]
pool_out['subtlex_token'] = pool_out['key'].str.split().str[-1]
if 'SUBTLEX_TABLE' in globals():
    # Preregistered variant rule: a head token absent from SUBTLEX-US takes the
    # value of its nearest spelling variant, tried in this fixed order: accents
    # removed (cafe for café), a final s added (outskirts), then a closed compound
    # read as its open form, whose head is the last listed word it ends with
    # (hitman -> man). The substitution is recorded in subtlex_variant.
    import unicodedata
    _cd = dict(zip(SUBTLEX_TABLE['subtlex_token'], SUBTLEX_TABLE['Lg10CD']))
    def _variant(tok):
        if tok in _cd: return tok
        t = unicodedata.normalize('NFKD', tok).encode('ascii', 'ignore').decode()
        if t in _cd: return t
        if t + 's' in _cd: return t + 's'
        for i in range(3, len(t) - 2):   # both parts at least three letters
            if t[:i] in _cd and t[i:] in _cd: return t[i:]
        return None
    pool_out['subtlex_variant'] = pool_out['subtlex_token'].map(_variant)
    pool_out['Lg10CD'] = pool_out['subtlex_variant'].map(_cd)
    pool_out['subtlex_variant'] = [v if v != t else '' for v, t in zip(pool_out['subtlex_variant'].fillna(''), pool_out['subtlex_token'])]
else:
    pool_out['Lg10CD'] = np.nan
    pool_out['subtlex_variant'] = ''
POOL_COLS = ['genre', 'element', 'element_type', 'key', 'element_taxonomy', 'film_count',
             'rel_freq_in_genre', 'posterior_smoothed', 'ensemble',
             'prevalence_pct', 'distinctiveness_pct', 'ensemble_pct', 'prev_tercile', 'dist_tercile',
             'manual_exclude', 'in_generation_analysis_set', 'stimulus_item_id', 'subtlex_token', 'subtlex_variant', 'Lg10CD']
pool_out = pool_out[POOL_COLS].sort_values(['genre', 'key']).reset_index(drop=True)
assert not pool_out.duplicated(['genre', 'key']).any(), 'duplicate (genre, key) rows in the pool export'
p = OUT / 'eligible_pool_predictors.csv'; pool_out.to_csv(p, index=False, encoding='utf-8'); written.append(p)
# 5. Coders' vocabulary: every narrative (element, type) pair after the
#    franchise blocklist, across the whole corpus, with no corpus statistics, so
#    coders map responses without seeing the predictors.
vocab = (stk[stk['is_narrative_element']][['element', 'element_type', 'element_taxonomy']]
           .drop_duplicates(['element', 'element_type'])
           .sort_values(['element', 'element_type']).reset_index(drop=True))
p = OUT / 'coding_vocabulary.csv'; vocab.to_csv(p, index=False, encoding='utf-8'); written.append(p)
print(f'coding vocabulary: {len(vocab):,} narrative (element, type) pairs')
print(f'eligible pool predictors: {len(pool_out):,} rows, '
      f'{int(pool_out["in_generation_analysis_set"].sum()):,} in the generation analysis set, '
      f'{int(pool_out["Lg10CD"].notna().sum()):,} with Lg10CD')

# 4. Report.
lines = []
def out(s=''):
    lines.append(s)
out('=' * 72)
out('NOTEBOOK 10: HUMAN-STUDY STIMULUS EXPORT')
out('=' * 72)
out(f'genres              : {GENRES}')
out(f'seed                : {SEED}')
out(f'min films in genre  : {MIN_FILMS_IN_GENRE}')
out(f'k per cell          : {K_PER_CELL}')
out(f'foils per genre     : {N_FOILS} (flat rule: film_count in target < {FOIL_MAX_FILMS_IN_TARGET} '
    f'or absent; fallback where the flat rule yields < {N_FOILS} candidates: '
    f'< max({FOIL_MAX_FILMS_IN_TARGET}, {FOIL_MAX_RATE_IN_TARGET:.1%} of films in target); '
    f'unique across set = {FOILS_UNIQUE_ACROSS_SET})')
out(f'fallback genres     : {FOIL_FALLBACK_GENRES if FOIL_FALLBACK_GENRES else "none"}')
out(f'allowed types       : {ALLOWED_TYPES}')
out(f'tercile cuts        : high > {HIGH_CUT:.3f}, low <= {LOW_CUT:.3f} (within-genre percentile)')
out()
out('BLOCKLIST')
out('-' * 72)
out(f'cross-check         : {nb09_status}')
out(f'narrative pairs     : {n_narr_before:,} before, {n_narr_after:,} after '
    f'({n_rows_flipped:,} per-cell rows forced non-narrative)')
out(f'manual exclusions   : {len(MANUAL_EXCLUDE)} strings in MANUAL_EXCLUDE '
    f'({n_manual_excluded} pool rows skipped at draw time)')
out()
out('ELIGIBLE POOL AND CELL SIZES (candidates available before sampling)')
out('-' * 72)
for g in GENRES:
    s = elig[elig['genre'] == g]
    rho = stats.spearmanr(s['prevalence_pct'], s['distinctiveness_pct']).statistic
    out(f'{g:<16} pool={len(s):4d}  ' + '  '.join(
        f'{CELL_CODES[c]}={int(cell_avail.loc[g, c]):3d}' for c in CELLS)
        + f'   rho(prev,dist)={rho:+.3f}')
if short_cells:
    out('cells short of K_PER_CELL: ' + ', '.join(f'{g}/{CELL_CODES[c]}={n}' for g, c, n in short_cells))
out()
out('SAMPLED STIMULI PER CELL (nouns + chunks) AND FOILS')
out('-' * 72)
for g in GENRES:
    out(f'{g:<16} ' + '  '.join(
        f"{CELL_CODES[c]}={type_mix[g][c]['noun'] + type_mix[g][c]['chunk']:2d} "
        f"({type_mix[g][c]['noun']}n+{type_mix[g][c]['chunk']}c)" for c in CELLS)
        + f'   foils={len(foils[g])}')
out()
out('FOILS (flat rule first; rate fallback only where the flat rule yields too few)')
out('-' * 72)
for g in GENRES:
    f = foils[g]
    homes = f['home_genre'].value_counts().to_dict() if len(f) else {}
    n_fb = int((f['foil_rule'] == 'rate_fallback').sum()) if len(f) else 0
    out(f'{g:<16} rule={FOIL_RULE_USED[g]:<13} threshold < {FOIL_THRESHOLD[g]:3d} films of {FILMS_IN_GENRE[g]:,}  '
        f'candidates: flat={n_cand_flat[g]:3d} fallback={n_cand_fallback[g]:3d}  '
        f'chosen={len(f)} ({len(f) - n_fb} flat + {n_fb} fallback)  '
        f'homes: ' + ', '.join(f'{h}={n}' for h, n in sorted(homes.items())))
out(f'genres that used the rate fallback: {FOIL_FALLBACK_GENRES if FOIL_FALLBACK_GENRES else "none"}')
for g in FOIL_FALLBACK_GENRES:
    out(f'  {g}: flat rule found {n_cand_flat[g]} candidates (< {N_FOILS} needed); '
        f'threshold raised to < {FOIL_THRESHOLD_FALLBACK[g]} films '
        f'(ceil({FOIL_MAX_RATE_IN_TARGET} x {FILMS_IN_GENRE[g]:,}))')
out(f'serve order (ascending candidates under the rule used): {serve_order}')
out()
out('STIMULUS STRINGS DRAWN IN MORE THAN ONE GENRE (allowed; reported for the design)')
out('-' * 72)
if not cross_genre_repeats:
    out('none')
for k, gs in sorted(cross_genre_repeats.items()):
    cells_k = long[(long['key'] == k) & (~long['is_foil'])].set_index('genre')['cell_code'].to_dict()
    out(f'{k!r}: ' + ', '.join(f'{g} ({cells_k[g]})' for g in gs))
out()
out('OVERLAP REJECTIONS (dropped because a whole-word sub- or super-string was already selected)')
out('-' * 72)
n_rej = sum(len(v) for v in overlap_log.values())
if n_rej == 0:
    out('none')
for g in GENRES:
    for c, k, b in overlap_log[g]:
        out(f'{g:<16} {CELL_CODES[c]}  {k!r} blocked by {b!r}')
out()
out('SUBTLEX')
out('-' * 72)
out(SUBTLEX_STATUS)
out()
out('TOTAL ITEMS')
out('-' * 72)
out(f'{len(long)} rows in stimuli_long ({int((~long["is_foil"]).sum())} stratified, '
    f'{int(long["is_foil"].sum())} foils)')
out()
out('DISPLAY TEXT')
out('-' * 72)
_dd = long[long['display_text'] != long['element']][['genre', 'element', 'display_text']]
out(f'{len(_dd)} items shown with DISPLAY_OVERRIDES applied (statistics use the pipeline string):')
for _r in _dd.itertuples(index=False):
    out(f'  {_r.genre:<16} {_r.element!r:<28} shown as {_r.display_text!r}')
p = OUT / 'stimuli_report.txt'; p.write_text('\n'.join(lines) + '\n', encoding='utf-8'); written.append(p)

print('\n'.join(lines))
print()
for p in written:
    print('wrote', p.name, f'({p.stat().st_size:,} bytes)')
""")

# ----------------------------------------------------------------------
md(r"""
## What a reviewer does with the review sheet

Open `stimuli_review_sheet.csv` in a spreadsheet. Each row is one stimulus; the items for a genre appear in an interleaved order that cycles through the four cells and the foils so that no block of the sheet is all one kind. For each row, fill `narrative_ok` with y if the string names a narrative element that a participant could recognize as a thing, person, place, event or idea in a story, and n if it is a lemmatization artifact (such as "clothe" or "apach"), a discourse word, a proper name that slipped past the classifier, or otherwise not a usable stimulus. Fill `phrasing` with the wording the participant will see when the raw string needs it (an article, a plural, or a short gloss), and leave it blank when the string can be shown as-is. Use `notes` for anything that should change the sampling rules rather than one item, such as a cell that is systematically off.

Nothing in this directory is final until that pass is done. The September 2026 audit found the Notebook 03b narrative boundary unreliable at the top of the rankings, which is exactly where the high-prevalence cells draw from, so the classification of every selected item has to be confirmed by hand before the stimuli go into the experiment. When the reviewer marks items n, rerun this notebook with those strings added to `MANUAL_EXCLUDE` in the configuration cell (or with a revised 03b output), rather than deleting rows from the files by hand, so the export stays reproducible from the seed. Excluded strings stay in the candidate list and are skipped when the seeded queue is walked, so a rejection replaces that one item with the next candidate in seeded order and leaves the rest of the cell unchanged; review passes can be iterated until no new item is rejected.

For the preregistration, the foil rule is the flat one: a foil has fewer than five films in the target genre or is absent from it. The report's FOILS section names any genre that needed the rate fallback and gives its threshold, and `foil_rule` in the long table marks every foil that came from the fallback tier, so the preregistration can state the exception by genre rather than describe a uniform rate rule.
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
OUTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '10_human_study_stimuli.ipynb')
with open(OUTPATH, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUTPATH, 'with', len(cells), 'cells')
