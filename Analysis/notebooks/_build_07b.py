#!/usr/bin/env python3
# Builder for notebooks/07b_court_validation.ipynb
# Run:  python3 notebooks/_build_07b.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------
md(r"""
# Notebook 07b — Court-case validation, forward-enrichment rebuild

This notebook reruns the judicial-benchmark validation with two corrections to Notebook 07.

**Metric.** Notebook 07 reported precision@K and Spearman rho. Those penalize the framework for flagging high-stockiness elements that no court has happened to rule on, which is backwards: the judicial record is a sparse, opportunistic sample of disputes, not a complete catalog of every stock element. The right question runs forward. Of the elements a court actually called scenes a faire, where do they fall in the framework's stockiness distribution? If the framework is valid they should concentrate in the high-stockiness tail. That is a one-sided enrichment test against a random baseline, reported alongside coverage as a separate number.

**Coverage.** Notebook 07 matched judicial elements to the pipeline vocabulary by exact and fuzzy string match only, reaching roughly one fifth of the curated elements. This notebook adds head-noun decomposition (an "imperial palace" judicial phrase reaches the pipeline element "palace") and an optional semantic nearest-neighbor layer over the Voyage element embeddings, then falls back to corpus-wide stockiness when an element is absent from its specific genre cell.

**Read-only on Mira's data.** This notebook only reads `judicial_elements_CURATED.parquet` and the other `07_court_cases/` artifacts. Every output is written to a new directory, `v4_rebuild/07b_court_validation/`. Nothing in `07_court_cases/` is modified.
""")

# ----------------------------------------------------------------------
code(r"""
# Setup, path auto-detection (Colab or local), configuration
import os, re, json
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
PATH_JUDICIAL  = V4 / '07_court_cases'    / 'judicial_elements_CURATED.parquet'
PATH_STOCK     = V4 / '03_stockiness'     / 'stockiness_long.parquet'
PATH_CLASS     = V4 / '03b_classification'/ 'stockiness_long_classified.parquet'
PATH_ELEM_EMB  = V4 / '05_embeddings'     / 'element_embeddings.npy'
PATH_ELEM_IDX  = V4 / '05_embeddings'     / 'element_embeddings_index.parquet'

# Output (NEW directory; Mira's 07_court_cases is never written to)
OUT = V4 / '07b_court_validation'
OUT.mkdir(parents=True, exist_ok=True)
print('OUT =', OUT)

# Configuration
SEED          = 20260219          # project preregistered seed
SEM_THRESHOLD = 0.55              # min cosine for a semantic match (sensitivity reported)
MODEL_NAME    = 'voyage-4-large'  # must match Notebook 05 exactly
rng = np.random.default_rng(SEED)
""")

# ----------------------------------------------------------------------
md("## Load inputs and build the stockiness lookup")
code(r"""
judicial = pd.read_parquet(PATH_JUDICIAL)
stk      = pd.read_parquet(PATH_STOCK)
cls      = (pd.read_parquet(PATH_CLASS)[['element','is_narrative_element','element_taxonomy']]
             .drop_duplicates('element'))
CANON = sorted(stk['genre'].unique().tolist())
print(f'judicial elements (all): {len(judicial):,}')
print(f'stockiness cells       : {len(stk):,}')
print(f'genres                 : {len(CANON)}')

# Percentile of the ensemble within each (genre, element_type) cell-set.
stk['pct'] = stk.groupby(['genre','element_type'])['ensemble'].rank(pct=True)

# Best representation of an (element, genre): the element_type row with the
# highest ensemble. The random baseline below is drawn from this same table,
# so the max-over-types construction applies equally to both arms.
best = (stk.sort_values('ensemble')
           .groupby(['element','genre'], as_index=False)
           .agg(ensemble=('ensemble','last'), pct=('pct','last'),
                element_type=('element_type','last')))

# Corpus-wide best across all genres for each element (genre fallback).
corp = (best.sort_values('ensemble')
            .groupby('element', as_index=False)
            .agg(pct_corp=('pct','last'), ensemble_corp=('ensemble','last')))

VOCAB = set(stk['element'].unique())
NARR  = cls.set_index('element')['is_narrative_element']
print(f'pipeline vocabulary    : {len(VOCAB):,} unique element strings')
""")

# ----------------------------------------------------------------------
md(r"""
## Normalize judicial elements and restrict to in-scope, valid-genre rows

Rows with `kind == 'no_extraction'` are cases where the curated record marks a scenes a faire holding but no element text could be extracted; their `element` field is null. They are not real elements and are dropped here rather than matched. (Left in, a null normalizes to the string "none", which embeds close to the pipeline element "empty" and produces spurious semantic matches.) They are reported separately as an unextractable count.
""")
code(r"""
def norm(s):
    if s is None:
        return ''
    t = re.sub(r'\s+', ' ', str(s).lower().strip())
    return '' if t in {'none', 'nan', 'null'} else t

J = judicial[judicial['scope_tag'] == 'in_scope'].copy()
n_inscope = len(J)
J = J[J['kind'] != 'no_extraction'].copy()
J['je_norm'] = J['element'].map(norm)
J['gen']     = J['relevant_genre']
J = J[J['gen'].isin(CANON)].copy()
J = J[J['je_norm'].str.len() > 0].reset_index(drop=True)
N_UNEXTRACTABLE = n_inscope - len(J)
print(f'in-scope judicial rows                  : {n_inscope:,}')
print(f'excluded (no_extraction / null element) : {N_UNEXTRACTABLE:,}')
print(f'elements carried into matching          : {len(J):,} '
      f'({J["je_norm"].nunique():,} unique strings)')
print('kind breakdown:', J['kind'].value_counts().to_dict())
""")

# ----------------------------------------------------------------------
md(r"""
## Matching layers 1 and 2: exact and head-noun decomposition

Layer 1 matches the normalized judicial string against the pipeline vocabulary directly. Layer 2 takes the head noun (final content token) of any unmatched multi-word phrase and matches that, so "japanese historical setting" reaches "setting" and "imperial palace" reaches "palace". The head is normalized to the pipeline's canonical singular lemma before lookup, so "ghost pirates" reaches "pirate" rather than a rare plural surface form, and a slash-separated phrase such as "western costume/gear" is also tried at "costume". Candidate forms are tried by linguistic shape and gated on vocabulary membership, never chosen by stockiness. Head nouns are themselves common words, so this layer raises the descriptive percentile below; the genre-specificity statistic compares the same element across genres and is not inflated by it. A short stop-head list blocks matches on doctrinally empty heads such as "scene".
""")
code(r"""
STOP_HEADS = {'the','a','an','of','scene','scenes','element','elements','thing','things'}

def singular_first(w):
    # Candidate forms with the singular/lemma BEFORE the surface plural, so we
    # match the pipeline's canonical singularized vocabulary ('pirates'->'pirate').
    c = []
    if w.endswith('ies') and len(w) > 4: c.append(w[:-3] + 'y')
    if w.endswith('s') and not w.endswith('ss') and len(w) > 3: c.append(w[:-1])
    if w.endswith('es') and len(w) > 4: c.append(w[:-2])
    c.append(w)
    out = []
    for x in c:
        if x not in out:
            out.append(x)
    return out

def resolve_head(phrase):
    # Head noun of the whole phrase and of each '/'-separated part; return the
    # first singular-first candidate present in the vocabulary. Gating on vocab
    # membership prevents over-stemming; preferring the lemma fixes the plural-
    # surface bug ('ghost pirates' -> 'pirate', not a rare 'pirates' entity) and
    # the slash bug ('western costume/gear' -> 'costume'). Candidates are tried
    # by linguistic form, never by stockiness.
    heads = []
    for p in re.split(r'/', phrase):
        toks = [t for t in re.findall(r"[a-z']+", p) if t not in {'the','a','an'}]
        if toks:
            heads.append(toks[-1])
    for h in heads:
        if h in STOP_HEADS:
            continue
        for cand in singular_first(h):
            if cand in VOCAB and cand not in STOP_HEADS:
                return cand
    return None

J['m_exact'] = J['je_norm'].isin(VOCAB)
J['head']    = J['je_norm'].map(resolve_head)
J['m_head']  = (~J['m_exact']) & J['head'].notna()

J['match_element'] = np.where(J['m_exact'], J['je_norm'],
                       np.where(J['m_head'], J['head'], None))
J['match_method']  = np.where(J['m_exact'], 'exact',
                       np.where(J['m_head'], 'head_noun', None))

n = len(J)
print(f'Layer 1 exact     : {J["m_exact"].sum():4d}  ({J["m_exact"].mean()*100:4.1f}%)')
print(f'Layer 2 +head-noun: {J["m_head"].sum():4d}  '
      f'(cumulative {J["match_element"].notna().mean()*100:4.1f}%)')
print(f'still unmatched   : {J["match_element"].isna().sum():4d}  '
      f'(candidates for the semantic layer)')
""")

# ----------------------------------------------------------------------
md(r"""
## Matching layer 3: semantic nearest neighbor (optional)

For elements still unmatched, embed the judicial phrase with the same Voyage model used in Notebook 05 and take its nearest neighbor among the 40,877 element embeddings, accepting the match when cosine similarity clears `SEM_THRESHOLD`. The judicial-phrase embeddings are cached to the output directory, so a second run costs no API calls. If no Voyage key is available the notebook skips this layer and proceeds with the exact and head-noun matches, so it always runs end to end.

The key is read from the Colab secret `SAF_Voyage_API` (or the same-named environment variable). Never paste a key into a cell.
""")
code(r"""
PATH_JEMB = OUT / 'judicial_phrase_embeddings.npy'
PATH_JEMB_IDX = OUT / 'judicial_phrase_embeddings_index.parquet'

unmatched = sorted(J.loc[J['match_element'].isna(), 'je_norm'].unique().tolist())
print(f'{len(unmatched)} unique unmatched phrases to consider for semantic matching')

def get_voyage_key():
    try:
        from google.colab import userdata  # type: ignore
        k = userdata.get('SAF_Voyage_API')
        if k:
            return k
    except Exception:
        pass
    return os.environ.get('SAF_Voyage_API') or os.environ.get('VOYAGE_API_KEY')

sem_map = {}   # je_norm -> (match_element, cosine)
if len(unmatched) == 0:
    print('Nothing to match semantically.')
else:
    # Load or compute judicial-phrase embeddings (cached).
    if PATH_JEMB.exists() and PATH_JEMB_IDX.exists():
        jemb = np.load(PATH_JEMB)
        jidx = pd.read_parquet(PATH_JEMB_IDX)
        cached = jidx['phrase'].tolist()
        print(f'Loaded {len(cached)} cached judicial-phrase embeddings.')
        need = [p for p in unmatched if p not in set(cached)]
    else:
        jemb, jidx, cached, need = None, None, [], list(unmatched)

    if need:
        key = get_voyage_key()
        if not key:
            print('No SAF_Voyage_API key found; skipping the semantic layer. '
                  'Exact + head-noun matches are used.')
        else:
            import voyageai  # type: ignore
            client = voyageai.Client(api_key=key)
            new_vecs = []
            B = 64
            for i in range(0, len(need), B):
                batch = need[i:i+B]
                res = client.embed(texts=batch, model=MODEL_NAME,
                                   input_type='document', truncation=True)
                new_vecs.extend(res.embeddings)
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            new_idx  = pd.DataFrame({'phrase': need})
            if jemb is not None:
                jemb = np.vstack([jemb, new_vecs])
                jidx = pd.concat([jidx, new_idx], ignore_index=True)
            else:
                jemb, jidx = new_vecs, new_idx
            np.save(PATH_JEMB, jemb)
            jidx.to_parquet(PATH_JEMB_IDX, index=False)
            print(f'Embedded {len(need)} new phrases; cache now {len(jidx)}.')

    # Nearest neighbor against the corpus element embeddings.
    if PATH_JEMB.exists():
        jemb = np.load(PATH_JEMB)
        jidx = pd.read_parquet(PATH_JEMB_IDX)
        elem_emb = np.load(PATH_ELEM_EMB)
        elem_idx = pd.read_parquet(PATH_ELEM_IDX).sort_values('embedding_row_index')
        def l2n(M):
            d = np.linalg.norm(M, axis=1, keepdims=True)
            return M / np.where(d == 0, 1.0, d)
        en = l2n(elem_emb.astype(np.float32))
        jn = l2n(jemb.astype(np.float32))
        sims = jn @ en.T                      # (n_phrase, 40877)
        top  = sims.argmax(axis=1)
        tops = sims[np.arange(len(top)), top]
        elem_strings = elem_idx['element'].to_numpy()
        for p, ti, sc in zip(jidx['phrase'].tolist(), top, tops):
            if p in set(unmatched) and sc >= SEM_THRESHOLD:
                sem_map[p] = (elem_strings[ti], float(sc))
        print(f'Semantic layer matched {len(sem_map)} of {len(unmatched)} '
              f'unmatched phrases at cosine >= {SEM_THRESHOLD}.')

# Fold semantic matches into the match columns.
for i, r in J.iterrows():
    if r['match_element'] is None and r['je_norm'] in sem_map:
        me, sc = sem_map[r['je_norm']]
        J.at[i, 'match_element'] = me
        J.at[i, 'match_method']  = 'semantic'
        J.at[i, 'sem_cosine']    = sc

J['matched'] = J['match_element'].notna()
print(f'\nTotal matched: {J["matched"].sum():4d} of {len(J):4d} '
      f'({J["matched"].mean()*100:4.1f}%)')
""")

# ----------------------------------------------------------------------
md("## Assemble the matched table, with corpus-wide genre fallback")
code(r"""
M = J[J['matched']].merge(
        best.rename(columns={'element':'match_element','genre':'gen'})
            [['match_element','gen','ensemble','pct']],
        on=['match_element','gen'], how='left')
M = M.merge(corp.rename(columns={'element':'match_element'}),
            on='match_element', how='left')

M['fallback'] = M['pct'].isna() & M['pct_corp'].notna()
M['pct_used'] = M['pct'].where(M['pct'].notna(), M['pct_corp'])
M['is_narrative'] = M['match_element'].map(NARR).fillna(False)

scored = M['pct_used'].notna().sum()
print(f'matched rows         : {len(M)}')
print(f'  scored in-genre    : {(M["pct"].notna()).sum()}')
print(f'  scored via fallback: {M["fallback"].sum()}')
print(f'  unscored (no cell) : {M["pct_used"].isna().sum()}')

keep = ['cluster_id','caseName','gen','element','je_norm','match_element',
        'match_method','sem_cosine','is_narrative','ensemble','pct','pct_corp',
        'pct_used','fallback','passage_quote']
keep = [c for c in keep if c in M.columns]
M[keep].to_parquet(OUT / 'matched_judicial_elements.parquet', index=False)
print('wrote matched_judicial_elements.parquet')
""")

# ----------------------------------------------------------------------
md(r"""
## Forward enrichment: do judicial elements concentrate in the high-stockiness tail?

Under the null that judicial scenes a faire calls are unrelated to the framework, their stockiness percentiles are uniform on the unit interval with median 0.5. A valid framework pushes them toward 1. The test is a one-sided Wilcoxon signed-rank against 0.5, plus a Mann-Whitney comparison against a random draw from the same best-per-cell table, plus a bootstrap interval on the median.
""")
code(r"""
def enrich(label, vals):
    v = pd.Series(vals).dropna().values
    if len(v) < 5:
        return {'arm': label, 'n': int(len(v))}
    w = (stats.wilcoxon(v - 0.5, alternative='greater').pvalue
         if len(set(v)) > 1 else np.nan)
    bs = [np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(2000)]
    return {'arm': label, 'n': int(len(v)),
            'median_pct': float(np.median(v)),
            'mean_pct': float(np.mean(v)),
            'median_lo': float(np.percentile(bs, 2.5)),
            'median_hi': float(np.percentile(bs, 97.5)),
            'frac_ge_75': float(np.mean(v >= 0.75)),
            'frac_ge_90': float(np.mean(v >= 0.90)),
            'wilcoxon_p_gt_half': float(w)}

baseline = best.sample(min(5000, len(best)), random_state=SEED)['pct']

rows = [
    enrich('all_matched',        M['pct_used']),
    enrich('in_genre_only',      M.loc[~M['fallback'], 'pct_used']),
    enrich('narrative_matched',  M.loc[M['is_narrative'], 'pct_used']),
    enrich('narrative_in_genre', M.loc[M['is_narrative'] & ~M['fallback'], 'pct_used']),
    enrich('random_baseline',    baseline),
]
enr = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(enr.to_string(index=False))

# Mann-Whitney: matched (in-genre) vs random baseline, one-sided greater.
mw = stats.mannwhitneyu(M.loc[~M['fallback'], 'pct_used'].dropna(),
                        baseline.dropna(), alternative='greater')
print(f'\nMann-Whitney (in-genre matched > random baseline): U={mw.statistic:.0f}, '
      f'p={mw.pvalue:.2e}')
enr.to_parquet(OUT / 'enrichment_summary.parquet', index=False)
""")

# ----------------------------------------------------------------------
md("## Per-genre enrichment")
code(r"""
g_rows = []
for g, sub in M.groupby('gen'):
    v = sub['pct_used'].dropna().values
    if len(v) < 3:
        continue
    g_rows.append({'genre': g, 'n': int(len(v)),
                   'median_pct': float(np.median(v)),
                   'frac_ge_75': float(np.mean(v >= 0.75))})
by_genre = pd.DataFrame(g_rows).sort_values('median_pct', ascending=False)
print(by_genre.to_string(index=False))
by_genre.to_parquet(OUT / 'enrichment_by_genre.parquet', index=False)
""")

# ----------------------------------------------------------------------
md(r"""
## Genre specificity (the primary judicial statistic)

The percentile above is descriptive. Most of it can be reproduced with raw film counts, because the elements courts name are common words, and with the head-noun step alone (September 2026 audit, validation-1). The statistic that separates genre structure from frequency is genre specificity: for each matched element scored in its court-assigned genre, its percentile there minus its mean percentile in the other genres where the same pipeline element appears. Each case contributes the median of its elements' differences, and a Wilcoxon signed-rank test over cases asks whether case differences are centered above zero. Elements that appear only in the court-assigned genre have no contrast and are counted separately.

The contrast is computed for the ensemble, for the two dimensions it summarizes, prevalence P(e|g) (`rel_freq_in_genre`) and distinctiveness P(g|e) (`posterior_smoothed`), and for raw film count as the frequency benchmark. Within a genre, prevalence is film count divided by the genre's size, so the two rank elements identically and their rows coincide by construction. A paired Wilcoxon test over cases compares the distinctiveness and prevalence contrasts directly. Every percentile is taken within (genre, element type) on the best-type row that the enrichment uses.
""")
code(r"""
from scipy.stats import wilcoxon

SPEC_MEASURES = {'ensemble': 'ensemble', 'prevalence': 'rel_freq_in_genre',
                 'distinctiveness': 'posterior_smoothed', 'film_count': 'film_count'}
for name, col in SPEC_MEASURES.items():
    stk[f'p_{name}'] = stk.groupby(['genre', 'element_type'])[col].rank(pct=True)
best_rows = stk.sort_values('ensemble').groupby(['element', 'genre']).tail(1)   # the row `best` keeps
P = best_rows.set_index(['element', 'genre'])[[f'p_{n}' for n in SPEC_MEASURES]].sort_index()

def specificity(sub):
    recs, only_court_genre = [], 0
    for r in sub.itertuples(index=False):
        if (r.match_element, r.gen) not in P.index:
            continue
        own = P.loc[(r.match_element, r.gen)]
        allg = P.loc[r.match_element]
        others = allg[allg.index != r.gen]
        if others.empty:
            only_court_genre += 1
            continue
        rec = {'case': r.cluster_id if pd.notna(r.cluster_id) else r.caseName, 'caseName': r.caseName,
               'gen': r.gen, 'element': r.element, 'match_element': r.match_element,
               'match_method': r.match_method, 'n_other_genres': int(len(others))}
        for n in SPEC_MEASURES:
            rec[f'own_{n}'] = float(own[f'p_{n}'])
            rec[f'diff_{n}'] = float(own[f'p_{n}'] - others[f'p_{n}'].mean())
        recs.append(rec)
    return pd.DataFrame(recs), only_court_genre

SUBSETS = {
    'narrative_in_genre':       M[M['is_narrative'] & M['pct'].notna()],
    'exact_narrative_in_genre': M[M['is_narrative'] & M['pct'].notna() & (M['match_method'] == 'exact')],
    'all_in_genre':             M[M['pct'].notna()],
}
spec_rows, spec_elems = [], []
for sname, sub in SUBSETS.items():
    E, only = specificity(sub)
    E['subset'] = sname
    spec_elems.append(E)
    for n in SPEC_MEASURES:
        case_med = E.groupby('case')[f'diff_{n}'].median()
        nz = case_med[case_med != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float('nan')
        spec_rows.append({'subset': sname, 'measure': n, 'n_elements': int(len(E)),
                          'n_only_court_genre': int(only), 'n_cases': int(len(case_med)),
                          'median_case_diff': float(case_med.median()),
                          'mean_case_diff': float(case_med.mean()),
                          'cases_positive': int((case_med > 0).sum()),
                          'wilcoxon_p_two_sided': p})
spec = pd.DataFrame(spec_rows)
# Which dimension carries the contrast: paired over cases, distinctiveness minus prevalence.
paired = []
for E in spec_elems:
    cm = E.groupby('case')[['diff_distinctiveness', 'diff_prevalence']].median()
    d = cm['diff_distinctiveness'] - cm['diff_prevalence']
    nz = d[d != 0]
    paired.append({'subset': E['subset'].iloc[0], 'n_cases': int(len(d)), 'median_case_gap': float(d.median()),
                   'cases_distinctiveness_larger': int((d > 0).sum()),
                   'wilcoxon_p_two_sided': float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float('nan')})
paired = pd.DataFrame(paired)
paired.to_parquet(OUT / 'genre_specificity_paired.parquet', index=False)
print(paired.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
pd.concat(spec_elems, ignore_index=True).to_parquet(OUT / 'genre_specificity_by_element.parquet', index=False)
spec.to_parquet(OUT / 'genre_specificity_summary.parquet', index=False)
print(spec.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
""")

# ----------------------------------------------------------------------
md(r"""
## Baselines for the two dimensions (added November 2026)

The raw gap above, the distinctiveness contrast minus the prevalence contrast, is not a fair comparison. A percentile of distinctiveness moves much more from genre to genre than a percentile of prevalence does, because a common word is common everywhere, so the distinctiveness contrast is larger for any element whatever a court said. Three baselines put the two dimensions on a common footing.

**A. Rank of the court's genre.** For each court-named element, order the genres in which the element appears by the element's within-genre percentile, and record where the court's genre falls, from 0 (lowest) to 1 (highest). If the court's genre were unrelated to the element, the expected value would be 0.5 on every measure, so the measures can be compared directly. The cell also reports how often the court's genre is the element's top genre, against the chance rate of one in the number of genres. Across genres the two contrasts are two views of one fact, that the element is over-represented in the court's genre, so this baseline tests genre specificity and cannot by itself separate the dimensions.

**B. Matched controls within the genre.** The two dimensions separate across elements inside one genre. For each court-named element, the controls are the other elements of the same genre and element type whose prevalence percentile lies within 0.02 of its own; the comparison is the court-named element's distinctiveness percentile minus the controls' mean. The mirror comparison matches on distinctiveness and compares prevalence. Both differences are in percentile points. A third row matches on prevalence and compares the composite, which shows what the composite adds beyond frequency. Court-named elements are excluded from the control pools, narrative subsets draw narrative controls, and each case contributes the median of its elements' differences.

**C. Head-noun null.** Most matches come from reducing a court's phrase to its last noun, and a last noun is usually a common word. The null draws random multi-word elements from the same genres, reduces them with the same function, and looks up the head noun's percentiles. If court-named head nouns are no higher than this null on a dimension, that dimension's percentile is produced by the matching step.
""")
code(r"""
DIMS = ['prevalence', 'distinctiveness', 'ensemble']

def midrank(own, others):
    others = np.asarray(others, dtype=float)
    return float(((others < own).sum() + 0.5 * (others == own).sum()) / len(others))

def case_test(s, mu=0.0):
    d = s - mu
    nz = d[d != 0]
    return {'n_cases': int(len(s)), 'median_case': float(s.median()), 'mean_case': float(s.mean()),
            'cases_above': int((d > 0).sum()),
            'wilcoxon_p_two_sided': float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float('nan')}

# ---- A. Rank of the court's genre among the element's genres ----
rank_rows, case_long = [], []   # case_long: one row per (baseline, subset, measure, case), for figures
for E in spec_elems:
    sname = E['subset'].iloc[0]
    recs = []
    for r in E.itertuples(index=False):
        allg = P.loc[r.match_element]
        own, oth = allg.loc[r.gen], allg[allg.index != r.gen]
        rec = {'case': r.case, 'n_genres': int(len(allg))}
        for n in DIMS:
            rec[f'rank_{n}'] = midrank(own[f'p_{n}'], oth[f'p_{n}'].values)
            rec[f'top_{n}'] = float(own[f'p_{n}'] >= allg[f'p_{n}'].max())
        recs.append(rec)
    R = pd.DataFrame(recs)
    cm = R.groupby('case').mean(numeric_only=True)
    for n in DIMS:
        case_long.append(pd.DataFrame({'baseline': 'court_genre_rank', 'subset': sname, 'measure': n,
                                       'case': cm.index.astype(str), 'value': cm[f'rank_{n}'].values}))
        rank_rows.append({'baseline': 'court_genre_rank', 'subset': sname, 'measure': n,
                          'n_elements': int(len(R)), 'null_value': 0.5,
                          'element_mean': float(R[f'rank_{n}'].mean()),
                          'top_genre_share': float(R[f'top_{n}'].mean()),
                          'top_genre_chance': float((1.0 / R['n_genres']).mean()),
                          **case_test(cm[f'rank_{n}'], 0.5)})
    rank_rows.append({'baseline': 'court_genre_rank', 'subset': sname,
                      'measure': 'distinctiveness_minus_prevalence', 'n_elements': int(len(R)),
                      'null_value': 0.0,
                      'element_mean': float((R['rank_distinctiveness'] - R['rank_prevalence']).mean()),
                      **case_test(cm['rank_distinctiveness'] - cm['rank_prevalence'], 0.0)})
rank_tab = pd.DataFrame(rank_rows)
print(rank_tab.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

# ---- B. Matched controls within the genre ----
MATCH_WINDOW, MIN_CONTROLS = 0.02, 5
Bt = best_rows[['element', 'genre', 'element_type'] + [f'p_{n}' for n in DIMS]].copy()
Bt['is_narr'] = Bt['element'].map(NARR).fillna(False).astype(bool)
court_keys = set(zip(M['match_element'], M['gen']))
Bt['court'] = [k in court_keys for k in zip(Bt['element'], Bt['genre'])]
POOL_NARR = {k: d for k, d in Bt[Bt['is_narr'] & ~Bt['court']].groupby(['genre', 'element_type'])}
POOL_ALL  = {k: d for k, d in Bt[~Bt['court']].groupby(['genre', 'element_type'])}
TYPE_OF = best_rows.set_index(['element', 'genre'])['element_type']

def matched_controls(E, pools, match_on, compare):
    recs = []
    for r in E.itertuples(index=False):
        own = P.loc[(r.match_element, r.gen)]
        pool = pools.get((r.gen, TYPE_OF.loc[(r.match_element, r.gen)]))
        if pool is None:
            continue
        c = pool[(pool[f'p_{match_on}'] - own[f'p_{match_on}']).abs() <= MATCH_WINDOW]
        if len(c) < MIN_CONTROLS:
            continue
        recs.append({'case': r.case, 'own': float(own[f'p_{compare}']),
                     'ctrl': float(c[f'p_{compare}'].mean()), 'n_ctrl': int(len(c))})
    R = pd.DataFrame(recs)
    R['diff'] = R['own'] - R['ctrl']
    return R

MATCH_PAIRS = [('prevalence', 'distinctiveness'), ('distinctiveness', 'prevalence'), ('prevalence', 'ensemble')]
match_rows = []
for E in spec_elems:
    sname = E['subset'].iloc[0]
    pools = POOL_ALL if sname == 'all_in_genre' else POOL_NARR
    for match_on, compare in MATCH_PAIRS:
        R = matched_controls(E, pools, match_on, compare)
        cd = R.groupby('case')['diff'].median()
        case_long.append(pd.DataFrame({'baseline': 'matched_controls', 'subset': sname,
                                       'measure': f'{compare}_given_{match_on}',
                                       'case': cd.index.astype(str), 'value': cd.values}))
        match_rows.append({'baseline': 'matched_controls', 'subset': sname,
                           'measure': f'{compare}_given_{match_on}', 'n_elements': int(len(R)),
                           'null_value': 0.0, 'median_controls': float(R['n_ctrl'].median()),
                           'own_median': float(R['own'].median()), 'control_median': float(R['ctrl'].median()),
                           'element_mean': float(R['diff'].mean()),
                           **case_test(cd, 0.0)})
match_tab = pd.DataFrame(match_rows)
print()
print(match_tab.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

# ---- C. Head-noun null ----
N_HEAD_REPS = 1000
rng_head = np.random.default_rng(SEED + 1)
hn = M[(M['match_method'] == 'head_noun') & M['is_narrative'] & M['pct'].notna()]
hn_obs = hn.join(P[[f'p_{n}' for n in DIMS]], on=['match_element', 'gen'])
chunks = stk.loc[(stk['element_type'] == 'chunk') & stk['element'].str.contains(' '),
                 ['element', 'genre']].drop_duplicates()
chunks['head'] = chunks['element'].map(resolve_head)
chunks = chunks[chunks['head'].notna() & (chunks['head'] != chunks['element'])]
chunks = chunks[chunks['head'].map(NARR).fillna(False).astype(bool)]
chunks = chunks.join(P[[f'p_{n}' for n in DIMS]], on=['head', 'genre']).dropna(subset=['p_prevalence'])
by_gen = {g: d[[f'p_{n}' for n in DIMS]].to_numpy() for g, d in chunks.groupby('genre')}
gen_counts = hn['gen'].value_counts()
null_meds = np.empty((N_HEAD_REPS, len(DIMS)))
for i in range(N_HEAD_REPS):
    draw = np.vstack([by_gen[g][rng_head.integers(0, len(by_gen[g]), k)] for g, k in gen_counts.items()])
    null_meds[i] = np.median(draw, axis=0)
head_rows = []
for j, n in enumerate(DIMS):
    obs = float(hn_obs[f'p_{n}'].median())
    head_rows.append({'baseline': 'head_noun_null', 'subset': 'narrative_in_genre_head_noun', 'measure': n,
                      'n_elements': int(len(hn_obs)), 'own_median': obs,
                      'null_value': float(np.median(null_meds[:, j])),
                      'null_lo': float(np.percentile(null_meds[:, j], 2.5)),
                      'null_hi': float(np.percentile(null_meds[:, j], 97.5)),
                      'share_null_at_or_above': float((null_meds[:, j] >= obs).mean())})
head_tab = pd.DataFrame(head_rows)
print()
print(f'head-noun null: {len(chunks):,} multi-word (element, genre) cells with a narrative head noun; '
      f'{N_HEAD_REPS} draws of {int(gen_counts.sum())} matched to the court rows by genre')
print(head_tab.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

dim_base = pd.concat([rank_tab, match_tab, head_tab], ignore_index=True)
dim_base.to_parquet(OUT / 'dimension_baselines.parquet', index=False)
pd.concat(case_long, ignore_index=True).to_parquet(OUT / 'dimension_baselines_by_case.parquet', index=False)
print('\nwrote dimension_baselines.parquet and dimension_baselines_by_case.parquet')
""")

# ----------------------------------------------------------------------
md("## Coverage summary")
code(r"""
cov = pd.DataFrame([
    {'layer': 'exact',        'n': int(J['m_exact'].sum())},
    {'layer': 'head_noun',    'n': int((J['match_method'] == 'head_noun').sum())},
    {'layer': 'semantic',     'n': int((J['match_method'] == 'semantic').sum())},
    {'layer': 'unmatched',    'n': int((~J['matched']).sum())},
])
cov['pct_of_in_scope'] = cov['n'] / len(J)
print(f'in-scope judicial elements: {len(J)}')
print(cov.to_string(index=False))
cov.to_parquet(OUT / 'coverage_summary.parquet', index=False)
""")

# ----------------------------------------------------------------------
md(r"""
## Combination elements (lighter touch)

Notebook 07 also tried to validate multi-element combinations and matched only one of sixty-one pairs. The coverage problem there is the same string-matching gap compounded across every member of a combination, so a combination matches only if all of its members do. We report combination-member coverage here and leave a full combination-stockiness validation to a later pass.
""")
code(r"""
combo = J[J['kind'] == 'combination_member'].copy()
if len(combo):
    by_combo = combo.groupby('combination_id')['matched'].agg(['size','sum'])
    full = (by_combo['size'] == by_combo['sum']).sum()
    print(f'combination members          : {len(combo)}')
    print(f'distinct combinations         : {by_combo.shape[0]}')
    print(f'member match rate             : {combo["matched"].mean()*100:.1f}%')
    print(f'combinations fully matched     : {full} of {by_combo.shape[0]}')
else:
    print('No combination members in scope.')
""")

# ----------------------------------------------------------------------
md("## Write the human-readable report")
code(r"""
lines = []
def out(s=''):
    print(s); lines.append(s)

out('=' * 70)
out('NOTEBOOK 07b — COURT-CASE VALIDATION (FORWARD ENRICHMENT)')
out('=' * 70)
out(f'in-scope judicial elements (valid genre): {len(J)}')
out(f'unextractable rows excluded              : {N_UNEXTRACTABLE}')
out(f'matched total                            : {int(J["matched"].sum())} '
    f'({J["matched"].mean()*100:.1f}%)')
out('')
out('COVERAGE BY LAYER')
out('-' * 70)
for _, r in cov.iterrows():
    out(f'  {r["layer"]:<12} {int(r["n"]):>4}  ({r["pct_of_in_scope"]*100:4.1f}%)')
out('')
out('FORWARD ENRICHMENT  (null median = 0.500; valid framework pushes toward 1)')
out('-' * 70)
for _, r in enr.iterrows():
    if 'median_pct' not in r or pd.isna(r.get('median_pct')):
        out(f'  {r["arm"]:<20} n={int(r["n"])} (too few)'); continue
    out(f'  {r["arm"]:<20} n={int(r["n"]):>4}  median={r["median_pct"]:.3f} '
        f'[{r["median_lo"]:.3f},{r["median_hi"]:.3f}]  '
        f'%>=75th={r["frac_ge_75"]*100:4.1f}  %>=90th={r["frac_ge_90"]*100:4.1f}  '
        f'p={r["wilcoxon_p_gt_half"]:.1e}')
out('')
out(f'Mann-Whitney in-genre matched > random baseline: p={mw.pvalue:.2e}')
out('')
out('PER-GENRE MEDIAN PERCENTILE')
out('-' * 70)
for _, r in by_genre.iterrows():
    out(f'  {r["genre"]:<18} n={int(r["n"]):>3}  median={r["median_pct"]:.3f}  '
        f'%>=75th={r["frac_ge_75"]*100:4.1f}')
out('')
out('GENRE SPECIFICITY  (primary statistic: percentile in the court-assigned genre')
out('minus mean percentile in the element\'s other genres; median per case;')
out('Wilcoxon signed-rank over cases, two-sided)')
out('-' * 70)
for sname in SUBSETS:
    sub = spec[spec['subset'] == sname]
    r0 = sub.iloc[0]
    out(f'  {sname}: {int(r0["n_elements"])} elements in {int(r0["n_cases"])} cases '
        f'({int(r0["n_only_court_genre"])} elements appear only in the court genre)')
    for _, r in sub.iterrows():
        out(f'    {r["measure"]:<16} case median {r["median_case_diff"]:+.3f}  '
            f'{int(r["cases_positive"])}/{int(r["n_cases"])} cases > 0  p={r["wilcoxon_p_two_sided"]:.1e}')
out('  (prevalence and film_count coincide by construction: within a genre,')
out('   prevalence is film count divided by the genre size)')
out('')
out('WHICH DIMENSION CARRIES IT  (distinctiveness contrast minus prevalence')
out('contrast, median over cases; paired Wilcoxon, two-sided)')
out('-' * 70)
for _, r in paired.iterrows():
    out(f'  {r["subset"]:<26} gap {r["median_case_gap"]:+.3f}  '
        f'{int(r["cases_distinctiveness_larger"])}/{int(r["n_cases"])} cases larger  p={r["wilcoxon_p_two_sided"]:.1e}')
out('  (this raw gap is not on a common scale: distinctiveness percentiles vary')
out('   more across genres than prevalence percentiles for any element; see the')
out('   baselines below)')
out('')
out('BASELINE A: RANK OF THE COURT GENRE among the element\'s genres')
out('(0 = lowest, 1 = highest; 0.5 if the court genre were unrelated; mean per')
out('case; Wilcoxon over cases against 0.5, two-sided)')
out('-' * 70)
for sname in SUBSETS:
    sub = rank_tab[rank_tab['subset'] == sname]
    out(f'  {sname}: {int(sub.iloc[0]["n_elements"])} elements in {int(sub.iloc[0]["n_cases"])} cases')
    for _, r in sub.iterrows():
        if r['measure'] == 'distinctiveness_minus_prevalence':
            out(f'    distinctiveness minus prevalence: case median {r["median_case"]:+.3f}  '
                f'{int(r["cases_above"])}/{int(r["n_cases"])} cases > 0  p={r["wilcoxon_p_two_sided"]:.1e}')
        else:
            out(f'    {r["measure"]:<16} case median {r["median_case"]:.3f}  '
                f'{int(r["cases_above"])}/{int(r["n_cases"])} cases > 0.5  p={r["wilcoxon_p_two_sided"]:.1e}  '
                f'top genre {r["top_genre_share"]*100:4.1f}% (chance {r["top_genre_chance"]*100:.1f}%)')
out('')
out('BASELINE B: MATCHED CONTROLS within the genre')
out(f'(controls: same genre and element type, matching percentile within {MATCH_WINDOW};')
out('court-named minus control mean, in percentile points; median per case;')
out('Wilcoxon over cases against 0, two-sided)')
out('-' * 70)
LABEL = {'distinctiveness_given_prevalence': 'distinctiveness, prevalence held fixed',
         'prevalence_given_distinctiveness': 'prevalence, distinctiveness held fixed',
         'ensemble_given_prevalence':        'composite, prevalence held fixed'}
for sname in SUBSETS:
    sub = match_tab[match_tab['subset'] == sname]
    out(f'  {sname}')
    for _, r in sub.iterrows():
        out(f'    {LABEL[r["measure"]]:<40} n={int(r["n_elements"]):>3}  court {r["own_median"]:.3f} vs '
            f'controls {r["control_median"]:.3f}  case median {r["median_case"]:+.3f}  '
            f'{int(r["cases_above"])}/{int(r["n_cases"])} cases > 0  p={r["wilcoxon_p_two_sided"]:.1e}')
out('')
out('BASELINE C: HEAD-NOUN NULL  (random multi-word elements of the same genres,')
out(f'reduced to their head noun; {N_HEAD_REPS} draws; median percentile)')
out('-' * 70)
for _, r in head_tab.iterrows():
    out(f'  {r["measure"]:<16} court head-noun matches {r["own_median"]:.3f}   null {r["null_value"]:.3f} '
        f'[{r["null_lo"]:.3f}, {r["null_hi"]:.3f}]   share of null draws at or above: {r["share_null_at_or_above"]:.3f}')
out('')
out('Reading the baselines: across genres both dimensions pick out the court\'s')
out('genre well above chance, and prevalence does so at least as well as')
out('distinctiveness, so the raw gap above was a difference of scale. Within a')
out('genre, court-named elements exceed matched controls on each dimension with')
out('the other held fixed. The prevalence difference is the larger one in the')
out('full set, but the head-noun null shows that the matching step alone')
out('produces the prevalence of the head-noun matches, while their')
out('distinctiveness is above that null. On exact matches the two differences')
out('are of similar size. The case record therefore supports "high on both"')
out('and does not rank the two dimensions.')
out('')
out('Interpretation: judicially identified scenes a faire sit high in their')
out('genre\'s ranking, well above the random baseline, but that percentile is')
out('descriptive: raw film count reproduces most of it (September 2026 audit).')
out('The primary judicial statistic is genre specificity: the same elements rank')
out('higher in the genre the court assigned than in their other genres. The')
out('baselines above, not the raw gap, bear on which dimension carries it.')
out('Precision@K from Notebook 07 is not reported because it inverts the')
out('inference: the case record is a sparse sample of disputes, not a complete')
out('catalog of stock elements.')

(OUT / 'enrichment_report.txt').write_text('\n'.join(lines))
print('\nwrote', OUT / 'enrichment_report.txt')
""")

# ----------------------------------------------------------------------
md(r"""
## What this establishes

Elements that courts have held to be scenes a faire sit near the top of their genre's distribution, far from the random baseline, across most genres. That percentile is descriptive, because frequency alone reproduces most of it. The test that bears on the framework is genre specificity: the same elements rank higher in the court-assigned genre than in their other genres. The raw contrast is larger for distinctiveness, but that is a difference of scale. On the baselines added in November 2026, both dimensions pick out the court's genre well above chance, court-named elements exceed matched controls on each dimension with the other held fixed, and the head-noun null shows that the prevalence of head-noun matches is produced by the matching step while their distinctiveness is not. The case record supports high on both dimensions and does not rank them. Coverage is reported as its own number rather than folded into a precision score, so the sparsity of the case record is described rather than charged against the framework.

Outputs in `v4_rebuild/07b_court_validation/`: `matched_judicial_elements.parquet`, `enrichment_summary.parquet`, `enrichment_by_genre.parquet`, `coverage_summary.parquet`, `genre_specificity_summary.parquet`, `genre_specificity_by_element.parquet`, `genre_specificity_paired.parquet`, `dimension_baselines.parquet`, `dimension_baselines_by_case.parquet`, `enrichment_report.txt`, and the cached judicial-phrase embeddings. Mira's `07_court_cases/` directory is untouched.
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
OUTPATH = 'notebooks/07b_court_validation.ipynb'
with open(OUTPATH, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUTPATH, 'with', len(cells), 'cells')
