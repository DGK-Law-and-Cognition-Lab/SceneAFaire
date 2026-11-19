#!/usr/bin/env python3
# Builder for notebooks/15_worked_example.ipynb
# Run:  python3 notebooks/_build_15.py      (from the Analysis folder)
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Notebook 15 — The worked example on the two numbers

Added November 11, 2026 for Part VI of the law review article. It computes, for each story feature of a constructed Western dispute, the numbers a court would read: how many Westerns contain the element (prevalence), what share of the films containing it are Westerns (distinctiveness), where each falls among the Western's elements, and whether the element's tie to the genre changed over time. Counting only: no language model is called and nothing upstream is changed.

**Inputs.** `03b_classification/stockiness_long_classified.parquet` (counts, the two numbers and the narrative classification); `10_human_study_stimuli/eligible_pool_predictors.csv` (the within-genre percentile ranks that Study 1 used, so the worked example reads elements on the same scale as the study); `08_temporal/stockiness_by_bin.parquet` (five-year periods, for the odds ratio over time, computed as in Notebook 14, Part C).

**The reference set for the percentiles.** Percentiles are ranks among the Western's narrative elements that appear in at least 20 Westerns, the pool Study 1 drew its items from. Below 20 films the per-genre rate does not replicate across halves of the corpus (Notebook 14, Part B3), so an element under the floor is reported with its counts and is not ranked. Thirds of the percentile scale give the reading: top third is high, bottom third is low.

**Outputs.** `v4_rebuild/15_worked_example/worked_example.csv`, `worked_example_report.txt`, and the LaTeX table `tables/T-VI-1_worked_example_two_numbers.tex` in the manuscript folder.
""")

code(r"""
import os, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
except Exception:
    pass

_CANDS = [
    Path('/content/drive/MyDrive/DGK Lab/Reasoning Lab/'
         'DGK Lab - Scene A Faire Computational/Analysis/v4_rebuild'),
    *[p / 'Analysis' / 'v4_rebuild' for p in [Path.cwd(), *Path.cwd().parents]],   # local run: any parent folder
    Path.cwd() / 'v4_rebuild',
    Path.cwd().parent / 'v4_rebuild',
]
V4 = next((c for c in _CANDS if c.exists()), None)
assert V4 is not None, 'Could not locate v4_rebuild; set V4 manually.'
OUT = V4 / '15_worked_example'
OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLES = V4.parent.parent / 'Docs' / 'Writing' / 'manuscript' / 'Scene A Faire Computational' / 'tables'
TMP = Path(tempfile.mkdtemp())
def read_parquet(p, **kw):
    local = TMP / p.name          # Drive sometimes serves partial parquet files; read a local copy
    shutil.copy(p, local)
    return pd.read_parquet(local, **kw)

GENRE = 'Western'
MIN_FILMS = 20                     # the Study 1 pool floor
LOW_CUT, HIGH_CUT = 1 / 3, 2 / 3   # thirds of the within-genre percentile
report = []
def out(s=''):
    print(s); report.append(s)
print('V4 =', V4); print('OUT =', OUT); print('manuscript tables:', MANUSCRIPT_TABLES, MANUSCRIPT_TABLES.exists())
""")

md(r"""
## The dispute

A constructed case, no real parties or works. The author of a Western screenplay sues the producers of a later Western; the producers move to dismiss on the ground that every shared element is stock. The ten elements were chosen to span the readings the two numbers can give, plus the two ways an element can fall outside the count altogether (the narrative filter, and the 20-film floor).
""")
code(r"""
DISPUTE = [
    ('outlaw',        'noun'),
    ('saloon',        'noun'),
    ('sheriff',       'noun'),
    ('stagecoach',    'noun'),
    ('wagon train',   'chunk'),
    ('bounty hunter', 'chunk'),
    ('father',        'noun'),
    ('lawyer',        'noun'),
    ('texas',         'entity_GPE'),
    ('chair',         'noun'),
]
cls = read_parquet(V4 / '03b_classification' / 'stockiness_long_classified.parquet')
pool = pd.read_csv(V4 / '10_human_study_stimuli' / 'eligible_pool_predictors.csv')
pool_g = pool[pool['genre'] == GENRE]
W = cls[cls['genre'] == GENRE]
print(f'{GENRE}: {len(W):,} element cells, {int(W.is_narrative_element.sum()):,} narrative; Study 1 pool: {len(pool_g)} elements with {MIN_FILMS}+ films')
films_in_genre = round(float((W['film_count'] / W['rel_freq_in_genre']).median()))
print('films in the genre:', films_in_genre)
""")

md(r"""
## The two numbers for each disputed story feature

Prevalence is the share of Westerns whose summary contains the element. Distinctiveness is shown two ways: the plain share of the films containing the element that are Westerns, and the smoothed version the pipeline ranks on (`posterior_smoothed`, α = 1), which differs from the plain share only for rare elements. The percentiles are read from the Study 1 pool file, and the counts there are checked against the scored table.
""")
code(r"""
def reading(p_pct, d_pct):
    third = lambda x: 'high' if x > HIGH_CUT else ('low' if x <= LOW_CUT else 'middle')
    return f'{third(p_pct)} / {third(d_pct)}'

rows = []
for el, et in DISPUTE:
    r = W[(W['element'] == el) & (W['element_type'] == et)]
    assert len(r) == 1, (el, et, len(r))
    r = r.iloc[0]
    row = {'element': el, 'element_type': et, 'taxonomy': r.element_taxonomy, 'narrative': bool(r.is_narrative_element),
           'films_in_genre': int(r.film_count), 'films_all_genres': int(r.total_film_count),
           'prevalence': float(r.rel_freq_in_genre), 'share_westerns': float(r.film_count / r.total_film_count),
           'distinctiveness_smoothed': float(r.posterior_smoothed)}
    p = pool_g[(pool_g['element'] == el) & (pool_g['element_type'] == et)]
    if not row['narrative']:
        row['status'] = f'filtered ({r.element_taxonomy.replace("_", " ")})'
    elif r.film_count < MIN_FILMS:
        row['status'] = f'below the {MIN_FILMS}-film floor'
    else:
        assert len(p) == 1, (el, 'not in the Study 1 pool')
        p = p.iloc[0]
        assert int(p.film_count) == row['films_in_genre'] and abs(p.rel_freq_in_genre - row['prevalence']) < 1e-9
        row.update(prevalence_pct=float(p.prevalence_pct), distinctiveness_pct=float(p.distinctiveness_pct),
                   status='ranked', reading=reading(p.prevalence_pct, p.distinctiveness_pct))
    rows.append(row)
ex = pd.DataFrame(rows)
show = ex[['element', 'status', 'films_in_genre', 'films_all_genres', 'prevalence', 'share_westerns', 'distinctiveness_smoothed',
           'prevalence_pct', 'distinctiveness_pct', 'reading']].copy()
show[['prevalence', 'share_westerns', 'distinctiveness_smoothed', 'prevalence_pct', 'distinctiveness_pct']] *= 100
out(f'WORKED EXAMPLE: {GENRE} ({films_in_genre:,} films in the genre)')
out(show.round(1).to_string(index=False))
""")

md(r"""
## Did the tie change over time?

The odds ratio between element and genre in five-year periods, as in Notebook 14, Part C: (a × d) ÷ (b × c), with 0.5 added to each count, where a is Westerns with the element, b other films with it, c Westerns without it and d other films without it. It does not move when the genre grows or shrinks. Periods count only when they hold at least 30 Westerns and at least 5 films with the element. The table gives the median over the periods before 1970 and over those from 1990 on, when there are at least two such periods. It is not reported for elements that are filtered or below the 20-film floor.
""")
code(r"""
bins = read_parquet(V4 / '08_temporal' / 'stockiness_by_bin.parquet',
                    columns=['element', 'element_type', 'genre', 'bin', 'film_count', 'total_film_count', 'N_bg', 'N_bin'])
MIN_BIN_FILMS, MIN_E_FILMS, MIN_PERIODS = 30, 5, 2
def odds_by_era(el, et):
    d = bins[(bins['element'] == el) & (bins['genre'] == GENRE) & (bins['element_type'] == et)]
    d = d[(d['N_bg'] >= MIN_BIN_FILMS) & (d['total_film_count'] >= MIN_E_FILMS)]
    a = d['film_count']; b = d['total_film_count'] - a; c = d['N_bg'] - a; dd = d['N_bin'] - d['N_bg'] - b
    orat = ((a + .5) * (dd + .5)) / ((b + .5) * (c + .5))
    early, late = orat[d['bin'] < 1970], orat[d['bin'] >= 1990]
    med = lambda x: float(x.median()) if len(x) >= MIN_PERIODS else np.nan     # a median needs at least two periods
    return med(early), len(early), med(late), len(late)
era = pd.DataFrame([dict(zip(['element', 'or_before_1970', 'periods_before_1970', 'or_from_1990', 'periods_from_1990'], (el, *odds_by_era(el, et))))
                    for el, et in DISPUTE])
ex = ex.merge(era, on='element')
out(''); out('Odds ratio, element and Western (median over five-year periods)')
out(era.round(1).to_string(index=False))
ex.to_csv(OUT / 'worked_example.csv', index=False)
""")

md(r"""
## The table for Part VI

Written straight into the manuscript's `tables/` folder, so the numbers in the article are the numbers this notebook computes.
""")
code(r"""
def pct(x):  return '' if pd.isna(x) else ('$>$99' if x >= 0.995 else f'{100 * x:.0f}')
def pc1(x):  return f'{100 * x:.1f}'
def orf(x):  return '--' if pd.isna(x) else (f'{x:.0f}' if x >= 10 else f'{x:.1f}')
lines = []
for r in ex.itertuples():
    name = r.element if r.element_type != 'entity_GPE' else 'Texas'
    if r.status == 'ranked':
        read = r.reading
        cols = [name, f'{r.films_in_genre:,}', pc1(r.prevalence), pct(r.prevalence_pct), pc1(r.share_westerns), pct(r.distinctiveness_pct),
                orf(r.or_before_1970), orf(r.or_from_1990), read]
    else:
        status = 'filtered: place name' if r.status.startswith('filtered') else f'under {MIN_FILMS} Westerns'
        cols = [name, f'{r.films_in_genre:,}', pc1(r.prevalence), '--', pc1(r.share_westerns), '--', '--', '--', status]
    lines.append(' & '.join(cols) + r' \\')
tex = r'''\begin{table}[htbp]
\centering
\caption{A constructed Western dispute, read on the two numbers (%s Westerns).}
\label{tab:worked-example}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrl}
\toprule
 & & \multicolumn{2}{c}{Prevalence} & \multicolumn{2}{c}{Distinctiveness} & \multicolumn{2}{c}{Odds ratio} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Story feature & Westerns & \%% & pctile & \%% & pctile & pre-1970 & 1990-- & Reading \\
\midrule
%s
\bottomrule
\end{tabular}

\medskip
\begin{minipage}{0.95\textwidth}\footnotesize
\emph{Note:} Westerns: number of Westerns whose summary contains the feature. Prevalence: percentage of Westerns that contain it. Distinctiveness: percentage of all films containing it that are Westerns. Percentiles rank the feature among the Western's story features found in at least %d Westerns (the pool Study~1 drew from); distinctiveness is ranked on the smoothed share. Odds ratio: median over five-year periods before 1970 and from 1990 on; it needs two or more periods with at least 30 Westerns and 5 films containing the feature. Reading: thirds of each percentile, prevalence / distinctiveness. A dash marks a value not computed.
\end{minipage}
\end{table}
''' % (f'{films_in_genre:,}', '\n'.join(lines), MIN_FILMS)
(OUT / 'T-VI-1_worked_example_two_numbers.tex').write_text(tex)
if MANUSCRIPT_TABLES.exists():
    (MANUSCRIPT_TABLES / 'T-VI-1_worked_example_two_numbers.tex').write_text(tex)
    print('wrote', MANUSCRIPT_TABLES / 'T-VI-1_worked_example_two_numbers.tex')
print(tex)
(OUT / 'worked_example_report.txt').write_text('\n'.join(report) + '\n')
""")

md(r"""
## Outputs

In `v4_rebuild/15_worked_example/`: `worked_example.csv`, `worked_example_report.txt`, `T-VI-1_worked_example_two_numbers.tex` (also written to the manuscript's `tables/` folder when it is reachable).
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
OUTPATH = 'notebooks/15_worked_example.ipynb'
with open(OUTPATH, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUTPATH, 'with', len(cells), 'cells')
