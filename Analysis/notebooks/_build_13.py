#!/usr/bin/env python3
# Builder for notebooks/13_event_elements.ipynb
# Run:  python3 notebooks/_build_13.py      (from the Analysis folder)
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Notebook 13 — Event features: scoring what happens, not only what is mentioned

**Why.** Notebook 02 extracts nouns, noun phrases, verbs and names one at a time. Courts usually describe scènes à faire as things that happen: "firing a machine gun", "a shootout", "escaping from prison". The pipeline has "gun" and "fire" separately but not the event. This notebook is a first pass at an event-level unit, built with a grammatical parser and counting only, with no language-model judgment.

**What an event is here.** For every verb in a plot summary, the parser gives its grammatical dependents. Three kinds of event are recorded, each as dictionary forms (lemmas):

| Kind | Rule | Example sentence | Event |
|---|---|---|---|
| verb + object | the verb's direct object, or the subject of a passive verb | "The gang robs a bank." / "A bank is robbed." | `rob bank` |
| verb + preposition + object | the object of a preposition attached to the verb | "He escapes from prison." | `escape from prison` |
| subject + verb | the verb's subject, or the agent of a passive verb | "The sheriff arrests him." | `sheriff arrest` |

Only common nouns count as subjects and objects, so names of people and places never enter. A verb's particle is kept ("blow up bridge"). The verbs *be*, *have* and *do* are skipped, and so are events about the summary itself ("the film ends", "the story begins"). An event counts once per film.

**Scores.** Exactly the two quantities of Notebook 03, at the film level: prevalence P(e|g) = films of genre g with the event ÷ films of genre g, and distinctiveness P(g|e) = (films of g with the event + α·P(g)) ÷ (films with the event + α), α = 1, with P(g) the genre's share of film-genre pairs. Events in fewer than 10 films are dropped.

**Status.** Exploratory, added November 2026 for the CogForum talk. It reads `01c_union/movies_union.parquet` and writes only to `v4_rebuild/13_event_elements/`. Nothing in Notebooks 02 to 12 depends on it, and the Study 1 predictors are unaffected. The parser is spaCy's small English model (`en_core_web_sm`); Notebook 02 used the large one, which differs in word vectors and name recognition, not in the kind of parse used here.
""")

code(r"""
import os, re, sys, shutil, tempfile, time
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
    Path('/Users/USER/Library/CloudStorage/GoogleDrive-ACCOUNT/'
         'My Drive/DGK Lab/Reasoning Lab/'
         'DGK Lab - Scene A Faire Computational/Analysis/v4_rebuild'),
    Path.cwd() / 'v4_rebuild',
    Path.cwd().parent / 'v4_rebuild',
]
V4 = next((c for c in _CANDS if c.exists()), None)
assert V4 is not None, 'Could not locate v4_rebuild; set V4 manually.'
OUT = V4 / '13_event_elements'
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path(tempfile.mkdtemp())
def read_parquet(p):
    local = TMP / p.name          # Drive sometimes serves partial parquet files; read a local copy
    shutil.copy(p, local)
    return pd.read_parquet(local)

MIN_FILMS = 10        # an event must appear in at least this many films
ALPHA     = 1.0       # smoothing, as in Notebook 03
SKIP_VERBS = {'be', 'have', 'do'}
# Words about the summary itself ("the film ends", "the story begins"), not about what happens in the story.
META_WORDS = {'film', 'movie', 'story', 'scene', 'cartoon', 'plot', 'audience', 'viewer', 'credit', 'credits',
              'narrator', 'flashback', 'episode', 'sequence', 'short', 'sequel', 'title'}
N_PROCESS = max(1, min(8, (os.cpu_count() or 2) - 1))
print('V4 =', V4); print('OUT =', OUT); print('processes =', N_PROCESS)
""")

md("## Load the films")
code(r"""
films = read_parquet(V4 / '01c_union' / 'movies_union.parquet')[['union_id', 'plot', 'genres_canonical', 'year']]
films['genres'] = films['genres_canonical'].map(lambda g: [str(x) for x in list(g)] if g is not None else [])
fg = films[['union_id', 'genres']].explode('genres').dropna().rename(columns={'genres': 'genre'})
fg = fg[fg['genre'].str.len() > 0]
N_g = fg.groupby('genre')['union_id'].nunique()
P_PRIOR = N_g / N_g.sum()                      # each genre's share of film-genre pairs
print(f'{len(films):,} films, {len(N_g)} genres, {len(fg):,} film-genre pairs')
print(N_g.sort_values(ascending=False).to_string())
""")

md(r"""
## Extract events with the parser

The parse of all summaries takes a few minutes on several processor cores. The result is cached in `event_index.parquet`; delete that file to force a fresh parse.
""")
code(r"""
PATH_INDEX = OUT / 'event_index.parquet'

def events_of(doc):
    out = set()
    for tok in doc:
        if tok.pos_ != 'VERB':
            continue
        v = tok.lemma_.lower()
        if v in SKIP_VERBS or not v.isalpha():
            continue
        prt = [c.lemma_.lower() for c in tok.children if c.dep_ == 'prt']
        if prt:
            v = v + ' ' + prt[0]
        def nouns(head):
            # the noun itself and any nouns joined to it by "and"/"or"
            res = [head] + [c for c in head.conjuncts]
            return [n.lemma_.lower() for n in res if n.pos_ == 'NOUN' and n.lemma_.isalpha() and len(n.lemma_) > 1]
        for c in tok.children:
            if c.dep_ in ('dobj', 'nsubjpass'):
                for n in nouns(c):
                    out.add((f'{v} {n}', 'verb_object'))
            elif c.dep_ == 'nsubj':
                for n in nouns(c):
                    out.add((f'{n} {v}', 'subject_verb'))
            elif c.dep_ == 'prep':
                for gc in c.children:
                    if gc.dep_ == 'pobj':
                        for n in nouns(gc):
                            out.add((f'{v} {c.lemma_.lower()} {n}', 'verb_prep_object'))
            elif c.dep_ == 'agent':
                for gc in c.children:
                    if gc.dep_ == 'pobj':
                        for n in nouns(gc):
                            out.add((f'{n} {v}', 'subject_verb'))
    return out

if PATH_INDEX.exists():
    idx = read_parquet(PATH_INDEX)
    print(f'loaded cached event index: {len(idx):,} rows')
else:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    except OSError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    nlp.max_length = 2_000_000
    t0 = time.time()
    rows = []
    texts = films['plot'].fillna('').astype(str).tolist()
    ids = films['union_id'].tolist()
    for uid, doc in zip(ids, nlp.pipe(texts, batch_size=64, n_process=N_PROCESS)):
        for ev, kind in events_of(doc):
            rows.append((uid, ev, kind))
    idx = pd.DataFrame(rows, columns=['union_id', 'event', 'event_type'])
    idx.to_parquet(PATH_INDEX, index=False)
    print(f'parsed {len(texts):,} summaries in {(time.time() - t0) / 60:.1f} min; {len(idx):,} event rows')
print(idx['event_type'].value_counts().to_string())
print(f"{idx[['event', 'event_type']].drop_duplicates().shape[0]:,} distinct events before the frequency rule")
""")

md("## Score every event in every genre")
code(r"""
is_meta = idx['event'].map(lambda e: any(w in META_WORDS for w in e.split()))
print(f'{int(is_meta.sum()):,} event rows are about the summary itself ("film end", "story begin") and are set aside')
idx_story = idx[~is_meta]
tot = idx_story.groupby(['event', 'event_type'])['union_id'].nunique().rename('total_film_count')
keep = tot[tot >= MIN_FILMS]
print(f'{len(keep):,} events appear in at least {MIN_FILMS} films')
ix = idx_story.merge(keep.reset_index(), on=['event', 'event_type'])
cells_ = (ix.merge(fg, on='union_id')
            .groupby(['event', 'event_type', 'genre'])['union_id'].nunique().rename('film_count').reset_index())
sc = cells_.merge(keep.reset_index(), on=['event', 'event_type'])
sc['films_in_genre'] = sc['genre'].map(N_g)
sc['p_prior_g'] = sc['genre'].map(P_PRIOR)
sc['prevalence'] = sc['film_count'] / sc['films_in_genre']
sc['distinctiveness_raw'] = sc['film_count'] / sc['total_film_count']
sc['distinctiveness'] = (sc['film_count'] + ALPHA * sc['p_prior_g']) / (sc['total_film_count'] + ALPHA)
sc['lift'] = sc['distinctiveness_raw'] / (sc['films_in_genre'] / len(films))    # 1 = no association with the genre
for m in ['prevalence', 'distinctiveness']:
    sc[f'p_{m}'] = sc.groupby(['genre', 'event_type'])[m].rank(pct=True)
sc.to_parquet(OUT / 'event_scores.parquet', index=False)
print(f'{len(sc):,} (event, genre) cells written to event_scores.parquet')
sc.sort_values('total_film_count', ascending=False).head(10)
""")

md(r"""
## What the events look like

For each genre, the most characteristic events (highest distinctiveness among events in at least 1% of the genre's films, with a floor of 8 films) and the most common ones (highest prevalence).
""")
code(r"""
lines = []
def out(s=''):
    print(s); lines.append(s)
out('NOTEBOOK 13 — EVENT ELEMENTS (exploratory)')
out(f'{len(films):,} films; {len(idx):,} event rows; {len(keep):,} events in at least {MIN_FILMS} films; {len(sc):,} (event, genre) cells')
out(idx['event_type'].value_counts().to_string())
tops = []
for g in sorted(N_g.index):
    d = sc[sc['genre'] == g]
    floor = max(8, int(np.ceil(0.01 * N_g[g])))
    char = d[d['film_count'] >= floor].sort_values('distinctiveness', ascending=False).head(15)
    comm = d.sort_values('prevalence', ascending=False).head(15)
    out(''); out(f'{g}  ({N_g[g]:,} films; characteristic events need {floor}+ films)')
    out('  most characteristic: ' + '; '.join(f"{r.event} ({r.film_count}/{r.total_film_count})" for r in char.itertuples()))
    out('  most common:         ' + '; '.join(f"{r.event} ({r.prevalence * 100:.1f}%)" for r in comm.itertuples()))
    tops.append(char.assign(list='characteristic')); tops.append(comm.assign(list='common'))
pd.concat(tops, ignore_index=True).to_parquet(OUT / 'top_events_by_genre.parquet', index=False)
(OUT / 'event_report.txt').write_text('\n'.join(lines))
print('\nwrote', OUT / 'event_report.txt')
""")

md(r"""
## Figures

The figures below are ordinary matplotlib plots with matplotlib's default look. Each is shown here and saved as a PNG in this notebook's `figures/` folder. The talk's slides use those PNG files as they are.
""")
code(r"""
import matplotlib.pyplot as plt
plt.rcdefaults()                                   # matplotlib's own defaults: white background, boxed axes, DejaVu Sans
plt.rcParams.update({'font.size': 10})            # figures are drawn at the size they have on the slide
BLUE, RED = 'tab:blue', 'tab:red'                  # prevalence is blue, distinctiveness is red, in every figure
FIGDIR = OUT / 'figures'
FIGDIR.mkdir(exist_ok=True)
def wilson(k, n, z=1.96):
    # 95% interval for a proportion k out of n (Wilson score interval)
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half
def mean_ci(v):
    # 95% interval for a mean, from the spread of the values (t distribution)
    from scipy import stats
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() - h, v.mean() + h
ERR = dict(ecolor='black', capsize=3, lw=1.2)        # error bars: black, thin, small caps
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / f'{name}.png', dpi=200, facecolor='white', bbox_inches='tight')
    plt.show()
""")

md(r"""
### The most characteristic events, three genres at a time

For each genre, the five most characteristic events among those in at least 1% of the genre's films. The bar is distinctiveness: of the films with the event, the percentage in this genre. The counts are above each bar (films in this genre / all films with the event), and the black line is a 95% interval for the percentage given those counts.
""")
code(r"""
top_tab = pd.concat(tops, ignore_index=True)
def event_bars(genres, name):
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.5), sharey=True)
    for ax, g in zip(axes, genres):
        d = top_tab[(top_tab['genre'] == g) & (top_tab['list'] == 'characteristic')].head(5)
        share = 100 * d['film_count'] / d['total_film_count']
        lo, hi = wilson(d['film_count'], d['total_film_count'])
        ax.bar(d['event'], share, color=RED, yerr=[share - 100 * lo, 100 * hi - share], error_kw=ERR)
        for i, r in enumerate(d.itertuples()):
            ax.text(i, 100 * hi[i] + 2, f'{r.film_count}/{r.total_film_count}', ha='center', va='bottom', fontsize=7)
        ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_title(g)
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', fontsize=9)
    axes[0].set_ylabel('% of films with the event\nthat are in this genre')
    save(fig, name)
event_bars(['Western', 'Horror', 'Science Fiction'], 'events_three_genres')
event_bars(['Crime', 'Fantasy', 'Comedy'], 'events_three_more_genres')
""")

md(r"""
### Events on both dimensions, six genres

Each dot is an event in at least 20 of the genre's films. The most common event and the most characteristic event (among those in at least 1% of the genre's films) are named.
""")
code(r"""
SIX = ['Western', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Comedy']
fig, axes = plt.subplots(2, 3, figsize=(9.1, 4.3), sharey=True)
for ax, g in zip(axes.ravel(), SIX):
    d = sc[(sc['genre'] == g) & (sc['film_count'] >= 20)]
    ax.scatter(100 * d['prevalence'], 100 * d['distinctiveness'], s=8, color='gray', alpha=0.4)
    common = d.nlargest(1, 'prevalence').iloc[0]
    cand = d[d['prevalence'] >= 0.01]
    char = (cand if len(cand) else d).nlargest(1, 'distinctiveness').iloc[0]
    for r, col in [(common, BLUE), (char, RED)]:
        ax.scatter(100 * r['prevalence'], 100 * r['distinctiveness'], s=40, color=col, zorder=3)
        ax.annotate(r['event'], (100 * r['prevalence'], 100 * r['distinctiveness']), xytext=(6, 6), textcoords='offset points', color=col)
    ax.set_xscale('log'); ax.set_title(f'{g} ({len(d):,} events)')
fig.supxlabel("Prevalence: % of the genre's films with the event (log scale)")
fig.supylabel("Distinctiveness: % of the event's films in the genre")
save(fig, 'events_both_dimensions')
""")

md(r"""
## How the two dimensions relate for events

The rank correlation between prevalence and distinctiveness within each genre, for events in at least 20 of the genre's films (the floor the human study used for single features).
""")
code(r"""
from scipy.stats import spearmanr
rows = []
for g, d in sc[sc['film_count'] >= 20].groupby('genre'):
    if len(d) >= 30:
        rows.append({'genre': g, 'n_events': len(d), 'spearman_prevalence_distinctiveness': spearmanr(d['prevalence'], d['distinctiveness'])[0]})
rel = pd.DataFrame(rows)
rel.to_parquet(OUT / 'event_dimension_correlation.parquet', index=False)
print(rel.round(3).to_string(index=False))
""")

md(r"""
## What this adds, and what it does not

This gives the pipeline a unit closer to how courts talk: something that happens, with who or what it happens to. The scores are the same two counts as before. It is a first pass. A parser makes mistakes, the events are two or three words and not whole scenes, and nothing here has been checked by hand. It is reported as exploratory.

Outputs in `v4_rebuild/13_event_elements/`: `event_index.parquet` (film, event, kind), `event_scores.parquet` (one row per event and genre), `top_events_by_genre.parquet`, `event_dimension_correlation.parquet`, `event_report.txt`, and the figures in `figures/`.
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
OUTPATH = 'notebooks/13_event_elements.ipynb'
with open(OUTPATH, 'w') as f:
    nbf.write(nb, f)
print('wrote', OUTPATH, 'with', len(cells), 'cells')
