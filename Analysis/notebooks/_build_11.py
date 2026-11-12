# Builder for notebooks/11_study1_response_coding.ipynb
# Run:  python3 notebooks/_build_11.py
# Codes the Study 1 generation responses exactly as preregistered (prereg Section 6.4).
import os
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r"""
# 11. Study 1: coding the generation responses

This notebook applies the coding pipeline of the Study 1 preregistration (Section 6.4) to the participants' free-text entries. It does not change any preregistered setting; every setting lives in `Analysis/study1_coding/study1_coding.py`. The files it uses are the ones in the GitHub release `study1-prereg` of DGK-Law-and-Cognition-Lab/SceneAFaire.

Order of work, as preregistered: (1) units and deterministic matches; (2) voyage-4-large candidate retrieval; (3) three coder models label every remaining unit, blind to genre; (4) two-of-three consensus; (5) reliability among the coders; (6) the human validation sample is exported and, once both human coders have finished, the registered decision rule picks the primary coding; (7) quality flags for exclusion rule 6, production indicators on the registered analysis set, and the genre-discrimination check S1.

API keys come from Colab Secrets only: `ANTHROPIC_API_KEY`, `SAF_OpenAI` (or `CharacterAI_OpenAI`), `SAF_Gemini` (or `CharacterAI_Gemini`), and `SAF_Voyage_API`. Every model response is cached to disk, so a rerun costs nothing and returns the same labels.
""")

code(r"""
import os, sys, json, hashlib, time
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/DGK Lab/Reasoning Lab/DGK Lab - Scene A Faire Computational')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'anthropic', 'openai', 'google-genai', 'voyageai'], check=True)
else:
    PROJECT_ROOT = Path.cwd().parents[1] if Path.cwd().name == 'notebooks' else Path.cwd()

AN = PROJECT_ROOT / 'Analysis'
CODING = AN / 'study1_coding'
V4 = AN / 'v4_rebuild'
RAW = PROJECT_ROOT / 'Data' / 'Study1_raw'          # the Pavlovia data/ folder, downloaded
OUT = V4 / '11_study1_coding'; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'cache'; CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(CODING))
import numpy as np, pandas as pd
import study1_coding as sc
print('project root:', PROJECT_ROOT)
""")

code(r"""
RUBRIC = (CODING / 'coding_rubric_v1.md').read_text()
""")

md("## Sessions and units")
code(r"""
# Pavlovia saves the session files as data_SAF_STUDY1_..., so match anywhere in the name.
frames = [pd.read_csv(p, dtype=str) for p in sorted(RAW.glob('*SAF_STUDY1_*.csv'))]
sessions = pd.concat(frames, ignore_index=True)
print(len(sessions), 'session files;', (sessions['is_test'].str.lower() == 'true').sum(), 'test sessions dropped')
units = sc.units_from_sessions(sessions)
units['norm'] = units['unit_text'].map(sc.normalize)
vocab = sc.Vocabulary.from_csv(V4 / '10_human_study_stimuli' / 'coding_vocabulary.csv')
units['det_match'] = units['norm'].map(lambda t: sc.deterministic_match(t, vocab))
print(len(units), 'units;', units['det_match'].notna().sum(), 'matched deterministically')
""")

md("## Candidate retrieval (voyage-4-large, input_type='document')")
code(r"""
def get_secret(*names):
    for n in names:
        try:
            from google.colab import userdata
            v = userdata.get(n)
            if v: return v
        except Exception:
            pass
        if os.environ.get(n): return os.environ[n]
    return None

emb_cache = CACHE / 'unit_embeddings.parquet'
todo = units[units['det_match'].isna()]
texts = sorted(set(todo['norm']) | set(units['entry_text'].map(sc.normalize)))
have = pd.read_parquet(emb_cache) if emb_cache.exists() else pd.DataFrame(columns=['text', 'vec'])
need = [t for t in texts if t and t not in set(have['text'])]
if need:
    import voyageai
    vo = voyageai.Client(api_key=get_secret('SAF_Voyage_API'))
    rows = []
    for i in range(0, len(need), 64):
        batch = need[i:i + 64]
        res = vo.embed(texts=batch, model=sc.SEMANTIC_MODEL, input_type=sc.SEMANTIC_INPUT_TYPE, truncation=True)
        rows += [{'text': t, 'vec': list(map(float, v))} for t, v in zip(batch, res.embeddings)]
    have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    have.to_parquet(emb_cache, index=False)
EMB = {t: np.asarray(v, dtype=np.float32) for t, v in zip(have['text'], have['vec'])}
index = sc.VocabularyIndex(V4 / '05_embeddings' / 'element_embeddings.npy',
                           V4 / '05_embeddings' / 'element_embeddings_index.parquet', vocab)
todo = todo.copy()
nn = index.nearest(np.stack([EMB[t] for t in todo['norm']])) if len(todo) else []
todo['candidates'] = [sc.build_candidates(t, vocab, n) for t, n in zip(todo['norm'], nn)]
print(len(todo), 'units go to the three coders')
""")

md("## Three coders, blind to genre (responses cached)")
code(r"""
callers = sc.make_callers(get_secret('ANTHROPIC_API_KEY'),
                          get_secret('SAF_OpenAI', 'CharacterAI_OpenAI'),
                          get_secret('SAF_Gemini', 'CharacterAI_Gemini'))
raw_cache = CACHE / 'coder_raw.jsonl'
done = {}
if raw_cache.exists():
    for line in raw_cache.read_text().splitlines():
        r = json.loads(line); done[(r['coder'], r['prompt_sha'])] = r['text']
failed = []
with raw_cache.open('a') as fh:
    for r in todo.itertuples(index=False):
        prompt = sc.unit_prompt(r.box, r.entry_text, r.unit_text, r.candidates)
        ph = hashlib.sha256(prompt.encode()).hexdigest()
        for cid, fn in callers.items():
            if (cid, ph) in done: continue
            res = sc.call_with_retries(lambda: fn(RUBRIC, prompt))
            if res is None:                     # API failure: not cached, so a rerun retries it
                failed.append((cid, r.unit_id)); continue
            text, served = res
            done[(cid, ph)] = text
            fh.write(json.dumps({'coder': cid, 'model': served, 'prompt_sha': ph,
                                 'unit_id': r.unit_id, 'text': text}) + '\n')
print(len(failed), 'calls failed after retries (rerun this cell to retry them)')
# The model that served each cached response; anything other than the registered model is a deviation.
if raw_cache.stat().st_size:
    print(pd.read_json(raw_cache, lines=True).groupby(['coder', 'model']).size())
votes = []
for r in todo.itertuples(index=False):
    ph = hashlib.sha256(sc.unit_prompt(r.box, r.entry_text, r.unit_text, r.candidates).encode()).hexdigest()
    v = {cid: sc.valid_labels(sc.parse_coder_output(done.get((cid, ph)) or ''), vocab) for cid in sc.CODERS}
    votes.append({'unit_id': r.unit_id, **{f'v_{k}': v[k] for k in v}, 'labels': sc.consensus(v)})
votes = pd.DataFrame(votes)
""")

md("## Consensus, reliability and the coded units")
code(r"""
coded = units.merge(votes, on='unit_id', how='left')
det = coded['det_match'].notna()
coded.loc[det, 'labels'] = coded.loc[det, 'det_match'].map(lambda s: [s])
coded['rule'] = np.where(det, 'deterministic', 'coders')
P = {cid: [sc.primary_label(x) for x in coded.loc[~det, f'v_{cid}']] for cid in sc.CODERS}
ratings = list(zip(*P.values()))
pairs = [('claude', 'gpt'), ('claude', 'gemini'), ('gpt', 'gemini')]
rel = {'fleiss_kappa': sc.fleiss_kappa(ratings),
       'krippendorff_alpha': sc.krippendorff_alpha_nominal(ratings),
       'mean_pairwise_cohen': float(np.nanmean([sc.cohen_kappa(P[a], P[b]) for a, b in pairs])),
       'n_units_coders': int((~det).sum()),
       'n_no_consensus': int(coded['labels'].map(lambda l: l == [sc.NO_CONSENSUS]).sum()),
       'share_missing_by_coder': {cid: float(np.mean([x is None for x in P[cid]])) for cid in sc.CODERS}}
rel['by_genre'] = {}
for g, d in coded.loc[~det].groupby('genre'):
    Pg = {cid: [sc.primary_label(x) for x in d[f'v_{cid}']] for cid in sc.CODERS}
    rg = list(zip(*Pg.values()))
    rel['by_genre'][g] = {'n_units': len(d), 'fleiss_kappa': sc.fleiss_kappa(rg),
                          'krippendorff_alpha': sc.krippendorff_alpha_nominal(rg),
                          'mean_pairwise_cohen': float(np.nanmean([sc.cohen_kappa(Pg[a], Pg[b]) for a, b in pairs]))}
print(json.dumps(rel, indent=2))
coded.to_parquet(OUT / 'units_coded_automated.parquet', index=False)
""")

md(r"""
## Human validation (registered decision rule)

Run the first cell once to export the sample and the two blind coder sheets. When both human coders have returned their sheets (`human_coder_A.csv`, `human_coder_B.csv`, labels in `label_1` to `label_3`), run the second cell. The automated coding is primary if the human coders agree with each other at Cohen's kappa of at least .60 and the mean kappa between each human and the automated coding is at least .60. Otherwise the two humans code every unit, disagreements are resolved by discussion, and that coding becomes primary, with the automated coding reported as a sensitivity analysis.
""")
code(r"""
sample = sc.validation_sample(units, n_per_genre=50)
sample.to_csv(OUT / 'validation_sample_key.csv', index=False)
sheet = sc.coder_sheet(sample)
for name in ['human_coder_A.csv', 'human_coder_B.csv']:
    if not (OUT / name).exists():                 # never overwrite a sheet a coder may have filled in
        sheet.to_csv(OUT / name, index=False)
print('validation sample of', len(sample), 'units; coder sheets in', OUT)
""")
code(r"""
def sheet_done(path):
    h = pd.read_csv(path, dtype=str).fillna('')
    return (h['label_1'].str.strip() != '').all()
PRIMARY = None
if not all(sheet_done(OUT / f) for f in ['human_coder_A.csv', 'human_coder_B.csv']):
    print('Waiting for both human coders to finish their sheets; the decision rule has not been applied.')
def human_primary(path):
    h = pd.read_csv(path, dtype=str).fillna('')
    labs = h[['label_1', 'label_2', 'label_3']].apply(lambda r: [x.strip() for x in r if x.strip()], axis=1)
    return dict(zip(h['unit_id'], labs.map(lambda l: sc.primary_label(sc.valid_labels({'labels': l}, vocab)))))
if all(sheet_done(OUT / f) for f in ['human_coder_A.csv', 'human_coder_B.csv']):
    A, B = human_primary(OUT / 'human_coder_A.csv'), human_primary(OUT / 'human_coder_B.csv')
    auto = dict(zip(coded['unit_id'], coded['labels'].map(sc.primary_label)))
    ids = list(sample['unit_id'])
    k_hh = sc.cohen_kappa([A[i] for i in ids], [B[i] for i in ids])
    k_ha = np.nanmean([sc.cohen_kappa([A[i] for i in ids], [auto[i] for i in ids]),
                       sc.cohen_kappa([B[i] for i in ids], [auto[i] for i in ids])])
    PRIMARY = 'automated' if (k_hh >= .60 and k_ha >= .60) else 'human'
    print(f'human-human kappa {k_hh:.3f}; mean human-automated kappa {k_ha:.3f}; primary coding: {PRIMARY}')
    # Reported, not part of the rule: the same agreement on the sampled units that went to the coder models.
    routed = set(coded.loc[coded['rule'] == 'coders', 'unit_id']); sub = [i for i in ids if i in routed]
    k_ha_sub = np.nanmean([sc.cohen_kappa([A[i] for i in sub], [auto[i] for i in sub]),
                           sc.cohen_kappa([B[i] for i in sub], [auto[i] for i in sub])])
    print(f'{len(sub)} sampled units went to the coders; mean human-automated kappa on them {k_ha_sub:.3f}')
    def pct_agree(x, y):
        pr = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        return float(np.mean([a == b for a, b in pr])) if pr else float('nan')
    pct = {'human_human': pct_agree([A[i] for i in ids], [B[i] for i in ids]),
           'human_automated_mean': float(np.mean([pct_agree([A[i] for i in ids], [auto[i] for i in ids]),
                                                  pct_agree([B[i] for i in ids], [auto[i] for i in ids])]))}
    print('percentage agreement', pct)
    json.dump({'kappa_human_human': k_hh, 'kappa_human_automated_mean': k_ha, 'primary': PRIMARY,
               'n_sample_routed_to_coders': len(sub), 'kappa_human_automated_mean_routed': k_ha_sub,
               'percent_agreement': pct, **rel},
              open(OUT / 'coding_decision.json', 'w'), indent=2)
""")

md("## Quality flags, production indicators, and the genre-discrimination check")
code(r"""
pool = pd.read_csv(V4 / '10_human_study_stimuli' / 'eligible_pool_predictors.csv')
flags = sc.quality_flags(coded); flags.to_csv(OUT / 'quality_flags.csv', index=False)
produced = sc.produced_elements(coded); produced.to_csv(OUT / 'produced_elements.csv', index=False)
pg = (coded[['participant_id', 'genre']].drop_duplicates()
      .assign(genre=lambda d: d['genre'].map(sc.GENRE_KEY_TO_DISPLAY)))
prod = sc.production_matrix(produced, pool, pg); prod.to_parquet(OUT / 'production_matrix.parquet', index=False)
print(sc.outside_pool_report(produced, pool))

# Sensitivity codings for H1 (prereg Section 6.4.7): liberal counts any coder's vocabulary label;
# strict keeps only the deterministic matches.
lib = coded.assign(labels=[r['labels'] if r['rule'] == 'deterministic'
                           else sc.any_coder_labels({c: r[f'v_{c}'] for c in sc.CODERS})
                           for _, r in coded.iterrows()])
strict = coded.assign(labels=[r['labels'] if r['rule'] == 'deterministic' else [sc.NO_CONSENSUS]
                              for _, r in coded.iterrows()])
for name, d in [('liberal', lib), ('strict', strict)]:
    sc.production_matrix(sc.produced_elements(d), pool, pg).to_parquet(OUT / f'production_matrix_{name}.parquet', index=False)

# S1: are a participant's entries closer to the cued genre's plot centroid than to the other genre's?
# The six centroids were computed once with sc.genre_centroids from the Notebook 05 plot embeddings
# and saved in study1_genre_centroids.npz (in the release), so S1 needs neither the plot embeddings
# nor the corpus file.
cz = np.load(CODING / 'study1_genre_centroids.npz')
centroids = {k: cz[k] for k in cz.files}
s1 = sc.genre_discrimination(coded, EMB, centroids); s1.to_csv(OUT / 's1_genre_discrimination.csv', index=False)
print(s1.groupby('cued_genre')[['cos_cued', 'cos_other']].mean())
""")

md(r"""
## Fallback: full human coding as the primary coding

Only if the decision rule above chose `human`. When the two coders have coded every unit and resolved their disagreements (`human_full_coding_resolved.csv`: `unit_id`, `label_1` to `label_3`), this cell builds the human versions of the quality flags and the production matrix, which the analysis script then uses as primary. The automated coding stays as the sensitivity analysis.
""")
code(r"""
if PRIMARY == 'human':
    h = pd.read_csv(OUT / 'human_full_coding_resolved.csv', dtype=str).fillna('')
    labs = h[['label_1', 'label_2', 'label_3']].apply(lambda r: [x.strip() for x in r if x.strip()], axis=1)
    hl = dict(zip(h['unit_id'], labs.map(lambda l: sc.valid_labels({'labels': l}, vocab) or [sc.NO_CONSENSUS])))
    missing = set(coded['unit_id']) - set(hl)
    assert not missing, f'{len(missing)} units have no resolved human code'
    hc = coded.assign(labels=coded['unit_id'].map(hl))
    sc.quality_flags(hc).to_csv(OUT / 'quality_flags_human.csv', index=False)
    sc.production_matrix(sc.produced_elements(hc), pool, pg).to_parquet(OUT / 'production_matrix_human.parquet', index=False)
    print('human primary coding written')
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_study1_response_coding.ipynb')
nbf.write(nb, out); print('wrote', out, 'with', len(cells), 'cells')
