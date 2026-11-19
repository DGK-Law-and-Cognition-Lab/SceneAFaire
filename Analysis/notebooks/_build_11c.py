#!/usr/bin/env python3
# Builder for notebooks/11c_study1_deidentify.ipynb
# Run:  python3 notebooks/_build_11c.py      (from the Analysis folder)
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Notebook 11c — De-identified Study 1 data for restricted-access deposit

Added November 18, 2026. Builds the participant-level Study 1 data set that is deposited on Zenodo under **restricted access**: researchers request it and accept a data use agreement that prohibits attempts to re-identify participants. That is the sharing the consent form allows ("De-identified data may also be shared with other qualified researchers for purposes of replication, secondary analysis, or meta-analysis under data use agreements that prohibit attempts to re-identify participants").

**Never put the output folder or the key file in the public GitHub repository.**

What the notebook does:

1. Reads every session file in `Data/Study1_raw/` and drops the investigator test sessions (identifiers beginning PILOT or TEST, or flagged `is_test`).
2. Replaces each SONA survey code with a random code (`P001`, ...). The same survey code always gets the same new code, so the registered duplicate-identifier exclusion still works. The key linking old and new codes is written to `Data/Study1_deid_key.csv`, which stays with the research team.
3. Removes or coarsens everything else that could help identify a person: platform and session identifiers, the browser user agent, the captcha answer, screen size, and the free-text fields (occupation, the race "other" text, and procedure feedback). Clock times are replaced by synthetic times that keep the order of sessions and every duration; page timestamps become milliseconds from the first page. Age is top-coded at 25 ("25+"), and the typed country is reduced to United States or other.
4. Keeps every field the registered analysis uses, in the same column names and file layout, so `study1_analysis.Rmd` and `study1_figures.Rmd` run on the deposit unchanged.
5. Applies the same new codes to the outputs of Notebook 11 (coded units, production matrices, quality flags, the genre-discrimination check, and the human validation sheets).
6. Checks the result: no original survey code appears anywhere in the output, the nine registered exclusion rules give the same counts on the de-identified files as on the raw files, and every production matrix has the same totals per participant.
""")

code(r"""
import json, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
except Exception:
    pass

_CANDS = [Path('/content/drive/MyDrive/DGK Lab/Reasoning Lab/DGK Lab - Scene A Faire Computational'),
          Path.cwd().parent, Path.cwd().parent.parent, Path.cwd()]
if (Path.home() / 'Library/CloudStorage').exists():
    _CANDS += list((Path.home() / 'Library/CloudStorage').glob('GoogleDrive-*/My Drive/DGK Lab/Reasoning Lab/DGK Lab - Scene A Faire Computational'))
ROOT = next(c for c in _CANDS if (c / 'Analysis' / 'v4_rebuild').exists())
RAW = ROOT / 'Data' / 'Study1_raw'
CODED = ROOT / 'Analysis' / 'v4_rebuild' / '11_study1_coding'
OUT = ROOT / 'Analysis' / 'v4_rebuild' / '11c_study1_deidentified'
KEY = ROOT / 'Data' / 'Study1_deid_key.csv'
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / 'sessions').mkdir(parents=True)
(OUT / 'coded').mkdir()
TMP = Path(tempfile.mkdtemp())
def read_parquet(p):
    local = TMP / p.name
    shutil.copy(p, local)
    return pd.read_parquet(local)
print('ROOT =', ROOT)
""")

md("## 1. Read the sessions and drop test sessions")
code(r"""
files = sorted(RAW.glob('*SAF_STUDY1_*.csv'))
raw = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
is_true = lambda x: x.str.strip().str.lower().eq('true')
test = raw['participant_id'].str.match(r'(?i)^(pilot|test)') | is_true(raw['is_test'])
raw = raw[~test].reset_index(drop=True)
print(f'{len(files)} files; {int(test.sum())} test session(s) dropped; {len(raw)} sessions kept')
""")

md("## 2. New participant codes (the key stays with the research team)")
code(r"""
ids = sorted(raw['participant_id'].unique())
if KEY.exists():
    key = pd.read_csv(KEY, dtype=str)
    missing = set(ids) - set(key['survey_code'])
    assert not missing, f'{len(missing)} survey codes are not in the existing key; delete the key to rebuild it'
else:
    rng = np.random.default_rng()          # deliberately unseeded: the codes must not be recomputable
    perm = rng.permutation(len(ids))
    key = pd.DataFrame({'survey_code': ids, 'pid': [f'P{i + 1:03d}' for i in perm]})
    key.to_csv(KEY, index=False)
MAP = dict(zip(key['survey_code'], key['pid']))
print(f'{len(MAP)} codes; key at {KEY}')
""")

md("## 3. De-identify the session table")
code(r"""
s = raw.copy()
s['participant_id'] = s['participant_id'].map(MAP)
DROP = ['prolific_pid', 'study_id', 'session_id', 'pavlovia_session', 'user_agent', 'captcha_response',
        'consent_time', 'screen_width', 'screen_height', 'occupation', 'race_other',
        'feedback_procedure', 'feedback_other']
s = s.drop(columns=[c for c in DROP if c in s.columns])

# synthetic clock: same order, same durations, no real dates
order = pd.to_datetime(raw['start_time'], utc=True, errors='coerce').rank(method='first')
base = pd.Timestamp('2000-01-01T00:00:00Z')
start = base + pd.to_timedelta(order.fillna(len(raw) + 1) * 60, unit='m')
dur = pd.to_numeric(raw['total_duration_ms'], errors='coerce')
s['start_time'] = start.dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
s['end_time'] = np.where(raw['end_time'].str.strip().eq(''), '',
                         (start + pd.to_timedelta(dur.fillna(0), unit='ms')).dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
def rel_pages(x):
    if not x.strip():
        return x
    p = json.loads(x)
    t0 = min(e['t'] for e in p)
    return json.dumps([{'p': e['p'], 't': e['t'] - t0} for e in p])
s['page_timestamps'] = raw['page_timestamps'].map(rel_pages)

age = pd.to_numeric(raw['age'], errors='coerce')
s['age'] = np.where(age.isna(), '', np.where(age >= 25, '25+', age.astype('Int64').astype(str)))
US = {'united states', 'us', 'usa', 'u.s.', 'u.s.a.', 'united states of america', 'america'}
s['country'] = raw['country'].map(lambda c: '' if not c.strip() else ('United States' if c.strip().lower() in US else 'other'))
print('kept columns:', len(s.columns))
print('age:', s['age'].value_counts().sort_index().to_dict())
print('country:', s['country'].value_counts().to_dict())
""")

code(r"""
# one file per session, in the layout the analysis reads; a participant with two sessions gets _a, _b
s['_n'] = s.groupby('participant_id').cumcount()
for _, r in s.iterrows():
    tag = 'SESSION' if r['_n'] == 0 else f'SESSION_{r["_n"] + 1}'
    r.drop('_n').to_frame().T.to_csv(OUT / 'sessions' / f'data_SAF_STUDY1_{r["participant_id"]}_{tag}.csv', index=False)
s = s.drop(columns='_n')
print(len(list((OUT / 'sessions').glob('*.csv'))), 'session files written')
""")

md("## 4. The Notebook 11 outputs under the new codes")
code(r"""
def remap_pid(df):
    df = df.copy()
    if 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].astype(str).map(MAP)
        assert df['participant_id'].notna().all(), 'a coded row has a survey code outside the kept sessions'
    if 'unit_id' in df.columns:
        parts = df['unit_id'].astype(str).str.split('|', n=1, expand=True)
        df['unit_id'] = parts[0].map(MAP) + '|' + parts[1]
        assert df['unit_id'].notna().all()
    return df

for name in ['units_coded_automated.parquet', 'production_matrix.parquet',
             'production_matrix_liberal.parquet', 'production_matrix_strict.parquet']:
    d = read_parquet(CODED / name)
    keep = d['participant_id'].astype(str).isin(MAP)
    print(f'{name}: {len(d):,} rows; {int((~keep).sum())} rows of test sessions dropped')
    remap_pid(d[keep]).to_parquet(OUT / 'coded' / name, index=False)
for name in ['produced_elements.csv', 'quality_flags.csv', 's1_genre_discrimination.csv',
             'human_coder_A.csv', 'human_coder_B.csv', 'validation_sample_key.csv']:
    d = pd.read_csv(CODED / name, dtype=str, keep_default_na=False)
    col = 'participant_id' if 'participant_id' in d.columns else 'unit_id'
    keep = d[col].str.split('|').str[0].isin(MAP)
    print(f'{name}: {len(d):,} rows; {int((~keep).sum())} rows of test sessions dropped')
    remap_pid(d[keep]).to_csv(OUT / 'coded' / name, index=False)
shutil.copy(CODED / 'coding_decision.json', OUT / 'coded' / 'coding_decision.json')
""")

md("## 5. Checks")
code(r"""
# (a) no survey code in any file name or identifier column; full-text hits are listed for inspection,
#     since a five-digit response time can equal a survey code by chance
pat = re.compile(r'(?<![0-9])(' + '|'.join(re.escape(i) for i in ids) + r')(?![0-9])')
leaks, text_hits = [], []
for f in OUT.rglob('*'):
    if not f.is_file():
        continue
    leaks += [(f.name, m) for m in pat.findall(f.name)]
    if f.suffix in ('.csv', '.parquet'):
        d = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f, dtype=str, keep_default_na=False)
        for c in ('participant_id', 'unit_id'):
            if c in d.columns:
                leaks += [(f.name, m) for m in set(pat.findall(' '.join(d[c].astype(str))))]
        text_hits += [(f.name, m) for m in set(pat.findall(d.to_csv(index=False)))]
print('survey codes in file names or identifier columns:', len(leaks), leaks[:5])
print('five-digit values elsewhere that equal a survey code (inspect; expected to be timings):', text_hits[:10])

# (b) the registered exclusion rules give the same counts
def rules(t, flagged):
    tt = lambda c: t[c].str.strip().str.lower().eq('true')
    t = t.sort_values('start_time')
    return pd.DataFrame({
        'r1_hidden_trap': ~tt('captcha_pass'),
        'r2_reading_speed': tt('spr_fast_any'),
        'r3_task_check': tt('task_check_failed_both'),
        'r4_attention': pd.to_numeric(t['attention_pass_total'], errors='coerce').fillna(-1) < 4,
        'r5_incomplete': t['end_time'].str.strip().eq('') | t['genre1_ratings'].str.strip().eq('') | t['genre2_ratings'].str.strip().eq(''),
        'r6_generation_quality': t['participant_id'].isin(flagged),
        'r7_ai_use': t['ai_usage'].eq('Yes'),
        'r8_duplicate_id': t['participant_id'].duplicated(),
        'r9_not_serious': t['seriousness'].eq('1')}).sum()
qf_raw = pd.read_csv(CODED / 'quality_flags.csv', dtype=str)
qf_new = pd.read_csv(OUT / 'coded' / 'quality_flags.csv', dtype=str)
before = rules(raw, set(qf_raw.loc[qf_raw['flag'].str.lower().eq('true'), 'participant_id']))
after = rules(s, set(qf_new.loc[qf_new['flag'].str.lower().eq('true'), 'participant_id']))
cmp_rules = pd.DataFrame({'raw': before, 'deidentified': after})
print(cmp_rules.to_string())

# (c) production totals per participant unchanged
pm_raw = read_parquet(CODED / 'production_matrix.parquet')
pm_raw = pm_raw[pm_raw['participant_id'].astype(str).isin(MAP)]
a = pm_raw.assign(participant_id=pm_raw['participant_id'].astype(str).map(MAP)).groupby('participant_id')['produced'].sum()
b = pd.read_parquet(OUT / 'coded' / 'production_matrix.parquet').groupby('participant_id')['produced'].sum()
same_totals = a.sort_index().equals(b.sort_index())
print('production totals identical:', same_totals)
assert not leaks and cmp_rules['raw'].equals(cmp_rules['deidentified']) and same_totals
print('ALL CHECKS PASSED')
""")

md("## 6. README and data use agreement for the deposit")
code(r"""
readme = '''# Study 1 participant data (de-identified), restricted access

Kamper, Bartholomew, Patel & Holyoak, "Common or Characteristic? An Empirical Method for Scenes a Faire."
Study 1 preregistration: https://osf.io/4azwv. Code and stimuli: https://doi.org/10.5281/zenodo.24356357.

## Access
These files are shared only with qualified researchers for replication, secondary analysis, or
meta-analysis, under the data use agreement in DATA_USE_AGREEMENT.md, as the participants' consent
form provides. Do not redistribute them.

## Contents
sessions/   one CSV per session, the format the experiment wrote (data_SAF_STUDY1_<pid>_SESSION.csv)
coded/      the outputs of the response-coding notebook (11) under the same participant codes:
            units_coded_automated.parquet, production_matrix*.parquet, produced_elements.csv,
            quality_flags.csv, s1_genre_discrimination.csv, human_coder_A.csv, human_coder_B.csv,
            validation_sample_key.csv, coding_decision.json

## What was changed to de-identify the data
- SONA survey codes were replaced with random codes (P001 ...). The same person always has the same code.
- Removed: platform and session identifiers, browser user agent, captcha answer, screen size, consent time,
  and the free-text fields occupation, race (other), and procedure feedback.
- Clock times are synthetic: sessions keep their order and every duration, but the dates are not real.
  page_timestamps give milliseconds from the first page.
- Age is top-coded at 25 ("25+"). Country is "United States" or "other".
- Investigator test sessions were removed.

## Running the registered analysis
Put sessions/ at Data/Study1_raw/ and coded/ at Analysis/v4_rebuild/11_study1_coding/ in a copy of the
replication package, then knit Analysis/study1_analysis/study1_analysis.Rmd with mode: real.
The nine registered exclusion rules give the same counts on these files as on the original files.
'''
dua = '''# Data use agreement: Study 1 participant data

By requesting and downloading these files, you agree that you will:

1. Use the data only for research purposes: replication, secondary analysis, or meta-analysis.
2. Make no attempt to identify any participant, alone or by linking the data with other information,
   and contact the authors at once if you believe you could identify someone.
3. Not share, post, or redistribute the data. Others who want the data must request it themselves.
4. Keep the data on secure, access-controlled storage and delete it when your project ends.
5. Cite the article and this data set in any work that uses the data.

The data come from a study certified exempt by the UCLA Office of the Human Research Protection
Program (IRB-25-2264). Requests are reviewed by the authors.
'''
(OUT / 'README.md').write_text(readme)
(OUT / 'DATA_USE_AGREEMENT.md').write_text(dua)
print(sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file())[:12], '...')
""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}, 'colab': {'provenance': []}}
nbf.write(nb, 'notebooks/11c_study1_deidentify.ipynb')
print('wrote notebooks/11c_study1_deidentify.ipynb')
