# Builder for notebooks/11b_blind_coder_sheets.ipynb
# Run:  python3 notebooks/_build_11b.py
# Blind copies of the Study 1 human coder sheets (prereg Section 6.4.6), and the restore step
# that writes the returned labels back for notebook 11's decision cell. Same logic as
# study1_analysis/blind_coder_sheets.py, which stays as the command-line version.
import os
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r"""
# 11b. Study 1: blind coder sheets for the human validation

Notebook 11 writes `human_coder_A.csv` and `human_coder_B.csv` with the unit identifier, which contains the participant's SONA code and genre slot. The preregistration (Section 6.4.6) says each coder's sheet shows only a unit identifier, the box, the entry and the unit, and not the genre or the participant. The coders therefore work from blind copies, `coder_A_sheet.csv` (David) and `coder_B_sheet.csv` (Mira). These replace the unit identifier with a plain row code (U001 to U300) and keep notebook 11's random order. `blind_key.csv` maps the row codes back to the unit identifiers. The registered notebook and module are not changed.

Order of work:
1. **Make the sheets.** Done Nov 6, 2026. The cell refuses to overwrite a sheet, so rerunning it is harmless.
2. **Coders fill their sheets.** Each returned sheet goes into `Analysis/v4_rebuild/11_study1_coding/` under its original name. Rows must not be sorted, added or deleted.
3. **Check a sheet.** This lists empty rows and labels that are neither a vocabulary string nor a category code, so the coder can fix typos. It writes nothing.
4. **Restore.** When both sheets pass, this writes the labels into `human_coder_A.csv` and `human_coder_B.csv`, backing up the old files first. Notebook 11's decision cell reads those files.

**Blinding.** Until both sheets are complete, neither coder opens the other's sheet, `blind_key.csv`, `validation_sample_key.csv`, `human_coder_*.csv`, `units_coded_automated.parquet` or the `cache/` folder. Check only your own sheet until both are in.
""")

code(r"""
import sys, shutil
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/DGK Lab/Reasoning Lab/DGK Lab - Scene A Faire Computational')
else:
    PROJECT_ROOT = Path.cwd().parents[1] if Path.cwd().name == 'notebooks' else Path.cwd()
import pandas as pd

AN = PROJECT_ROOT / 'Analysis'
OUT = AN / 'v4_rebuild' / '11_study1_coding'
VOCAB = AN / 'v4_rebuild' / '10_human_study_stimuli' / 'coding_vocabulary.csv'
CODES = {'ABSTRACT_THEME', 'OUTSIDE_VOCABULARY', 'UNINTERPRETABLE'}
LABELS = ['label_1', 'label_2', 'label_3']
CODERS = ['A', 'B']
assert OUT.is_dir(), f'not found: {OUT}'
print('project root:', PROJECT_ROOT)
""")

md(r"""
## Functions

`read_sheet` reads with `utf-8-sig`, so a sheet saved from Excel with a byte-order mark still has its `row` column. Labels are checked the way notebook 11's decision cell reads them. A category code is accepted in any case. Any other label is lowercased and must be a string of `coding_vocabulary.csv`.
""")
code(r"""
class Stop(Exception):
    pass

def read_sheet(path):
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig').fillna('')

def load_vocab():
    return set(pd.read_csv(VOCAB, usecols=['element'])['element'].astype(str))

def make():
    sheets = {c: read_sheet(OUT / f'human_coder_{c}.csv') for c in CODERS}
    if list(sheets['A']['unit_id']) != list(sheets['B']['unit_id']):
        raise Stop('the two notebook sheets list the units in different orders.')
    for c in CODERS:
        if (OUT / f'coder_{c}_sheet.csv').exists():
            print(f'coder_{c}_sheet.csv already exists and may hold a coder\'s work; nothing was written.')
            return
    key = pd.DataFrame({'row': [f'U{i + 1:03d}' for i in range(len(sheets['A']))],
                        'unit_id': sheets['A']['unit_id']})
    key.to_csv(OUT / 'blind_key.csv', index=False)
    for c in CODERS:
        s = sheets[c].drop(columns='unit_id')
        s.insert(0, 'row', key['row'])
        s.to_csv(OUT / f'coder_{c}_sheet.csv', index=False)
    print(f'wrote coder_A_sheet.csv and coder_B_sheet.csv ({len(key)} rows) and blind_key.csv')

def problems_in(c, vocab, expected_rows):
    s = read_sheet(OUT / f'coder_{c}_sheet.csv')
    if 'row' not in s.columns or list(s['row']) != list(expected_rows):
        raise Stop(f'coder {c}\'s sheet has rows missing, added or reordered. Ask for the sheet again, unsorted.')
    probs = []
    for r in s.itertuples(index=False):
        labs = [getattr(r, l).strip() for l in LABELS if getattr(r, l).strip()]
        if not labs:
            probs.append((c, r.row, '(no label)'))
        for lab in labs:
            if lab.upper() not in CODES and lab.lower() not in vocab:
                probs.append((c, r.row, lab))
    return s, probs

def expected_rows():
    # The row codes alone; reading them does not reveal any unit identifier.
    n = len(read_sheet(OUT / 'human_coder_A.csv'))
    return [f'U{i + 1:03d}' for i in range(n)]

def check(c):
    _, probs = problems_in(c, load_vocab(), expected_rows())
    if probs:
        print(f'{len(probs)} labels to fix in coder {c}\'s sheet (row, label):')
        for p in probs:
            print('  ', p[1], p[2])
    else:
        print(f'coder {c}\'s sheet is complete and every label is valid.')

def restore():
    vocab = load_vocab()
    key = read_sheet(OUT / 'blind_key.csv')
    filled, problems = {}, []
    for c in CODERS:
        s, probs = problems_in(c, vocab, key['row'])
        filled[c] = s; problems += probs
    if problems:
        print(f'{len(problems)} labels to fix before restoring (coder, row, label):')
        for p in problems:
            print('  ', *p)
        print('Nothing was written.')
        return
    for c in CODERS:
        dest = OUT / f'human_coder_{c}.csv'
        if dest.exists():
            shutil.copy(dest, OUT / f'human_coder_{c}.before_restore.csv')
        s = filled[c].drop(columns='row')
        s.insert(0, 'unit_id', key['unit_id'])
        s.to_csv(dest, index=False)
    print('both sheets valid; wrote human_coder_A.csv and human_coder_B.csv for the decision cell')
""")

md(r"""
## Step 1. Make the blind sheets

Already done. If both sheets exist, this cell prints a message and writes nothing.
""")
code(r"""
make()
""")

md(r"""
## Step 3. Check a returned sheet

Save the returned sheet into `Analysis/v4_rebuild/11_study1_coding/` under its original name (`coder_A_sheet.csv` or `coder_B_sheet.csv`), replacing the blank one. Then set `CODER` and run the cell. Until both sheets are in, check only your own.
""")
code(r"""
CODER = 'A'
check(CODER)
""")

md(r"""
## Step 4. Restore both sheets for notebook 11

Run this when both sheets pass the check. Then open `11_study1_response_coding.ipynb` and run it from the top. Every model call is cached, so this costs nothing. The decision cell applies the registered rule and writes `coding_decision.json`. With that file in place, knit `study1_analysis/study1_analysis.Rmd` in real mode.
""")
code(r"""
restore()
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11b_blind_coder_sheets.ipynb')
nbf.write(nb, out); print('wrote', out, 'with', len(cells), 'cells')
