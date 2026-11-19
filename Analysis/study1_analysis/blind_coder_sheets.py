"""Blind copies of the Study 1 human coder sheets (prereg Section 6.4.6).

Notebook 11 writes human_coder_A.csv and human_coder_B.csv with the unit
identifier, which contains the participant's SONA code and genre slot. The
preregistration says the coders' sheets show neither the participant nor the
genre, so the coders work from these copies instead, which replace the unit
identifier with a plain row code (U001, U002, ...) and keep the notebook's
random order. The registered notebook and module are not changed.

  make     write coder_A_sheet.csv and coder_B_sheet.csv for the coders, and
           blind_key.csv (row code to unit_id) for the investigators only.
  restore  check the returned sheets against the vocabulary and write the labels
           back into human_coder_A.csv and human_coder_B.csv for the notebook's
           decision cell. Nothing is written while a sheet has an empty row or a
           label that is neither a vocabulary string nor a category code, so typos
           can be corrected by the coder first.

The Colab version, with a per-sheet check, is notebooks/11b_blind_coder_sheets.ipynb
(built by notebooks/_build_11b.py). This script is the command-line equivalent.

Usage:
  python blind_coder_sheets.py make    <11_study1_coding dir>
  python blind_coder_sheets.py restore <11_study1_coding dir> <coding_vocabulary.csv>
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

CODES = {'ABSTRACT_THEME', 'OUTSIDE_VOCABULARY', 'UNINTERPRETABLE'}
LABELS = ['label_1', 'label_2', 'label_3']
CODERS = ['A', 'B']


def make(out: Path) -> None:
    sheets = {c: pd.read_csv(out / f'human_coder_{c}.csv', dtype=str).fillna('') for c in CODERS}
    if list(sheets['A']['unit_id']) != list(sheets['B']['unit_id']):
        sys.exit('STOP: the two notebook sheets list the units in different orders.')
    key = pd.DataFrame({'row': [f'U{i + 1:03d}' for i in range(len(sheets['A']))],
                        'unit_id': sheets['A']['unit_id']})
    for c in CODERS:
        dest = out / f'coder_{c}_sheet.csv'
        if dest.exists():
            sys.exit(f'STOP: {dest.name} already exists; it may hold a coder\'s work. Nothing was written.')
    key.to_csv(out / 'blind_key.csv', index=False)
    for c in CODERS:
        s = sheets[c].drop(columns='unit_id')
        s.insert(0, 'row', key['row'])
        s.to_csv(out / f'coder_{c}_sheet.csv', index=False)
    print(f'wrote coder_A_sheet.csv and coder_B_sheet.csv ({len(key)} rows) and blind_key.csv in {out}')


def restore(out: Path, vocab_path: Path) -> None:
    vocab = set(pd.read_csv(vocab_path, usecols=['element'])['element'].astype(str))
    key = pd.read_csv(out / 'blind_key.csv', dtype=str)
    problems, filled = [], {}
    for c in CODERS:
        s = pd.read_csv(out / f'coder_{c}_sheet.csv', dtype=str, encoding='utf-8-sig').fillna('')   # Excel may add a BOM
        if list(s['row']) != list(key['row']):
            sys.exit(f'STOP: coder {c}\'s sheet has rows missing, added or reordered.')
        for r in s.itertuples(index=False):
            labs = [getattr(r, l).strip() for l in LABELS if getattr(r, l).strip()]
            if not labs:
                problems.append((c, r.row, '(no label)'))
            for lab in labs:
                if lab.upper() not in CODES and lab.lower() not in vocab:
                    problems.append((c, r.row, lab))
        filled[c] = s
    if problems:
        print(f'{len(problems)} labels to fix before restoring (coder, row, label):')
        for p in problems:
            print('  ', *p)
        sys.exit('Nothing was written.')
    for c in CODERS:
        dest = out / f'human_coder_{c}.csv'
        if dest.exists():
            shutil.copy(dest, out / f'human_coder_{c}.before_restore.csv')
        s = filled[c].drop(columns='row')
        s.insert(0, 'unit_id', key['unit_id'])
        s.to_csv(dest, index=False)
    print('both sheets valid; wrote human_coder_A.csv and human_coder_B.csv for the decision cell')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'make':
        make(Path(sys.argv[2]))
    elif len(sys.argv) >= 4 and sys.argv[1] == 'restore':
        restore(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)
