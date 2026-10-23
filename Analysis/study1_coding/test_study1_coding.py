"""Offline tests for study1_coding.py: no API calls. Run with
    python3 -m pytest -q test_study1_coding.py
from this folder. Uses the real coding vocabulary and the frozen embeddings."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study1_coding as sc

V4 = HERE.parent / 'v4_rebuild'
VOCAB = sc.Vocabulary.from_csv(V4 / '10_human_study_stimuli' / 'coding_vocabulary.csv')


def test_normalize():
    assert sc.normalize("The Cowboys") == 'cowboys'
    assert sc.normalize("a father's death") == 'father death'
    assert sc.normalize('right-hand man!') == 'right hand man'
    assert sc.normalize('lots of guns') == 'guns'
    assert sc.normalize('“The Big Game”') == 'big game'
    assert sc.normalize('the chosen one') == 'chosen one'          # internal words kept


def test_singular_candidates():
    assert sc.singular_candidates('wolves')[0] == 'wolf'
    assert 'city' in sc.singular_candidates('cities')
    assert sc.singular_candidates('men')[0] == 'man'
    assert sc.singular_candidates('glass') == ['glass']


def test_deterministic_match_uses_vocabulary():
    assert sc.deterministic_match('cowboys', VOCAB) == 'cowboy'
    assert sc.deterministic_match('saloon', VOCAB) == 'saloon'
    assert sc.deterministic_match('bounty hunters', VOCAB) == 'bounty hunter'
    assert sc.deterministic_match('creepy old abandoned house', VOCAB) is None   # left to the coders
    assert 'abandoned house' in sc.right_anchored_matches('creepy old abandoned house', VOCAB)


def test_preferred_type_noun_first():
    assert sc.Vocabulary(pd.DataFrame({'element': ['house', 'house', 'house'],
                                       'element_type': ['verb', 'entity_ORG', 'noun']})).preferred_type('house') == 'noun'


def test_units_split_and_skip_removed_and_tests():
    entries = [{'text': 'sheriff, deputy', 'category': 'characters', 'order': 1, 'removed': False},
               {'text': 'horse', 'category': 'objects', 'order': 2, 'removed': True},
               {'text': 'good versus evil', 'category': 'themes', 'order': 3, 'removed': False}]
    sessions = pd.DataFrame([
        {'participant_id': '12347', 'is_test': False, 'g1': 'western', 'g2': 'horror',
         'genre1_gen_entries': json.dumps(entries), 'genre2_gen_entries': '[]'},
        {'participant_id': 'PILOT_X', 'is_test': True, 'g1': 'western', 'g2': 'horror',
         'genre1_gen_entries': json.dumps(entries), 'genre2_gen_entries': '[]'}])
    u = sc.units_from_sessions(sessions)
    assert list(u['unit_text']) == ['sheriff', 'deputy', 'good versus evil']   # comma split; 'versus' not split
    assert set(u['participant_id']) == {'12347'}


def test_valid_labels_filters_and_ranks():
    assert sc.valid_labels({'labels': ['Sheriff', 'not a real vocab string xyz']}, VOCAB) == ['sheriff']
    assert sc.valid_labels({'labels': ['abstract_theme']}, VOCAB) == ['ABSTRACT_THEME']
    assert sc.valid_labels({'labels': ['sheriff', 'OUTSIDE_VOCABULARY']}, VOCAB) == ['sheriff']
    assert sc.valid_labels({'labels': []}, VOCAB) is None
    assert sc.valid_labels(None, VOCAB) is None


def test_parse_coder_output_tolerates_fences():
    assert sc.parse_coder_output('```json\n{"labels": ["saloon"], "note": "x"}\n```')['labels'] == ['saloon']
    assert sc.parse_coder_output('Here: {"labels": ["saloon"], "note": "x"} done')['labels'] == ['saloon']
    assert sc.parse_coder_output('nonsense') is None


def test_consensus_rules():
    assert sc.consensus({'a': ['sheriff'], 'b': ['sheriff'], 'c': ['marshal']}) == ['sheriff']
    assert sc.consensus({'a': ['sheriff', 'deputy'], 'b': ['sheriff', 'deputy'], 'c': ['sheriff']}) == ['deputy', 'sheriff']
    assert sc.consensus({'a': ['sheriff'], 'b': ['marshal'], 'c': ['lawman']}) == [sc.NO_CONSENSUS]
    assert sc.consensus({'a': ['sheriff'], 'b': ['sheriff'], 'c': None}) == ['sheriff']
    assert sc.consensus({'a': ['sheriff'], 'b': ['marshal'], 'c': None}) == [sc.NO_CONSENSUS]
    assert sc.consensus({'a': ['ABSTRACT_THEME'], 'b': ['ABSTRACT_THEME'], 'c': ['evil']}) == ['ABSTRACT_THEME']
    assert sc.consensus({'a': ['sheriff'], 'b': None, 'c': None}) == [sc.NO_CONSENSUS]


def test_any_coder_labels_is_liberal():
    assert sc.any_coder_labels({'a': ['sheriff'], 'b': ['marshal'], 'c': None}) == ['marshal', 'sheriff']
    assert sc.any_coder_labels({'a': ['ABSTRACT_THEME'], 'b': None, 'c': None}) == [sc.NO_CONSENSUS]


def test_kappas_on_known_values():
    a = ['x', 'x', 'y', 'y', 'x', 'y']
    assert sc.cohen_kappa(a, a) == pytest.approx(1.0)
    # Two raters, 50 units, classic example: po = .7, pe = .5 -> kappa .4
    r1 = ['y'] * 25 + ['n'] * 25
    r2 = ['y'] * 20 + ['n'] * 5 + ['y'] * 10 + ['n'] * 15
    assert sc.cohen_kappa(r1, r2) == pytest.approx(0.4)
    # Fleiss with perfect agreement is 1; Krippendorff nominal with perfect agreement is 1
    perfect = [['x', 'x', 'x'], ['y', 'y', 'y'], ['x', 'x', 'x']]
    assert sc.fleiss_kappa(perfect) == pytest.approx(1.0)
    assert sc.krippendorff_alpha_nominal(perfect) == pytest.approx(1.0)
    # Two raters, no missing data: nominal alpha = 1 - (N-1)/N * (1-po)/(1-pe), with N = 2 x units
    # pairable values and pe from the pooled marginals (Krippendorff's small-sample form of Scott's pi).
    two = [[x, y] for x, y in zip(r1, r2)]
    po = 0.7; p_y = (25 + 30) / 100; pe = p_y ** 2 + (1 - p_y) ** 2; N = 100
    assert sc.krippendorff_alpha_nominal(two) == pytest.approx(1 - (N - 1) / N * (1 - po) / (1 - pe), rel=1e-9)


def test_instruction_echo_and_quality_flags():
    assert sc.is_instruction_echo('The big game')
    assert sc.is_instruction_echo('Character Types')
    assert sc.is_instruction_echo('Western')
    assert not sc.is_instruction_echo('shootout')
    uc = pd.DataFrame([
        {'participant_id': 'p1', 'slot': 1, 'genre': 'western', 'entry_order': i, 'entry_text': t, 'labels': lab}
        for i, (t, lab) in enumerate([('asdf', ['UNINTERPRETABLE']), ('qwer', ['UNINTERPRETABLE']),
                                      ('saloon', ['saloon']), ('zzz', ['UNINTERPRETABLE'])])])
    f = sc.quality_flags(uc)
    assert bool(f['flag'].iloc[0]) and f['share_uninterpretable'].iloc[0] == 0.75


def test_production_matrix_counts_once_and_restricts_to_set():
    pool = pd.DataFrame({'genre': ['Western', 'Western', 'Western'], 'key': ['saloon', 'sheriff', 'horse'],
                         'in_generation_analysis_set': [True, True, False]})
    units = pd.DataFrame([
        {'participant_id': 'p1', 'genre': 'western', 'box': 'settings', 'entry_order': 1, 'labels': ['saloon']},
        {'participant_id': 'p1', 'genre': 'western', 'box': 'settings', 'entry_order': 2, 'labels': ['saloon']},
        {'participant_id': 'p1', 'genre': 'western', 'box': 'objects', 'entry_order': 3, 'labels': ['horse']},
        {'participant_id': 'p1', 'genre': 'western', 'box': 'themes', 'entry_order': 4, 'labels': ['ABSTRACT_THEME']}])
    prod = sc.produced_elements(units)
    assert len(prod) == 2                                   # saloon once, horse once
    m = sc.production_matrix(prod, pool, pd.DataFrame({'participant_id': ['p1'], 'genre': ['Western']}))
    assert set(m['key']) == {'saloon', 'sheriff'}           # horse is outside the analysis set
    assert dict(zip(m['key'], m['produced'])) == {'saloon': 1, 'sheriff': 0}


def test_vocabulary_index_retrieves_itself():
    idx = sc.VocabularyIndex(V4 / '05_embeddings' / 'element_embeddings.npy',
                             V4 / '05_embeddings' / 'element_embeddings_index.parquet', VOCAB)
    assert len(idx.strings) == len(VOCAB.strings)
    i = idx.strings.index('saloon')
    nn = idx.nearest(idx.M[i:i + 1])[0]
    assert nn[0][0] == 'saloon' and nn[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(nn) == sc.TOP_K_CANDIDATES


def test_unit_prompt_hides_genre():
    p = sc.unit_prompt('settings', 'old saloon', 'old saloon', ['saloon', 'bar'])
    for g in ['Western', 'western', 'Horror', 'genre:']:
        assert g not in p
    assert '1. saloon' in p and 'Settings or Time Periods' in p


def test_validation_sample_stratified_and_blind():
    units = pd.DataFrame({'unit_id': [f'u{i}' for i in range(120)],
                          'genre': ['western'] * 60 + ['horror'] * 60,
                          'box': 'settings', 'entry_text': 'x', 'unit_text': 'x'})
    s = sc.validation_sample(units, n_per_genre=50)
    assert s['genre'].value_counts().to_dict() == {'western': 50, 'horror': 50}
    sheet = sc.coder_sheet(s)
    assert 'genre' not in sheet.columns and len(sheet) == 100


def test_genre_centroids_and_discrimination():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(6, 8)).astype(np.float32)
    plot_idx = pd.DataFrame({'union_id': range(6), 'embedding_row_index': range(6)})
    union = pd.DataFrame({'union_id': range(6), 'genres_canonical': [['Western'], ['Western'], ['Horror'],
                                                                      ['Horror', 'Comedy'], ['Comedy'], ['Drama']]})
    c = sc.genre_centroids(E, plot_idx, union)
    assert set(c) == {'western', 'horror', 'comedy'}
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in c.values())
    coded = pd.DataFrame([
        {'participant_id': 'p1', 'slot': 1, 'genre': 'western', 'entry_order': 1, 'entry_text': 'saloon'},
        {'participant_id': 'p1', 'slot': 2, 'genre': 'horror', 'entry_order': 1, 'entry_text': 'ghost'}])
    emb = {'saloon': c['western'], 'ghost': c['horror']}
    s1 = sc.genre_discrimination(coded, emb, c)
    assert len(s1) == 2 and (s1['cos_cued'] > s1['cos_other']).all()
    assert s1.loc[s1['cued_genre'] == 'western', 'cos_cued'].iloc[0] == pytest.approx(1.0, abs=1e-5)
