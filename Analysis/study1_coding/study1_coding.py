"""Study 1 v2: automated coding of the generation-task responses.

Described in the Study 1 v2 preregistration (Section 6.4). The pipeline maps
every free-text entry onto the corpus vocabulary in five stages:

  1. Units. Each non-removed entry is split on explicit list delimiters
     (comma, semicolon, slash, ampersand, plus, line break) into units.
  2. Deterministic match. A unit whose normalized text, or whose text with the
     head noun singularized, is a vocabulary string is labeled with it. No model
     is involved, and nothing about the genre is used.
  3. Candidates. Every other unit is embedded with voyage-4-large
     (input_type='document', the setting of Notebooks 05 and 07b) and compared
     with the frozen embeddings of the 17,965 narrative vocabulary strings; the
     15 nearest strings, plus any string reached by dropping words from the left,
     form its candidate list.
  4. Three coders. Claude Opus 5.5, GPT-5.4-mini and Gemini 3 Flash each label the
     unit independently from the frozen rubric (coding_rubric_v1.md), the box,
     the entry and the candidate list. The genre is never shown. Labels that are
     neither vocabulary strings nor category codes are discarded.
  5. Consensus. A label is kept when at least two of three coders give it (two of
     two when one coder's answer is missing). A vocabulary label outranks a
     category code. A unit with no consensus is NO_CONSENSUS and counts as
     not produced.

Everything except the API calls is deterministic and is tested offline in
test_study1_coding.py. API keys come from Colab Secrets only.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Frozen settings
# ----------------------------------------------------------------------------
SEED = 20260219
TOP_K_CANDIDATES = 15
SEMANTIC_MODEL = 'voyage-4-large'
SEMANTIC_INPUT_TYPE = 'document'
CATEGORY_CODES = ('ABSTRACT_THEME', 'OUTSIDE_VOCABULARY', 'UNINTERPRETABLE')
NO_CONSENSUS = 'NO_CONSENSUS'
MAX_LABELS = 3
TYPE_PREFERENCE = ('noun', 'chunk', 'verb')   # entity_* types after these
SPLIT_PATTERN = re.compile(r'[,;/&+\n]+')

# Coder models. If a model is unavailable at coding time, the closest successor in
# the same family is used and the substitution is reported as a deviation.
CODERS = {
    'claude': {'family': 'anthropic', 'model': 'claude-opus-5-5'},
    'gpt': {'family': 'openai', 'model': 'gpt-5.4-mini'},
    'gemini': {'family': 'google', 'model': 'gemini-3-flash-preview'},
}

LEADING_STOPWORDS = {
    'a', 'an', 'the', 'some', 'many', 'several', 'various', 'multiple', 'few',
    'lots', 'lot', 'of', 'his', 'her', 'their', 'its', 'my', 'our', 'your',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
}
IRREGULAR_SINGULAR = {
    'men': 'man', 'women': 'woman', 'children': 'child', 'feet': 'foot', 'teeth': 'tooth',
    'mice': 'mouse', 'geese': 'goose', 'people': 'person', 'oxen': 'ox', 'dice': 'die',
}

# Instruction text participants saw; entries that only echo it count toward the
# quality flag of exclusion rule 6.
BOX_LABELS = {
    'settings': 'Settings or Time Periods', 'characters': 'Character Types',
    'objects': 'Objects or Props', 'events': 'Plot Events or Actions', 'themes': 'Themes or Ideas',
}
BOX_HINTS = [
    'Where and when do these films usually take place?', 'What kinds of characters usually appear?',
    'What objects or props usually appear?', 'What usually happens? Events, conflicts, or actions.',
    'What themes or messages are common?',
]
INSTRUCTION_EXAMPLES = ['locker room', 'coach', 'the big game', 'big game']
GENRE_NAMES = ['western', 'horror', 'science fiction', 'crime', 'comedy', 'fantasy', 'sci-fi', 'sci fi']


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# Normalization and deterministic matching
# ----------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, unify punctuation, drop possessive endings and leading
    articles, quantifiers and possessive pronouns. Internal words are kept."""
    t = unicodedata.normalize('NFC', str(text)).lower().strip()
    t = t.replace('’', "'").replace('‘', "'")
    t = re.sub(r"'s\b", '', t)                 # father's death -> father death
    t = re.sub(r'[-_]', ' ', t)                 # right-hand man -> right hand man
    t = re.sub(r"[^\w\s']", ' ', t)
    t = t.replace("'", '')
    toks = t.split()
    while toks and toks[0] in LEADING_STOPWORDS:
        toks = toks[1:]
    return ' '.join(toks)


def singular_candidates(word: str) -> List[str]:
    """Singular forms of an English plural, most specific first, then the word."""
    out: List[str] = []
    if word in IRREGULAR_SINGULAR:
        out.append(IRREGULAR_SINGULAR[word])
    if word.endswith('ves') and len(word) > 4:
        out += [word[:-3] + 'f', word[:-3] + 'fe']
    if word.endswith('ies') and len(word) > 4:
        out.append(word[:-3] + 'y')
    if word.endswith('es') and len(word) > 4:
        out.append(word[:-2])
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        out.append(word[:-1])
    out.append(word)
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w); uniq.append(w)
    return uniq


@dataclass
class Vocabulary:
    """The narrative vocabulary: element strings with their types."""
    table: pd.DataFrame                      # columns element, element_type
    strings: set = field(init=False)
    types: Dict[str, List[str]] = field(init=False)

    def __post_init__(self):
        t = self.table.copy()
        t['element'] = t['element'].astype(str)
        self.strings = set(t['element'])
        grouped = t.groupby('element')['element_type'].apply(list).to_dict()
        def rank(tp):
            return TYPE_PREFERENCE.index(tp) if tp in TYPE_PREFERENCE else len(TYPE_PREFERENCE)
        self.types = {k: sorted(v, key=lambda tp: (rank(tp), tp)) for k, v in grouped.items()}

    @classmethod
    def from_csv(cls, path) -> 'Vocabulary':
        return cls(pd.read_csv(path, usecols=['element', 'element_type']))

    def preferred_type(self, element: str) -> Optional[str]:
        tps = self.types.get(element)
        return tps[0] if tps else None

    def __contains__(self, s: str) -> bool:
        return s in self.strings


def head_singular_forms(norm: str) -> List[str]:
    toks = norm.split()
    if not toks:
        return []
    return [' '.join(toks[:-1] + [h]) for h in singular_candidates(toks[-1])]


def deterministic_match(norm: str, vocab: Vocabulary) -> Optional[str]:
    """Exact match of the normalized unit, or of it with the head noun made singular."""
    if not norm:
        return None
    for form in head_singular_forms(norm):
        if form in vocab:
            return form
    return None


def right_anchored_matches(norm: str, vocab: Vocabulary) -> List[str]:
    """Vocabulary strings reached by dropping words from the left (head singularized).
    Offered to the coders as candidates; never assigned automatically."""
    toks = norm.split()
    found = []
    for i in range(1, len(toks)):
        sub = ' '.join(toks[i:])
        for form in head_singular_forms(sub):
            if form in vocab and form not in found:
                found.append(form)
    return found


# ----------------------------------------------------------------------------
# Units from the experiment's data files
# ----------------------------------------------------------------------------
def units_from_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per unit. `sessions` holds one row per participant with columns
    participant_id, is_test, g1, g2, genre1_gen_entries, genre2_gen_entries."""
    rows = []
    for r in sessions.itertuples(index=False):
        if str(getattr(r, 'is_test', False)).lower() == 'true':
            continue
        for slot in (1, 2):
            genre = getattr(r, f'g{slot}')
            entries = json.loads(getattr(r, f'genre{slot}_gen_entries') or '[]')
            for e in entries:
                if e.get('removed'):
                    continue
                parts = [p.strip() for p in SPLIT_PATTERN.split(str(e['text'])) if p.strip()]
                for k, part in enumerate(parts):
                    rows.append({
                        'participant_id': str(r.participant_id), 'slot': slot, 'genre': genre,
                        'box': e['category'], 'entry_order': e['order'], 'part': k,
                        'entry_text': e['text'], 'unit_text': part,
                        'unit_id': f'{r.participant_id}|{slot}|{e["order"]}|{k}',
                    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Candidate retrieval from frozen embeddings
# ----------------------------------------------------------------------------
class VocabularyIndex:
    """Frozen voyage-4-large embeddings of the narrative vocabulary strings."""

    def __init__(self, emb_path, idx_path, vocab: Vocabulary):
        E = np.load(emb_path, mmap_mode='r')
        idx = pd.read_parquet(idx_path, columns=['element', 'element_type', 'embedding_row_index'])
        idx['element'] = idx['element'].astype(str)
        idx = idx[idx['element'].isin(vocab.strings)]
        idx = idx.sort_values(['element', 'element_type']).drop_duplicates('element')
        self.strings = idx['element'].tolist()
        M = np.asarray(E[idx['embedding_row_index'].to_numpy()], dtype=np.float32)
        self.M = M / np.linalg.norm(M, axis=1, keepdims=True)

    def nearest(self, query_vecs: np.ndarray, k: int = TOP_K_CANDIDATES) -> List[List[Tuple[str, float]]]:
        Q = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        S = Q @ self.M.T
        top = np.argsort(-S, axis=1)[:, :k]
        return [[(self.strings[j], float(S[i, j])) for j in top[i]] for i in range(len(Q))]


def build_candidates(norm: str, vocab: Vocabulary, nn: List[Tuple[str, float]]) -> List[str]:
    out = []
    for s in right_anchored_matches(norm, vocab) + [s for s, _ in nn]:
        if s not in out:
            out.append(s)
    return out


# ----------------------------------------------------------------------------
# Coder prompts, calls and parsing
# ----------------------------------------------------------------------------
def unit_prompt(box: str, entry_text: str, unit_text: str, candidates: Sequence[str]) -> str:
    lines = [f'Box: {BOX_LABELS.get(box, box)}',
             f'Entry as typed: "{entry_text}"']
    if unit_text != entry_text:
        lines.append(f'Part of the entry to code: "{unit_text}"')
    lines.append('Candidate vocabulary strings (most similar first):')
    lines += [f'{i + 1}. {c}' for i, c in enumerate(candidates)]
    lines.append('Return the JSON object described in the rubric.')
    return '\n'.join(lines)


OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {'labels': {'type': 'array', 'items': {'type': 'string'}},
                   'note': {'type': 'string'}},
    'required': ['labels', 'note'],
    'additionalProperties': False,
}


def parse_coder_output(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.startswith('```'):
        t = '\n'.join(l for l in t.split('\n') if not l.strip().startswith('```'))
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find('{'), t.rfind('}')
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


def valid_labels(parsed: Optional[dict], vocab: Vocabulary) -> Optional[List[str]]:
    """Labels that are vocabulary strings or category codes; None if nothing usable."""
    if not parsed or not isinstance(parsed.get('labels'), list):
        return None
    out = []
    for lab in parsed['labels'][:MAX_LABELS]:
        if not isinstance(lab, str):
            continue
        s = lab.strip()
        if s.upper() in CATEGORY_CODES:
            s = s.upper()
        else:
            s = s.lower()
            if s not in vocab:
                continue
        if s not in out:
            out.append(s)
    if any(l in CATEGORY_CODES for l in out) and any(l not in CATEGORY_CODES for l in out):
        out = [l for l in out if l not in CATEGORY_CODES]    # vocabulary labels outrank codes
    return out or None


def make_callers(anthropic_key: str, openai_key: str, google_key: str) -> Dict[str, Callable[[str, str], Tuple[str, str]]]:
    """Return {coder_id: f(system_rubric, user_prompt) -> (raw text, model that served it)}.
    Imports are local so the deterministic parts of the module run without the SDKs installed."""
    import anthropic
    import openai
    from google import genai as google_genai

    a_client = anthropic.Anthropic(api_key=anthropic_key)
    o_client = openai.OpenAI(api_key=openai_key)
    g_client = google_genai.Client(api_key=google_key)

    def call_claude(system: str, prompt: str) -> Tuple[str, str]:
        # Opus 5.5: thinking is always on and sampling parameters are not accepted;
        # effort is the only control. The rubric is cached as a stable system prefix.
        # No server-side refusal fallback: every Claude label comes from the registered
        # model, and a refusal is a missing answer (consensus is then two of two).
        resp = a_client.messages.create(
            model=CODERS['claude']['model'],
            max_tokens=4000,
            system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
            output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}},
            messages=[{'role': 'user', 'content': prompt}],
        )
        if resp.stop_reason == 'refusal':
            return '', resp.model
        return next((b.text for b in resp.content if b.type == 'text'), ''), resp.model

    def call_gpt(system: str, prompt: str) -> Tuple[str, str]:
        resp = o_client.chat.completions.create(
            model=CODERS['gpt']['model'],
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
            max_completion_tokens=3000,     # includes reasoning tokens
            response_format={'type': 'json_object'},
        )
        return resp.choices[0].message.content or '', resp.model

    def call_gemini(system: str, prompt: str) -> Tuple[str, str]:
        resp = g_client.models.generate_content(
            model=CODERS['gemini']['model'],
            contents=prompt,
            config={'system_instruction': system, 'max_output_tokens': 3000,   # includes thinking tokens
                    'response_mime_type': 'application/json'},
        )
        return resp.text or '', getattr(resp, 'model_version', None) or CODERS['gemini']['model']

    return {'claude': call_claude, 'gpt': call_gpt, 'gemini': call_gemini}


def call_with_retries(fn: Callable[[], Tuple[str, str]], retries: int = 3,
                      base_delay: float = 2.0) -> Optional[Tuple[str, str]]:
    for attempt in range(retries):
        try:
            return fn()
        except Exception:                      # transient API failures; the unit is retried
            if attempt == retries - 1:
                return None
            time.sleep(base_delay ** (attempt + 1))
    return None


# ----------------------------------------------------------------------------
# Consensus and reliability
# ----------------------------------------------------------------------------
def consensus(votes: Dict[str, Optional[List[str]]]) -> List[str]:
    """Labels given by at least two of three coders, or by both of two when one
    coder's answer is missing. Vocabulary labels outrank category codes."""
    present = [v for v in votes.values() if v]
    if len(present) < 2:
        return [NO_CONSENSUS]
    need = 2
    counts: Dict[str, int] = {}
    for v in present:
        for lab in set(v):
            counts[lab] = counts.get(lab, 0) + 1
    if len(present) == 2:
        kept = [l for l, c in counts.items() if c == 2]
    else:
        kept = [l for l, c in counts.items() if c >= need]
    vocab_kept = sorted(l for l in kept if l not in CATEGORY_CODES)
    if vocab_kept:
        return vocab_kept
    codes = [l for l in kept if l in CATEGORY_CODES]
    return codes[:1] if codes else [NO_CONSENSUS]


def any_coder_labels(votes: Dict[str, Optional[List[str]]]) -> List[str]:
    """Liberal coding for the sensitivity analysis: every vocabulary label that any
    coder gave. A unit with none is NO_CONSENSUS (it counts as not produced)."""
    labs = sorted({l for v in votes.values() if v for l in v if l not in CATEGORY_CODES})
    return labs or [NO_CONSENSUS]


def primary_label(labels: Optional[List[str]]) -> Optional[str]:
    """One label per unit for agreement statistics: the alphabetically first
    vocabulary label, else the category code."""
    if not labels:
        return None
    vocab_labels = sorted(l for l in labels if l not in CATEGORY_CODES)
    return vocab_labels[0] if vocab_labels else labels[0]


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float('nan')
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = {}, {}
    for x, y in pairs:
        ca[x] = ca.get(x, 0) + 1; cb[y] = cb.get(y, 0) + 1
    pe = sum(ca.get(k, 0) * cb.get(k, 0) for k in set(ca) | set(cb)) / (n * n)
    return float('nan') if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[Optional[str]]]) -> float:
    """Units with a label from every rater only (complete cases)."""
    units = [r for r in ratings if all(x is not None for x in r)]
    if not units:
        return float('nan')
    m = len(units[0]); N = len(units)
    cats = sorted({x for r in units for x in r})
    counts = np.array([[sum(1 for x in r if x == c) for c in cats] for r in units], dtype=float)
    P_i = (np.sum(counts ** 2, axis=1) - m) / (m * (m - 1))
    p_j = counts.sum(axis=0) / (N * m)
    P_bar, P_e = P_i.mean(), np.sum(p_j ** 2)
    return float('nan') if P_e == 1 else float((P_bar - P_e) / (1 - P_e))


def krippendorff_alpha_nominal(ratings: Sequence[Sequence[Optional[str]]]) -> float:
    """Nominal alpha; missing labels allowed (units need at least two labels)."""
    units = [[x for x in r if x is not None] for r in ratings]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float('nan')
    cats = sorted({x for u in units for x in u})
    ix = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[ix[u[a]], ix[u[b]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1); n = n_c.sum()
    if n <= 1:
        return float('nan')
    D_o = (n - np.trace(o)) / n
    D_e = (n * n - np.sum(n_c ** 2)) / (n * (n - 1))
    return float('nan') if D_e == 0 else float(1 - D_o / D_e)


# ----------------------------------------------------------------------------
# Quality flags for exclusion rule 6
# ----------------------------------------------------------------------------
_ECHO = {normalize(s) for s in INSTRUCTION_EXAMPLES + list(BOX_LABELS.values()) + BOX_HINTS + GENRE_NAMES}


def is_instruction_echo(entry_text: str) -> bool:
    return normalize(entry_text) in _ECHO


def quality_flags(units_coded: pd.DataFrame) -> pd.DataFrame:
    """Per participant and genre: share of entries whose units are all
    UNINTERPRETABLE, and share of entries that only echo the instructions.
    A participant is flagged under exclusion rule 6 if either share is at least
    one half in either genre."""
    e = (units_coded.groupby(['participant_id', 'slot', 'genre', 'entry_order'])
         .agg(entry_text=('entry_text', 'first'),
              all_uninterp=('labels', lambda s: all(l == ['UNINTERPRETABLE'] for l in s)))
         .reset_index())
    e['echo'] = e['entry_text'].map(is_instruction_echo)
    g = (e.groupby(['participant_id', 'slot', 'genre'])
         .agg(n_entries=('entry_order', 'size'),
              share_uninterpretable=('all_uninterp', 'mean'),
              share_instruction_echo=('echo', 'mean'))
         .reset_index())
    g['flag'] = (g['share_uninterpretable'] >= 0.5) | (g['share_instruction_echo'] >= 0.5)
    return g


# ----------------------------------------------------------------------------
# Production indicators on the registered analysis set
# ----------------------------------------------------------------------------
GENRE_KEY_TO_DISPLAY = {'western': 'Western', 'horror': 'Horror', 'science_fiction': 'Science Fiction',
                        'crime': 'Crime', 'comedy': 'Comedy', 'fantasy': 'Fantasy'}


def produced_elements(units_coded: pd.DataFrame) -> pd.DataFrame:
    """One row per participant, genre and vocabulary label produced (counted once)."""
    rows = []
    for r in units_coded.itertuples(index=False):
        for lab in r.labels:
            if lab in CATEGORY_CODES or lab == NO_CONSENSUS:
                continue
            rows.append({'participant_id': r.participant_id, 'genre': GENRE_KEY_TO_DISPLAY.get(r.genre, r.genre),
                         'key': lab, 'box': r.box, 'entry_order': r.entry_order})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return (d.sort_values('entry_order')
             .drop_duplicates(['participant_id', 'genre', 'key'])
             .reset_index(drop=True))


def production_matrix(produced: pd.DataFrame, pool: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Participant x element rows over the registered generation analysis set.
    `participants` has participant_id and genre (one row per participant-genre)."""
    a = pool[pool['in_generation_analysis_set'].astype(str).str.lower() == 'true'][['genre', 'key']]
    m = participants.merge(a, on='genre')
    hit = produced[['participant_id', 'genre', 'key']].assign(produced=1)
    m = m.merge(hit, on=['participant_id', 'genre', 'key'], how='left')
    m['produced'] = m['produced'].fillna(0).astype(int)
    return m


def outside_pool_report(produced: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Produced vocabulary labels that are not in the genre's analysis set."""
    a = pool[pool['in_generation_analysis_set'].astype(str).str.lower() == 'true'][['genre', 'key']].assign(in_set=True)
    m = produced.merge(a, on=['genre', 'key'], how='left')
    m['in_set'] = m['in_set'].fillna(False).astype(bool)
    return m.groupby('genre')['in_set'].agg(n='size', share_in_set='mean').reset_index()


# ----------------------------------------------------------------------------
# Human validation sample
# ----------------------------------------------------------------------------
def validation_sample(units: pd.DataFrame, n_per_genre: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Random units, stratified by genre, for the two human coders. Coder sheets
    show only unit_id, box and text."""
    parts = []
    for g, d in units.groupby('genre'):
        parts.append(d.sample(n=min(n_per_genre, len(d)), random_state=seed))
    s = pd.concat(parts).sort_values('unit_id').reset_index(drop=True)
    return s[['unit_id', 'genre', 'box', 'entry_text', 'unit_text']]


def coder_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """What each human coder sees: no genre, no model labels, random order."""
    sh = sample[['unit_id', 'box', 'entry_text', 'unit_text']].sample(frac=1, random_state=SEED + 1)
    return sh.assign(label_1='', label_2='', label_3='', note='').reset_index(drop=True)


# ----------------------------------------------------------------------------
# Secondary check S1: are the entries genre-specific?
# ----------------------------------------------------------------------------
GENRE_DISPLAY_TO_KEY = {v: k for k, v in GENRE_KEY_TO_DISPLAY.items()}


def genre_centroids(plot_E, plot_idx: pd.DataFrame, union: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Unit-length mean of the normalized voyage-4-large plot embeddings of each
    study genre's films (the Notebook 05 embeddings)."""
    g = union[['union_id', 'genres_canonical']].explode('genres_canonical')
    g = g[g['genres_canonical'].isin(GENRE_DISPLAY_TO_KEY)]
    g = g.merge(plot_idx, on='union_id')
    out = {}
    for disp, d in g.groupby('genres_canonical'):
        X = np.asarray(plot_E[np.sort(d['embedding_row_index'].to_numpy())], dtype=np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        c = X.mean(axis=0)
        out[GENRE_DISPLAY_TO_KEY[disp]] = c / np.linalg.norm(c)
    return out


def genre_discrimination(coded: pd.DataFrame, emb: Dict[str, np.ndarray],
                         centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per participant and genre slot: mean cosine of the participant's entries to
    the cued genre's centroid and to the centroid of the participant's other genre."""
    e = coded.drop_duplicates(['participant_id', 'slot', 'entry_order'])[['participant_id', 'slot', 'genre', 'entry_text']]
    other = e.groupby('participant_id')['genre'].agg(lambda s: sorted(set(s))).to_dict()
    rows = []
    for (pid, slot, genre), d in e.groupby(['participant_id', 'slot', 'genre']):
        others = [g for g in other[pid] if g != genre]
        if not others:
            continue
        V = [emb[sc_t] for sc_t in d['entry_text'].map(normalize) if sc_t in emb]
        if not V:
            continue
        V = np.stack(V); V = V / np.linalg.norm(V, axis=1, keepdims=True)
        rows.append({'participant_id': pid, 'slot': slot, 'cued_genre': genre, 'other_genre': others[0],
                     'n_entries': len(V),
                     'cos_cued': float((V @ centroids[genre]).mean()),
                     'cos_other': float((V @ centroids[others[0]]).mean())})
    return pd.DataFrame(rows)
