"""Partie B — Reconnaissance d'entités nommées (NER) : 3 approches évaluées sur CoNLL-2003.

1. Règles  : spaCy EntityRuler = dictionnaire (gazetteer) construit sur le train + règles de motifs
             (titres de civilité, suffixes d'organisation, adjectifs de nationalité).
2. spaCy   : modèle statistique pré-entraîné en_core_web_sm (CNN, entraîné sur OntoNotes, pas sur
             CoNLL) -> ses 18 étiquettes sont projetées sur les 4 de CoNLL (PER, ORG, LOC, MISC).
3. CRF     : Conditional Random Field (sklearn-crfsuite) entraîné sur CoNLL-2003 avec des traits
             classiques (forme du mot, suffixes, POS, mots voisins).

Évaluation au niveau entité (une entité est correcte si frontières ET type sont exacts) avec seqeval.
"""
import pickle
import time
from collections import Counter, defaultdict

import sklearn_crfsuite
import spacy
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from spacy.tokens import Doc

from src.config import CONLL_DIR, PROCESSED_DIR

CONLL_TYPES = ["PER", "ORG", "LOC", "MISC"]

# projection des étiquettes OntoNotes (spaCy) vers CoNLL ; les autres (DATE, CARDINAL...) -> O
SPACY_TO_CONLL = {"PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
                  "NORP": "MISC", "EVENT": "MISC", "WORK_OF_ART": "MISC", "LAW": "MISC",
                  "LANGUAGE": "MISC", "PRODUCT": "MISC"}


# ---------------------------------------------------------------- données CoNLL
def load_conll(split):
    """Liste de phrases [(tokens, pos, tags)] ; les tags IOB1 du fichier sont convertis en IOB2."""
    sents, toks, pos, tags = [], [], [], []
    for line in (CONLL_DIR / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("-DOCSTART-"):
            if toks:
                sents.append((toks, pos, iob1_to_iob2(tags)))
                toks, pos, tags = [], [], []
            continue
        w, p, _, t = line.split()
        toks.append(w); pos.append(p); tags.append(t)
    if toks:
        sents.append((toks, pos, iob1_to_iob2(tags)))
    return sents


def iob1_to_iob2(tags):
    out = []
    for i, t in enumerate(tags):
        if t.startswith("I-") and (i == 0 or tags[i - 1] == "O" or tags[i - 1][2:] != t[2:]):
            out.append("B-" + t[2:])
        else:
            out.append(t)
    return out


def spans_from_tags(tags):
    """[(début, fin_exclue, type)] à partir d'une séquence IOB2."""
    spans, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):
        if start is not None and not t.startswith("I-"):
            spans.append((start, i, typ)); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            start, typ = i, t[2:]
    return spans


def tags_from_spans(n, spans):
    tags = ["O"] * n
    for s, e, typ in spans:
        tags[s] = "B-" + typ
        for i in range(s + 1, e):
            tags[i] = "I-" + typ
    return tags


# ---------------------------------------------------------------- 1. règles (EntityRuler)
TITLES = ["Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Dr", "Dr.", "President", "Prime", "Minister", "Chancellor",
          "Senator", "Governor", "King", "Queen", "Prince", "Pope", "General", "Coach", "Professor", "Sir"]
ORG_SUFFIXES = ["Inc", "Inc.", "Corp", "Corp.", "Ltd", "Ltd.", "Co", "Co.", "Group", "Bank", "Party", "Union",
                "Club", "Council", "Commission", "Association", "Ministry", "Agency", "University", "Institute",
                "Committee", "Federation", "Airlines", "Army", "Organisation", "Organization", "United", "City"]


def build_rule_ner(train_sents, min_count=1):
    """Construit un pipeline spaCy vide + EntityRuler (dictionnaire du train + motifs)."""
    counts = defaultdict(Counter)
    for toks, _, tags in train_sents:
        for s, e, typ in spans_from_tags(tags):
            counts[tuple(toks[s:e])][typ] += 1
    patterns = []
    for words, c in counts.items():
        label, n = c.most_common(1)[0]
        if n >= min_count and words[0][:1].isupper():
            patterns.append({"label": label, "pattern": [{"ORTH": w} for w in words]})
    cap = {"IS_TITLE": True}
    patterns += [
        # Titre + Nom propre -> PER (le titre est retiré ensuite, comme dans les annotations CoNLL)
        {"label": "PER", "pattern": [{"ORTH": {"IN": TITLES}}, {**cap, "OP": "+"}]},
        # Mots en majuscule + suffixe d'organisation -> ORG
        {"label": "ORG", "pattern": [{**cap, "OP": "+"}, {"ORTH": {"IN": ORG_SUFFIXES}}]},
        # Adjectifs de nationalité (German, Japanese, Swedish...) -> MISC
        {"label": "MISC", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]+(ian|ese|ish|ean)$"}}]},
    ]
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns(patterns)
    return nlp, len(patterns)


def rule_predict(nlp, tokens):
    doc = nlp(Doc(nlp.vocab, words=tokens))
    spans = []
    for ent in doc.ents:
        s = ent.start
        while s < ent.end - 1 and tokens[s] in TITLES:   # on retire "Mr.", "President"...
            s += 1
        spans.append((s, ent.end, ent.label_))
    return tags_from_spans(len(tokens), spans)


# ---------------------------------------------------------------- 2. spaCy pré-entraîné
def spacy_predict(nlp, tokens):
    """NER spaCy sur une phrase déjà tokenisée (on garde la tokenisation CoNLL pour l'évaluation)."""
    doc = nlp(Doc(nlp.vocab, words=tokens))
    spans = [(e.start, e.end, SPACY_TO_CONLL[e.label_]) for e in doc.ents if e.label_ in SPACY_TO_CONLL]
    return tags_from_spans(len(tokens), spans)


# ---------------------------------------------------------------- 3. CRF
def word2features(toks, pos, i):
    w = toks[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "w[-3:]": w[-3:], "w[-2:]": w[-2:], "w[:3]": w[:3],
         "w.isupper": w.isupper(), "w.istitle": w.istitle(), "w.isdigit": w.isdigit(),
         "w.has_hyphen": "-" in w, "w.shape": "".join("X" if c.isupper() else "x" if c.islower()
                                                       else "d" if c.isdigit() else c for c in w)[:6],
         "pos": pos[i], "pos[:2]": pos[i][:2]}
    for off in (-2, -1, 1, 2):                     # fenêtre de contexte de 2 mots
        j = i + off
        if 0 <= j < len(toks):
            f.update({f"{off}:w.lower": toks[j].lower(), f"{off}:w.istitle": toks[j].istitle(),
                      f"{off}:w.isupper": toks[j].isupper(), f"{off}:pos": pos[j]})
        else:
            f[f"{off}:pad"] = True
    return f


def sent2features(toks, pos):
    return [word2features(toks, pos, i) for i in range(len(toks))]


def train_crf(train_sents, c1=0.1, c2=0.05, max_iter=100):
    X = [sent2features(t, p) for t, p, _ in train_sents]
    y = [tags for _, _, tags in train_sents]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter,
                               all_possible_transitions=True)
    crf.fit(X, y)
    return crf


def get_crf(train_sents, path=PROCESSED_DIR / "crf_conll.pkl"):
    """Entraîne le CRF (1 à 2 min sur CPU) ou recharge le modèle sauvegardé."""
    if path.exists():
        return pickle.loads(path.read_bytes())
    crf = train_crf(train_sents)
    path.write_bytes(pickle.dumps(crf))
    return crf


# ---------------------------------------------------------------- évaluation
def evaluate(name, gold, pred, seconds):
    return {"model": name, "precision": precision_score(gold, pred), "recall": recall_score(gold, pred),
            "f1": f1_score(gold, pred), "time_s": round(seconds, 1)}


def run_all(test_sents, rule_nlp, spacy_nlp, crf):
    """Applique les 3 approches au test. Retourne prédictions + temps d'exécution."""
    preds, times = {}, {}
    t = time.time(); preds["Règles (EntityRuler)"] = [rule_predict(rule_nlp, s[0]) for s in test_sents]
    times["Règles (EntityRuler)"] = time.time() - t
    t = time.time(); preds["spaCy en_core_web_sm"] = [spacy_predict(spacy_nlp, s[0]) for s in test_sents]
    times["spaCy en_core_web_sm"] = time.time() - t
    t = time.time(); preds["CRF (sklearn-crfsuite)"] = crf.predict([sent2features(s[0], s[1]) for s in test_sents])
    times["CRF (sklearn-crfsuite)"] = time.time() - t
    return preds, times


def per_type_report(gold, pred):
    return classification_report(gold, pred, output_dict=True, zero_division=0)


# ---------------------------------------------------------------- application au corpus Wikipedia
def crf_on_spacy_doc(crf, doc):
    """Applique le CRF à un Doc spaCy (tokens + tags PTB de spaCy). Retourne [(texte, type)]."""
    out = []
    for sent in doc.sents:
        toks = [t.text for t in sent]
        pos = [t.tag_ for t in sent]
        tags = crf.predict_single(sent2features(toks, pos))
        out += [(" ".join(toks[s:e]), typ) for s, e, typ in spans_from_tags(list(tags))]
    return out


def agreement(spacy_docs, crf):
    """Accord entre spaCy et le CRF sur le corpus Wikipedia (pas d'annotation de référence ici) :
    F1 en prenant spaCy comme pseudo-référence, sur les types communs PER / ORG / LOC."""
    ref, hyp = Counter(), Counter()   # clé = (n° du document, texte de l'entité, type)
    for i, doc in enumerate(spacy_docs):
        for e in doc.ents:
            if SPACY_TO_CONLL.get(e.label_) in ("PER", "ORG", "LOC"):
                ref[(i, e.text.replace(" ", ""), SPACY_TO_CONLL[e.label_])] += 1
        for text, typ in crf_on_spacy_doc(crf, doc):
            if typ in ("PER", "ORG", "LOC"):
                hyp[(i, text.replace(" ", ""), typ)] += 1
    common = sum((ref & hyp).values())
    p, r = common / max(sum(hyp.values()), 1), common / max(sum(ref.values()), 1)
    return {"entités spaCy": sum(ref.values()), "entités CRF": sum(hyp.values()), "communes": common,
            "accord_precision": p, "accord_recall": r, "accord_f1": 2 * p * r / max(p + r, 1e-9)}
