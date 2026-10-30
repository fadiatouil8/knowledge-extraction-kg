"""Évaluation automatique des triplets avec Wikidata comme référence.

Wikidata n'est pas une annotation "parfaite" du texte, d'où deux mesures complémentaires :
- Précision (borne inférieure) : part des triplets extraits qui correspondent à un fait Wikidata
  de la personne. Un triplet absent de Wikidata n'est pas forcément faux (Wikidata est incomplet).
- Rappel "ancré dans le texte" : parmi les faits Wikidata dont l'objet est écrit dans le texte
  (libellé ou alias retrouvé tel quel), part retrouvée par l'extracteur. On n'exige donc pas de
  trouver des faits que le texte ne mentionne pas.
"""
import difflib
import re

import pandas as pd

from src.config import RELATIONS
from src.relation_extraction import display_name, strip_accents, year_of


def norm(s):
    s = strip_accents(s).lower()
    s = re.sub(r"\b(the|of|at|in|for|and)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def obj_match(pred, fact):
    """Le texte extrait correspond-il à un fait Wikidata (libellé ou alias) ?"""
    p = norm(pred)
    if not p:
        return False
    for cand in [fact["label"]] + fact.get("aliases", []):
        c = norm(cand)
        if not c:
            continue
        if p == c:
            return True
        short, long_ = sorted((p, c), key=len)
        if len(short) >= 4 and short in long_ and len(short) / len(long_) >= 0.5:
            return True
        if difflib.SequenceMatcher(None, p, c).ratio() >= 0.88:
            return True
    return False


def fact_in_text(fact, text_norm):
    for cand in [fact["label"]] + fact.get("aliases", []):
        c = norm(cand)
        if len(c) >= 4 and f" {c} " in f" {text_norm} ":
            return True
    return False


def evaluate_triples(triples, corpus, wikidata, doc_ids=None, relations=RELATIONS):
    """Retourne un DataFrame précision / rappel / F1 par relation (+ ligne TOTAL)."""
    doc_ids = set(doc_ids) if doc_ids is not None else {d["id"] for d in corpus}
    by_doc = {d["id"]: d for d in corpus if d["id"] in doc_ids}
    rows = []
    for rel in relations:
        n_pred = n_pred_ok = n_gold = n_gold_found = 0
        for qid, d in by_doc.items():
            name = display_name(d["title"]).lower()
            preds = [t["object"] for t in triples
                     if t["doc_id"] == qid and t["relation"] == rel and t["subject"].lower() == name]
            preds = list(dict.fromkeys(p.lower() for p in preds))         # sans doublons
            facts = wikidata[qid][rel]
            text = d.get("clean", d["text"])
            if rel == "born_on":
                gold_years = set(facts)
                ok = [p for p in preds if year_of(p) in gold_years]
                grounded = [y for y in gold_years if y in text]
                found = [y for y in grounded if any(year_of(p) == y for p in preds)]
            else:
                ok = [p for p in preds if any(obj_match(p, f) for f in facts)]
                tn = norm(text)
                grounded = [f for f in facts if fact_in_text(f, tn)]
                found = [f for f in grounded if any(obj_match(p, f) for p in preds)]
            n_pred += len(preds); n_pred_ok += len(ok)
            n_gold += len(grounded); n_gold_found += len(found)
        rows.append({"relation": rel, "extraits": n_pred, "corrects (Wikidata)": n_pred_ok,
                     "faits Wikidata dans le texte": n_gold, "retrouvés": n_gold_found})
    df = pd.DataFrame(rows)
    total = df.drop(columns="relation").sum()
    df.loc[len(df)] = {"relation": "TOTAL", **total.to_dict()}
    df["précision"] = (df["corrects (Wikidata)"] / df["extraits"].clip(lower=1)).round(3)
    df["rappel"] = (df["retrouvés"] / df["faits Wikidata dans le texte"].clip(lower=1)).round(3)
    df["F1"] = (2 * df["précision"] * df["rappel"] / (df["précision"] + df["rappel"]).clip(lower=1e-9)).round(3)
    return df


def groundedness(triples, corpus):
    """Part des triplets dont l'objet apparaît (normalisé) dans le texte source : mesure simple
    des hallucinations, surtout utile pour le LLM."""
    texts = {d["id"]: norm(d.get("clean", d["text"])) for d in corpus}
    if not triples:
        return 0.0
    ok = 0
    for t in triples:
        o = norm(t["object"])
        y = year_of(t["object"]) if t["relation"] == "born_on" else None
        if (y and y in texts[t["doc_id"]]) or (o and o in texts[t["doc_id"]]):
            ok += 1
    return ok / len(triples)
