"""Partie C — Extraction de relations par analyse en dépendances (approche "classique").

Principe (simple et explicable) :
1. On parcourt les entités de chaque phrase (NER spaCy) + les groupes nominaux contenant
   "Prize", "Medal", "Award"... (souvent mal typés par la NER).
2. Pour chaque entité, on remonte l'arbre de dépendances (ancêtres de sa tête syntaxique) et on
   s'arrête au premier mot "déclencheur" connu : born -> born_in / born_on, educate -> educated_at,
   professor -> worked_at, win -> award_received, etc.
3. Le sujet de la relation est le sujet grammatical (nsubj) du verbe gouverneur. Les pronoms
   he/she/who et les mentions du nom de famille sont rattachés à la personne de l'article
   (résolution de coréférence heuristique : chaque article parle d'une seule personne).
4. Le type de l'entité doit être compatible avec la relation (born_in -> lieu, worked_at -> organisation).
"""
import re
import unicodedata

from src.config import RELATIONS

# mots déclencheurs (lemmes spaCy) pour chaque relation
TRIGGERS = {
    "born": {"bear", "born", "birth", "birthplace", "native"},
    "died_in": {"die", "death"},
    "educated_at": {"study", "graduate", "educate", "attend", "student", "doctorate", "phd", "ph.d.",
                    "degree", "alumnus", "enrol", "enroll", "matriculate", "thesis", "doctoral", "bachelor",
                    "diploma", "undergraduate", "graduation", "education", "studies"},
    "worked_at": {"work", "professor", "join", "position", "faculty", "lecturer", "teach", "researcher",
                  "director", "employ", "appoint", "chair", "head", "scientist", "research", "career",
                  "postdoc", "postdoctoral", "staff", "emeritus", "physicist", "engineer", "serve",
                  "assistant", "investigator", "spokesperson", "lead", "found", "establish", "move"},
    "award_received": {"win", "receive", "award", "share", "recipient", "give", "laureate", "honor",
                       "honour", "earn", "co-recipient", "elect"},
}
DEGREES = {"phd", "ph.d.", "doctorate", "degree", "bachelor", "master", "diploma", "b.sc", "m.sc", "ba", "bs",
           "ms", "ma", "habilitation", "dphil"}
AWARD_RE = re.compile(r"\b(Prize|Medal|Award|Fellowship|Lecture|Lectureship|Order of|Légion|Legion of Honour|"
                      r"Knight|Wolf Prize|Breakthrough)\b")
ORG_HINT = re.compile(r"\b(University|Institute|College|Laborator(y|ies)|Lab|School|Academy|Observatory|"
                      r"Center|Centre|Corporation|Company|Society|CERN|IBM|Bell|Max Planck|ETH|MIT)\b")
# disciplines souvent étiquetées ORG par erreur ("Professor of Theoretical Physics")
FIELD_RE = re.compile(r"\b(Physics|Engineering|Chemistry|Mathematics|Sciences?|Astronomy|Electrical|Optics)\b")
PLACE_LABELS = {"GPE", "LOC"}
ORG_LABELS = {"ORG", "FAC"}
PRONOUNS = {"he", "she", "who", "his", "her", "him", "they"}
STOP_TITLE = {"sir", "jr.", "jr", "iii", "ii", "von", "van", "der", "de", "la", "le"}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canonical(text):
    """Forme canonique d'une entité (nœud du KG) : sans article initial, ponctuation de bord, années."""
    t = text.strip(" ,.;:\"'()")
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
    t = re.sub(r"'s$", "", t)
    t = re.sub(r"^\d{4}(?:\s*-\s*\d{2,4})?\s+", "", t)               # "1988 Nobel Prize" -> "Nobel Prize"
    t = re.sub(r"\bNobel Prize for (Physics|Chemistry)\b", r"Nobel Prize in \1", t)
    t = re.sub(r"\s+(for|with|in recognition)\s+(his|her|their|the)\b.*$", "", t)   # "... for his work"
    if t == "Nobel Prize" or t == "Nobel prize":
        t = "Nobel Prize in Physics"                                # corpus = lauréats du Nobel de physique
    return re.sub(r"\s+", " ", t).strip(" ,.")


def year_of(text):
    m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", text)
    return m.group(1) if m else None


def display_name(title):
    """Titre Wikipedia -> nom de la personne ("David Lee (physicist)" -> "David Lee")."""
    return re.sub(r"\s*\([^)]*\)$", "", title)


def name_tokens(title):
    toks = {strip_accents(w).lower().strip(",.") for w in title.replace("-", " ").split()}
    return {t for t in toks if len(t) > 2 and t not in STOP_TITLE}


class DependencyRelationExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    # ------------------------------------------------------------ sujet
    def _is_main(self, span_text, main_toks):
        toks = {strip_accents(w).lower().strip(",.'") for w in span_text.split()}
        return bool(toks & main_toks) or bool(toks & PRONOUNS)

    def _subject(self, token, main, main_toks, doc):
        """Sujet de la relation : nsubj du premier verbe ancêtre qui en possède un."""
        for anc in [token] + list(token.ancestors):
            subj = [c for c in anc.children if c.dep_ in ("nsubj", "nsubjpass")]
            if not subj:
                continue
            s = subj[0]
            if s.lower_ in ("who", "which", "that"):        # proposition relative : on continue de remonter
                continue
            span = doc[s.left_edge.i:s.right_edge.i + 1]
            if self._is_main(span.text, main_toks) or s.lower_ in PRONOUNS:
                return main
            ents = [e for e in doc.ents if e.start <= s.i < e.end and e.label_ == "PERSON"]
            return ents[0].text if ents else None
        return main   # pas de sujet explicite (ex. "Born in Belgium, ...") -> personne de l'article

    # ------------------------------------------------------------ relation
    @staticmethod
    def _trigger(head):
        """Premier ancêtre déclencheur. Retourne (relation_brute, token déclencheur)."""
        for anc in head.ancestors:
            lem = anc.lemma_.lower()
            for rel, words in TRIGGERS.items():
                if lem in words or anc.lower_ in words:
                    # "received his PhD from X" : receive + diplôme -> études, pas un prix
                    if rel == "award_received" and any(c.lower_ in DEGREES or c.lemma_.lower() in DEGREES
                                                       for c in anc.children if c.dep_ == "dobj"):
                        return "educated_at", anc
                    return rel, anc
        return None, None

    @staticmethod
    def _type_ok(rel, label, text):
        if rel in ("born_in", "died_in"):
            return label in PLACE_LABELS
        if rel in ("educated_at", "worked_at"):
            if AWARD_RE.search(text) or (FIELD_RE.search(text) and not ORG_HINT.search(text)):
                return False
            if len(text.split()) == 1 and not text.isupper() and not ORG_HINT.search(text):
                return False      # mot isolé ambigu ("Electrical") sauf sigle (CERN, UCSB)
            return label in ORG_LABELS or bool(ORG_HINT.search(text))
        if rel == "award_received":
            return bool(AWARD_RE.search(text))
        return False

    def _candidates(self, sent):
        """Entités de la phrase + groupes nominaux "prix" non couverts par la NER."""
        cands = [(e, e.label_) for e in sent.ents]
        covered = {i for e in sent.ents for i in range(e.start, e.end)}
        for chunk in sent.noun_chunks:
            if AWARD_RE.search(chunk.text) and not any(i in covered for i in range(chunk.start, chunk.end)):
                cands.append((chunk, "AWARD"))
        return cands

    def extract_doc(self, doc, title):
        """Triplets (sujet, relation, objet) d'un article, avec la phrase source comme provenance."""
        title = display_name(title)
        main_toks = name_tokens(title)
        triples = []
        for i, sent in enumerate(doc.sents):
            for span, label in self._candidates(sent):
                if label == "PERSON" and self._is_main(span.text, main_toks):
                    continue
                text = canonical(span.text)
                if not text:
                    continue
                # dates de naissance dans la parenthèse du début : "(11 May 1924 - 13 September 2021)"
                if label == "DATE" and i == 0 and re.search(r"\d{4}\s*-\s*.*\d{4}", span.text):
                    triples.append(self._t(title, "born_on", span.text.split("-")[0].strip(), sent))
                    continue
                if AWARD_RE.search(span.text) and label in ("WORK_OF_ART", "LAW", "EVENT", "PRODUCT", "ORG",
                                                             "AWARD", "PERSON", "NORP", "GPE"):
                    label = "AWARD"
                raw, trig = self._trigger(span.root)
                if label == "AWARD":
                    rel = "award_received"   # un prix rattaché à la personne, quel que soit le verbe
                elif raw == "born":
                    rel = "born_on" if label == "DATE" else "born_in"
                elif raw == "award_received" and label == "PERSON" and AWARD_RE.search(sent.text):
                    rel = "shared_prize_with"    # "He shared the Nobel Prize with X"
                else:
                    rel = raw
                if rel is None:
                    continue
                if rel == "born_on":
                    if label != "DATE" or not year_of(span.text):
                        continue
                elif rel == "shared_prize_with":
                    pass
                elif not self._type_ok(rel, label, span.text):
                    continue
                subj = self._subject(span.root, title, main_toks, doc)
                if subj is None or (rel == "shared_prize_with" and canonical(subj) == text):
                    continue
                triples.append(self._t(canonical(subj), rel, text if rel != "born_on" else span.text, sent,
                                       trig.text if trig is not None else ""))
        return dedup(triples)

    @staticmethod
    def _t(s, r, o, sent, trigger=""):
        return {"subject": s, "relation": r, "object": o.strip(" ,.()"), "sentence": sent.text,
                "trigger": trigger}

    def extract_corpus(self, corpus, docs):
        out = []
        for d, doc in zip(corpus, docs):
            for t in self.extract_doc(doc, d["title"]):
                t["doc_id"] = d["id"]
                out.append(t)
        return out


def dedup(triples):
    seen, out = set(), []
    for t in triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


ALL_RELATIONS = RELATIONS + ["shared_prize_with"]
