"""Partie A — Nettoyage, normalisation, tokenisation, POS tagging et lemmatisation.

Deux niveaux de prétraitement sont utilisés volontairement :
- clean_text()      : nettoyage *léger* (bruit Wikipedia, espaces, guillemets). On garde la casse
                      et la ponctuation car la NER et le parsing en dépendent ("Columbia University"
                      n'est reconnu que grâce aux majuscules).
- normalize_text()  : normalisation *forte* (minuscules, sans ponctuation ni mots vides, lemmes)
                      utilisée uniquement pour la représentation vectorielle (TF-IDF, clustering).
"""
import re
import unicodedata

import nltk
import spacy

_NLP = None


def get_nlp():
    """Charge le modèle spaCy une seule fois (tokeniseur, POS, lemmes, dépendances, NER)."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def ensure_nltk():
    """Télécharge les petites ressources NLTK nécessaires (quelques Mo, une seule fois)."""
    for res, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


def clean_text(text):
    """Nettoyage léger d'une introduction Wikipedia."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)          # renvois de notes
    text = re.sub(r"(?<=[\s(])/[^/\d]{1,40}?/(?=[\s,;)])", "", text)    # transcriptions API /ˈmaks/
    text = re.sub(r"\((?:listen|pronunciation)[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\[[^\]]*\]", "", text)                             # API entre crochets, [O.S. ...]
    # annotations de langue "(German: ... ;", "; romanized: ... ;" en début d'article
    text = re.sub(r"([(;])\s*(?:[A-Z][a-z]+ )?(?:[A-Z][a-z]+|romanized)(?: pronunciation)?:[^;()]*;?",
                  r"\1", text)
    text = re.sub(r"\b[A-Z]{2}:\s*[,;]\s*", "", text)                  # restes "US: , UK: ;"
    text = re.sub(r"\(\s*[;,]?\s*", "(", text)                         # "( ; born ..." -> "(born ..."
    text = re.sub(r"\(\s*\)", "", text)                                # parenthèses vides
    text = text.replace("–", "-").replace("—", "-")          # tirets
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s+", " ", text)                                   # espaces multiples / sauts de ligne
    return text.strip()


def nltk_pipeline(text):
    """Tokenisation, POS tagging et lemmatisation avec NLTK (pour comparaison avec spaCy)."""
    ensure_nltk()
    from nltk.stem import WordNetLemmatizer
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    # WordNet a besoin de la catégorie grammaticale pour bien lemmatiser ("was" -> "be")
    wn_pos = {"J": "a", "V": "v", "N": "n", "R": "r"}
    lemmas = [lemmatizer.lemmatize(w.lower(), wn_pos.get(t[0], "n")) for w, t in tags]
    return [(w, t, l) for (w, t), l in zip(tags, lemmas)]


def spacy_table(doc, n=None):
    """Tableau token / POS universel / tag fin / lemme / dépendance pour un Doc spaCy."""
    rows = [{"token": t.text, "pos": t.pos_, "tag": t.tag_, "lemma": t.lemma_, "dep": t.dep_,
             "head": t.head.text, "stop": t.is_stop} for t in doc]
    return rows[:n] if n else rows


def normalize_doc(doc):
    """Normalisation forte : lemmes en minuscules, sans mots vides, ponctuation, nombres."""
    return " ".join(t.lemma_.lower() for t in doc
                    if t.is_alpha and not t.is_stop and len(t) > 2)


def preprocess_corpus(corpus):
    """Applique tout le prétraitement au corpus. Retourne (corpus enrichi, liste de Doc spaCy)."""
    nlp = get_nlp()
    for d in corpus:
        d["clean"] = clean_text(d["text"])
    docs = list(nlp.pipe([d["clean"] for d in corpus], batch_size=32))
    for d, doc in zip(corpus, docs):
        d["normalized"] = normalize_doc(doc)
        d["n_tokens"] = len(doc)
        d["n_sents"] = len(list(doc.sents))
    return corpus, docs
