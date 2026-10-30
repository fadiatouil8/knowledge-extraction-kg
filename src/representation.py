"""Partie A — Représentation numérique du texte.

- TF-IDF (scikit-learn) sur le texte normalisé (lemmes).
- Embeddings Transformer : modèle sentence-transformers/all-MiniLM-L6-v2 (BERT distillé, 22 M de
  paramètres, ~90 Mo, rapide sur CPU), chargé directement avec Hugging Face Transformers.
- Exploitation non supervisée : clustering K-Means + recherche de documents similaires.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEED

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tfidf_matrix(texts, max_features=2000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    return X, vec


def top_terms(X, vec, row, k=8):
    """Termes de plus fort poids TF-IDF pour un document."""
    scores = X[row].toarray().ravel()
    terms = np.array(vec.get_feature_names_out())
    return list(terms[scores.argsort()[::-1][:k]])


def bert_embeddings(texts, batch_size=16):
    """Embeddings de phrases (mean pooling + normalisation L2), sur CPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1)        # moyenne sur les vrais tokens
            out.append(torch.nn.functional.normalize(emb, dim=1).numpy())
    return np.vstack(out)


def choose_k(X, k_range=range(2, 9)):
    """Choisit le nombre de clusters avec le score silhouette."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        scores[k] = silhouette_score(X, labels, metric="cosine")
    return max(scores, key=scores.get), scores


def kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)


def cluster_top_terms(X_tfidf, vec, labels, k=8):
    """Termes TF-IDF moyens les plus forts par cluster (pour interpréter les groupes)."""
    terms = np.array(vec.get_feature_names_out())
    res = {}
    for c in sorted(set(labels)):
        mean = np.asarray(X_tfidf[labels == c].mean(axis=0)).ravel()
        res[c] = list(terms[mean.argsort()[::-1][:k]])
    return res


def most_similar(X, idx, titles, k=5):
    sims = cosine_similarity(X[idx:idx + 1], X).ravel()
    order = [i for i in sims.argsort()[::-1] if i != idx][:k]
    return [(titles[i], round(float(sims[i]), 3)) for i in order]
