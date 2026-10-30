# Knowledge Extraction from Unstructured Text — Knowledge Graph des lauréats du Nobel de physique

Projet de Master 2 Informatique : extraire des connaissances structurées (entités, relations, faits)
à partir de textes bruts, construire un **Knowledge Graph**, puis comparer une chaîne de traitement
**classique** (règles + apprentissage automatique) avec un **LLM**.

- **Corpus** : 150 introductions d'articles Wikipedia (anglais) de lauréats du prix Nobel de physique (~21 500 mots).
- **Références pour l'évaluation** : CoNLL-2003 (NER) et Wikidata (relations).
- **100 % CPU**, bibliothèques légères, aucune clé API obligatoire.

## Résultats principaux

| Tâche | Résultat |
|---|---|
| NER sur CoNLL-2003 (test, F1 entité) | CRF **0,814** · règles (EntityRuler) 0,613 · spaCy `en_core_web_sm` 0,595 |
| Relations vs Wikidata (approche classique) | précision 0,753 · rappel 0,762 · **F1 0,757** |
| Knowledge Graph classique | 492 nœuds, 699 arcs, exporté en RDF/Turtle, GraphML, CSV, HTML interactif |
| Relations vs Wikidata (LLM Qwen2.5-1.5B, CPU) | précision 0,544 · rappel 0,636 · **F1 0,586** · 89 % des objets ancrés dans le texte |
| Temps de calcul (150 textes) | classique 2,5 s · LLM ~72 min |

Détails et discussion : [report/rapport.pdf](report/rapport.pdf) (4 pages) et le notebook.

![Comparaison classique vs LLM](outputs/figures/D_classic_vs_llm.png)

## Structure

```
├── notebooks/knowledge_extraction.ipynb   # notebook principal, exécuté (parties A à D)
├── src/
│   ├── config.py               # chemins et constantes
│   ├── data_collection.py      # téléchargement Wikipedia + Wikidata + CoNLL-2003
│   ├── preprocessing.py        # A : nettoyage, tokenisation, POS, lemmatisation (NLTK / spaCy)
│   ├── representation.py       # A : TF-IDF, embeddings MiniLM, K-Means, similarité
│   ├── ner.py                  # B : NER par règles (EntityRuler), spaCy, CRF + évaluation seqeval
│   ├── relation_extraction.py  # C : extraction de relations par analyse en dépendances
│   ├── kg.py                   # C/D : KG NetworkX, exports RDF/GraphML/CSV, visualisation, QA, recommandation
│   ├── evaluation.py           # évaluation des triplets avec Wikidata
│   └── llm_extraction.py       # D : extraction par LLM (Qwen2.5 local ou API compatible OpenAI) avec cache
├── data/
│   ├── raw/wiki_nobel_physics.json      # corpus (150 textes)
│   ├── raw/wikidata_facts.json          # faits de référence
│   └── processed/llm_cache_*.json       # réponses du LLM (évite de relancer 1 h 30 de calcul)
├── outputs/
│   ├── figures/                # graphiques et visualisations du KG (PNG/SVG)
│   ├── kg/                     # KG exportés : .ttl (RDF), .graphml, _triples.csv, _interactive.html
│   └── results/                # tableaux de résultats (CSV)
├── report/rapport.md, rapport.pdf       # rapport (5 pages max)
└── presentation/                        # support et préparation de l'oral
```

## Installation

Prérequis : Python 3.10 à 3.12, ~4 Go d'espace disque (dont ~3 Go pour le LLM local, optionnel).

```bash
python -m venv .venv
```

Activer l'environnement (Windows : `.venv\Scripts\activate`, Linux/macOS : `source .venv/bin/activate`), puis :

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

## Exécution

1. **Données** (déjà fournies pour Wikipedia/Wikidata ; télécharge CoNLL-2003, ~5 Mo) :

```bash
python -m src.data_collection
```

2. **Notebook** : ouvrir `notebooks/knowledge_extraction.ipynb` et tout exécuter (~3 min sur CPU), ou :

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/knowledge_extraction.ipynb
```

3. **(Optionnel) Relancer le LLM** : les réponses sont déjà en cache dans `data/processed/`. Pour les régénérer
   avec le modèle local (Qwen2.5-1.5B-Instruct, ~3 Go téléchargés, ~1 h 30 sur un CPU 8 cœurs, 8 Go de RAM) :

```bash
python -m src.llm_extraction --n-docs 150 --n-sents 100
```

   Ou avec une API gratuite compatible OpenAI, par exemple Groq (clé gratuite sur console.groq.com) ou Ollama :

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=<votre_clé> LLM_MODEL=llama-3.1-8b-instant python -m src.llm_extraction --backend openai
```

   Dans ce cas, mettre `LLM_MODEL = "llama-3.1-8b-instant"` dans la cellule D.1 du notebook.

4. **Rapport PDF** (Markdown → PDF via Edge/Chrome headless) :

```bash
python report/build_report.py
```

## Limites connues

- Le LLM utilisé est petit (1,5 milliard de paramètres) pour tenir sur CPU avec 8 Go de RAM ; un modèle plus gros
  (via Groq ou Ollama) donnerait de meilleurs résultats, mais n'a pas été évalué ici.
- La précision mesurée avec Wikidata est une borne inférieure (Wikidata est incomplet).
- La coréférence est résolue par une heuristique (pronoms → personne de l'article).
- Pas de liage d'entités : « University of California » et « UC Berkeley » restent deux nœuds distincts.

## Sources des données

Voir [data/LICENSE_DATA.md](data/LICENSE_DATA.md) (Wikipedia CC BY-SA 4.0, Wikidata CC0, CoNLL-2003 recherche uniquement).
