"""Collecte des données du projet (à lancer une seule fois, résultats mis en cache dans data/raw).

1. Corpus : introductions des articles Wikipedia (EN) des lauréats du prix Nobel de physique
   (liste obtenue via Wikidata, échantillon de N_DOCS personnes, graine fixe).
2. Vérité terrain pour les relations : faits Wikidata (lieu/date de naissance, lieu de décès,
   universités, employeurs, prix) des mêmes personnes.
3. Corpus annoté CoNLL-2003 (anglais) pour entraîner le CRF et évaluer la NER.

Usage : python -m src.data_collection
"""
import json
import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import CONLL_DIR, CORPUS_FILE, N_DOCS, SEED, WIKIDATA_FILE

HEADERS = {"User-Agent": "KG-Master-Student-Project/1.0 (educational use)"}
SPARQL_URL = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"
CONLL_BASE = "https://raw.githubusercontent.com/glample/tagger/master/dataset/"
CONLL_FILES = {"train": "eng.train", "dev": "eng.testa", "test": "eng.testb"}

# session HTTP avec nouvelles tentatives automatiques (les API Wikimedia coupent parfois)
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2,
                                                        status_forcelist=[429, 500, 502, 503, 504])))

# relation du projet -> propriété Wikidata
WIKIDATA_PROPS = {
    "born_in": "P19",
    "died_in": "P20",
    "educated_at": "P69",
    "worked_at": "P108",
    "award_received": "P166",
}


def sparql(query):
    r = SESSION.get(SPARQL_URL, params={"query": query, "format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def get_laureates():
    """Liste (QID, titre Wikipedia) des lauréats du Nobel de physique."""
    q = """
    SELECT DISTINCT ?p ?article WHERE {
      ?p wdt:P166 wd:Q38104 ; wdt:P31 wd:Q5 .
      ?article schema:about ?p ; schema:isPartOf <https://en.wikipedia.org/> .
    }"""
    rows = sparql(q)
    people = []
    for row in rows:
        qid = row["p"]["value"].rsplit("/", 1)[-1]
        title = requests.utils.unquote(row["article"]["value"].rsplit("/wiki/", 1)[-1]).replace("_", " ")
        people.append((qid, title))
    return sorted(set(people))


def get_intros(titles):
    """Texte brut de l'introduction des articles (20 titres par requête, limite de l'API)."""
    texts = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        params = {"action": "query", "format": "json", "prop": "extracts", "exintro": 1,
                  "explaintext": 1, "exlimit": 20, "redirects": 1, "titles": "|".join(batch)}
        data = SESSION.get(WIKI_API, params=params, timeout=60).json()
        # gestion des redirections : on revient au titre demandé
        redirects = {r["to"]: r["from"] for r in data["query"].get("redirects", [])}
        for page in data["query"]["pages"].values():
            title = redirects.get(page["title"], page["title"])
            texts[title] = page.get("extract", "")
        time.sleep(0.5)
    return texts


def get_wikidata_facts(qids):
    """Faits Wikidata : {qid: {"label", "aliases", relation: [{"label", "aliases"}] ou [année]}}."""
    facts = {q: {r: [] for r in list(WIKIDATA_PROPS) + ["born_on"]} for q in qids}
    for i in range(0, len(qids), 50):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 50])
        # nom et alias des personnes
        rows = sparql(f"""
        SELECT ?p ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
          VALUES ?p {{ {values} }}
          ?p rdfs:label ?label FILTER(lang(?label)="en")
          OPTIONAL {{ ?p skos:altLabel ?alias FILTER(lang(?alias)="en") }}
        }} GROUP BY ?p ?label""")
        for row in rows:
            q = row["p"]["value"].rsplit("/", 1)[-1]
            facts[q]["label"] = row["label"]["value"]
            facts[q]["aliases"] = [a for a in row["aliases"]["value"].split("|") if a]
        # date de naissance (on garde l'année)
        rows = sparql(f"""
        SELECT ?p ?d WHERE {{ VALUES ?p {{ {values} }} ?p wdt:P569 ?d . }}""")
        for row in rows:
            q = row["p"]["value"].rsplit("/", 1)[-1]
            facts[q]["born_on"].append(row["d"]["value"][:4])
        # relations entité -> entité
        for rel, prop in WIKIDATA_PROPS.items():
            rows = sparql(f"""
            SELECT ?p ?o ?oLabel (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
              VALUES ?p {{ {values} }}
              ?p wdt:{prop} ?o .
              ?o rdfs:label ?oLabel FILTER(lang(?oLabel)="en")
              OPTIONAL {{ ?o skos:altLabel ?alias FILTER(lang(?alias)="en") }}
            }} GROUP BY ?p ?o ?oLabel""")
            for row in rows:
                q = row["p"]["value"].rsplit("/", 1)[-1]
                facts[q][rel].append({"label": row["oLabel"]["value"],
                                      "aliases": [a for a in row["aliases"]["value"].split("|") if a]})
            time.sleep(1)
    return facts


def download_conll():
    CONLL_DIR.mkdir(parents=True, exist_ok=True)
    for split, fname in CONLL_FILES.items():
        path = CONLL_DIR / f"{split}.txt"
        if not path.exists():
            r = SESSION.get(CONLL_BASE + fname, timeout=120)
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
            print(f"CoNLL-2003 {split} téléchargé ({len(r.text) // 1024} Ko)")


def main():
    download_conll()
    if CORPUS_FILE.exists() and WIKIDATA_FILE.exists():
        print("Corpus déjà présent, rien à faire.")
        return
    people = get_laureates()
    print(f"{len(people)} lauréats trouvés sur Wikidata")
    random.Random(SEED).shuffle(people)
    intros = get_intros([t for _, t in people[:N_DOCS + 20]])  # marge si certains textes sont vides
    corpus = []
    for qid, title in people:
        text = intros.get(title, "")
        if len(text.split()) >= 40:  # on écarte les introductions trop courtes
            corpus.append({"id": qid, "title": title, "text": text,
                           "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")})
        if len(corpus) == N_DOCS:
            break
    CORPUS_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(corpus)} textes sauvegardés dans {CORPUS_FILE.name}")

    facts = get_wikidata_facts([d["id"] for d in corpus])
    WIKIDATA_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Faits Wikidata sauvegardés dans {WIKIDATA_FILE.name}")


if __name__ == "__main__":
    main()
