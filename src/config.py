"""Chemins et constantes partagés par tous les modules du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONLL_DIR = RAW_DIR / "conll2003"

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
KG_DIR = OUTPUT_DIR / "kg"

CORPUS_FILE = RAW_DIR / "wiki_nobel_physics.json"
WIKIDATA_FILE = RAW_DIR / "wikidata_facts.json"

SEED = 42
N_DOCS = 150

# Les 6 relations visées dans le KG (schéma commun pour l'approche classique et le LLM)
RELATIONS = ["born_in", "born_on", "died_in", "educated_at", "worked_at", "award_received"]

for d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, KG_DIR):
    d.mkdir(parents=True, exist_ok=True)
