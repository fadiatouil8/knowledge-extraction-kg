"""Partie D — Extraction d'entités et de relations avec un LLM (approche générative, sans entraînement).

Deux backends gratuits :
- "hf"     : modèle local Hugging Face exécuté sur CPU (par défaut Qwen2.5-1.5B-Instruct, ~3 Go).
             Aucune clé API, mais lent sur CPU (quelques tokens/seconde).
- "openai" : n'importe quelle API compatible OpenAI, par exemple
               * Groq (gratuit avec une clé) : LLM_BASE_URL=https://api.groq.com/openai/v1
                 LLM_MODEL=llama-3.1-8b-instant  LLM_API_KEY=<clé>
               * Ollama en local : LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b

Les sorties brutes sont mises en cache (data/processed/llm_cache_*.json) : le notebook relit ce cache
et n'a donc pas besoin de relancer le LLM. Le script est reprenable s'il est interrompu.

Usage : python -m src.llm_extraction --n-docs 150 --n-sents 100
"""
import argparse
import json
import os
import re
import time

from src.config import CORPUS_FILE, PROCESSED_DIR
from src.relation_extraction import ALL_RELATIONS, display_name

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

KG_SYSTEM = """You are an information extraction system. Read a short biography and extract facts about the MAIN person only.
Return ONLY a JSON object with exactly these keys:
"born_in": city or country where the person was born,
"born_on": date of birth,
"died_in": city or country where the person died,
"educated_at": universities or schools the person studied at,
"worked_at": universities, laboratories, institutes or companies where the person worked,
"award_received": prizes, medals and awards the person received,
"shared_prize_with": people who shared a prize with the person.
Each value is a list of short names copied from the text. Use [] when the text does not say it. Never guess."""

KG_EXAMPLE_IN = ("Main person: Jane Doe\nText: Jane Doe (born 3 May 1950 in Lyon) is a French physicist. "
                 "She studied at the University of Paris and worked at CERN. She shared the 2001 Nobel Prize "
                 "in Physics with John Smith and received the Dirac Medal.")
KG_EXAMPLE_OUT = json.dumps({"born_in": ["Lyon"], "born_on": ["3 May 1950"], "died_in": [],
                             "educated_at": ["University of Paris"], "worked_at": ["CERN"],
                             "award_received": ["Nobel Prize in Physics", "Dirac Medal"],
                             "shared_prize_with": ["John Smith"]})

NER_SYSTEM = """You are a named entity recognizer. Extract the named entities of the sentence.
Return ONLY a JSON object {"PER": [...], "ORG": [...], "LOC": [...], "MISC": [...]} where
PER = persons, ORG = organizations, companies, sports teams, LOC = countries, cities, regions,
MISC = nationalities, events, languages and other proper names. Copy entities exactly as written."""
NER_EXAMPLE_IN = "Sentence: German striker Jurgen Klinsmann joined Bayern Munich after the World Cup in Italy ."
NER_EXAMPLE_OUT = json.dumps({"PER": ["Jurgen Klinsmann"], "ORG": ["Bayern Munich"], "LOC": ["Italy"],
                              "MISC": ["German", "World Cup"]})


class LLMClient:
    def __init__(self, backend="hf", model=None):
        self.backend = backend
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model_name = model or DEFAULT_HF_MODEL
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            # bfloat16 divise la mémoire par 2 (1.5B -> ~3 Go de RAM), indispensable avec 8 Go
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.bfloat16).eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        else:
            self.model_name = model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
            self.base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
            self.api_key = os.environ.get("LLM_API_KEY", "ollama")

    def chat(self, messages, max_new_tokens=300):
        if self.backend == "hf":
            import torch
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        import requests
        r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model_name, "messages": messages, "temperature": 0,
                                "max_tokens": max_new_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def few_shot(system, ex_in, ex_out, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": ex_in},
            {"role": "assistant", "content": ex_out}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ analyse des sorties
def parse_json(raw):
    """Extrait le premier objet JSON de la réponse. Retourne (dict ou None, json_valide)."""
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None, False
    txt = m.group(0)
    for attempt in (txt, re.sub(r",\s*([}\]])", r"\1", txt)):   # 2e essai : virgules finales
        try:
            obj = json.loads(attempt)
            return (obj, True) if isinstance(obj, dict) else (None, False)
        except json.JSONDecodeError:
            continue
    return None, False


EMPTY_RE = re.compile(r"^(not mentioned|not specified|not stated|unknown|none|n/?a|null|-)\.?$", re.I)


def as_list(v):
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, list):        # ex. ["Eddington Medal", "Royal Astronomical Society", "1969"]
                x = next((y for y in x if isinstance(y, str)), "")
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def kg_triples_from_cache(cache, corpus):
    """Transforme les réponses brutes du LLM en triplets + statistiques de conformité."""
    triples, stats = [], {"docs": 0, "json_valide": 0, "clés_hors_schéma": 0, "valeurs_vides_écrites": 0,
                          "temps_total_s": 0.0}
    titles = {d["id"]: d["title"] for d in corpus}
    for qid, entry in cache.items():
        if qid not in titles:
            continue
        stats["docs"] += 1
        stats["temps_total_s"] += entry["seconds"]
        obj, ok = parse_json(entry["raw"])
        stats["json_valide"] += ok
        if not obj:
            continue
        stats["clés_hors_schéma"] += sum(1 for k in obj if k not in ALL_RELATIONS)
        name = display_name(titles[qid])
        for rel in ALL_RELATIONS:
            for o in dict.fromkeys(as_list(obj.get(rel, []))):
                if EMPTY_RE.match(o):                     # "Not mentioned" au lieu de [] : non conforme
                    stats["valeurs_vides_écrites"] += 1
                elif o.lower() != name.lower():
                    triples.append({"subject": name, "relation": rel, "object": o, "doc_id": qid,
                                    "sentence": "", "trigger": "LLM"})
    return triples, stats


def ner_tags_from_output(tokens, raw):
    """Convertit la réponse NER du LLM (listes de chaînes) en étiquettes IOB2 alignées sur les tokens."""
    from src.ner import tags_from_spans
    obj, ok = parse_json(raw)
    spans, used = [], set()
    if obj:
        for typ in ("PER", "ORG", "LOC", "MISC"):
            for ent in as_list(obj.get(typ, [])):
                target = ent.replace(" ", "")
                for i in range(len(tokens)):
                    acc = ""
                    for j in range(i, len(tokens)):
                        acc += tokens[j]
                        if acc == target:
                            if not used & set(range(i, j + 1)):
                                spans.append((i, j + 1, typ)); used |= set(range(i, j + 1))
                            break
                        if not target.startswith(acc):
                            break
    return tags_from_spans(len(tokens), spans), ok


# ------------------------------------------------------------------ exécution avec cache
def cache_path(task, model_name):
    return PROCESSED_DIR / f"llm_cache_{task}_{re.sub(r'[^A-Za-z0-9.]+', '_', model_name)}.json"


def load_cache(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_kg(client, corpus, n_docs):
    path = cache_path("kg", client.model_name)
    cache = load_cache(path)
    for k, d in enumerate(corpus[:n_docs]):
        if d["id"] in cache:
            continue
        user = f"Main person: {display_name(d['title'])}\nText: {d.get('clean', d['text'])}"
        t = time.time()
        raw = client.chat(few_shot(KG_SYSTEM, KG_EXAMPLE_IN, KG_EXAMPLE_OUT, user), max_new_tokens=300)
        cache[d["id"]] = {"raw": raw, "seconds": round(time.time() - t, 2)}
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[KG {k + 1}/{n_docs}] {d['title']} ({cache[d['id']]['seconds']} s)", flush=True)
    return cache


def run_ner(client, sents, n_sents):
    path = cache_path("ner", client.model_name)
    cache = load_cache(path)
    for k, (toks, _, gold) in enumerate(sents[:n_sents]):
        key = str(k)
        if key in cache:
            continue
        t = time.time()
        raw = client.chat(few_shot(NER_SYSTEM, NER_EXAMPLE_IN, NER_EXAMPLE_OUT, "Sentence: " + " ".join(toks)),
                          max_new_tokens=120)
        # tokens et étiquettes de référence stockés avec la réponse : l'évaluation est autonome
        cache[key] = {"raw": raw, "seconds": round(time.time() - t, 2), "tokens": toks, "gold": gold}
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[NER {k + 1}/{n_sents}] ({cache[key]['seconds']} s)", flush=True)
    return cache


def ner_sample(test_sents, n, seed=42):
    """Échantillon fixe de phrases du test CoNLL contenant au moins une entité (hors titres en MAJUSCULES)."""
    import random
    pool = [s for s in test_sents if any(t != "O" for t in s[2]) and len(s[0]) >= 6
            and not all(w.isupper() for w in s[0] if w.isalpha())]
    return random.Random(seed).sample(pool, n)


def main():
    from src.ner import load_conll
    from src.preprocessing import clean_text
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["hf", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-docs", type=int, default=150)
    ap.add_argument("--n-sents", type=int, default=100)
    ap.add_argument("--task", default="both", choices=["kg", "ner", "both"])
    args = ap.parse_args()
    client = LLMClient(args.backend, args.model)
    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    for d in corpus:
        d["clean"] = clean_text(d["text"])
    if args.task in ("ner", "both"):
        run_ner(client, ner_sample(load_conll("test"), args.n_sents), args.n_sents)
    if args.task in ("kg", "both"):
        run_kg(client, corpus, args.n_docs)


if __name__ == "__main__":
    main()
