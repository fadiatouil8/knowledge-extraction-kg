"""Partie C — Construction, export, visualisation et interrogation du Knowledge Graph.

Le KG est un MultiDiGraph NetworkX : nœuds = entités typées, arcs = relations (avec la phrase
source comme provenance). Exports : CSV des triplets, RDF Turtle (rdflib), GraphML (Gephi/Neo4j),
HTML interactif (pyvis).
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef

from src.relation_extraction import strip_accents, year_of

OBJECT_TYPE = {"born_in": "Place", "died_in": "Place", "born_on": "Year", "educated_at": "Organization",
               "worked_at": "Organization", "award_received": "Award", "shared_prize_with": "Person"}
TYPE_COLOR = {"Person": "#4C72B0", "Place": "#55A868", "Organization": "#DD8452", "Award": "#C44E52",
              "Year": "#8C8C8C"}
EX = Namespace("http://example.org/nobel-kg/")
COMMON = "Nobel Prize in Physics"   # partagé par tout le corpus : peu informatif pour comparer/visualiser


def node_label(rel, obj):
    """Les dates de naissance sont regroupées par année pour relier les personnes nées la même année."""
    return year_of(obj) if rel == "born_on" else obj


def build_kg(triples):
    G = nx.MultiDiGraph()
    for t in triples:
        s, rel = t["subject"], t["relation"]
        o = node_label(rel, t["object"])
        if not o:
            continue
        if s not in G:
            G.add_node(s, type="Person")
        if o not in G:
            G.add_node(o, type=OBJECT_TYPE.get(rel, "Entity"))
        G.add_edge(s, o, relation=rel, sentence=t.get("sentence", ""), doc_id=t.get("doc_id", ""))
    # tout sujet d'un triplet est une personne, même s'il est apparu d'abord comme objet mal typé
    for s in {t["subject"] for t in triples}:
        if s in G:
            G.nodes[s]["type"] = "Person"
    return G


def kg_stats(G):
    types = Counter(nx.get_node_attributes(G, "type").values())
    rels = Counter(r for _, _, r in G.edges(data="relation"))
    und = nx.Graph(G)
    comps = sorted(nx.connected_components(und), key=len, reverse=True)
    return {"nœuds": G.number_of_nodes(), "arcs": G.number_of_edges(), "types de nœuds": dict(types),
            "relations": dict(rels), "composantes connexes": len(comps),
            "taille plus grande composante": len(comps[0]) if comps else 0,
            "densité": round(nx.density(und), 5)}


def top_nodes(G, node_type=None, k=10):
    """Nœuds les plus connectés (degré entrant = nombre de personnes qui pointent vers eux)."""
    deg = [(n, G.in_degree(n)) for n, d in G.nodes(data=True) if node_type is None or d["type"] == node_type]
    return sorted(deg, key=lambda x: -x[1])[:k]


# ------------------------------------------------------------------ exports
def uri(name):
    return EX[re.sub(r"[^A-Za-z0-9_]", "_", strip_accents(name)).strip("_")]


def to_rdf(G, path):
    g = Graph()
    g.bind("ex", EX)
    for n, d in G.nodes(data=True):
        g.add((uri(n), RDF.type, EX[d["type"]]))
        g.add((uri(n), RDFS.label, Literal(n)))
    for s, o, d in G.edges(data=True):
        g.add((uri(s), EX[d["relation"]], uri(o)))
    g.serialize(destination=str(path), format="turtle")
    return len(g)


def export_all(G, triples, out_dir, prefix):
    pd.DataFrame(triples).to_csv(out_dir / f"{prefix}_triples.csv", index=False, encoding="utf-8")
    n_rdf = to_rdf(G, out_dir / f"{prefix}.ttl")
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((s, o, {"relation": d["relation"]}) for s, o, d in G.edges(data=True))
    nx.write_graphml(H, out_dir / f"{prefix}.graphml")
    return n_rdf


# ------------------------------------------------------------------ visualisation
def draw_subgraph(G, nodes, path, title, figsize=(13, 9), seed=7, edge_labels=True):
    H = G.subgraph(nodes)
    pos = nx.spring_layout(H, k=0.9, seed=seed)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [TYPE_COLOR.get(H.nodes[n]["type"], "#999") for n in H]
    sizes = [300 + 120 * H.degree(n) for n in H]
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.4, arrows=True, arrowsize=10, ax=ax,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
    if edge_labels:
        labels = {(s, o): d["relation"] for s, o, d in H.edges(data=True)}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=labels, font_size=6, font_color="#444", ax=ax)
    for t, c in TYPE_COLOR.items():
        ax.scatter([], [], c=c, label=t)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def ego_nodes(G, person, radius=1):
    return list(nx.ego_graph(G.to_undirected(as_view=True), person, radius=radius).nodes)


def overview_nodes(G, k_hubs=12):
    """Les organisations/prix les plus partagés + les personnes qui y sont reliées."""
    awards = [a for a in top_nodes(G, "Award", k_hubs // 2 + 1) if a[0] != COMMON][:k_hubs // 2]
    hubs = [n for n, _ in top_nodes(G, "Organization", k_hubs // 2) + awards]
    people = {p for h in hubs for p in G.predecessors(h)}
    return hubs + list(people)


def to_pyvis(G, path):
    from pyvis.network import Network
    net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources="remote")
    for n, d in G.nodes(data=True):
        net.add_node(n, label=n, color=TYPE_COLOR.get(d["type"], "#999"), title=d["type"],
                     size=8 + 2 * min(G.degree(n), 20))
    for s, o, d in G.edges(data=True):
        net.add_edge(s, o, title=d["relation"] + " : " + d.get("sentence", "")[:200], label=d["relation"],
                     font={"size": 7})
    net.barnes_hut(gravity=-6000, spring_length=120)
    net.write_html(str(path), notebook=False)


# ------------------------------------------------------------------ applications : QA et recommandation
def find_node(G, name, node_type=None):
    """Recherche approximative d'un nœud (insensible à la casse et aux accents)."""
    q = re.sub(r"^the\s+", "", strip_accents(name).lower().strip())   # les nœuds n'ont pas d'article
    cands = [n for n, d in G.nodes(data=True) if node_type is None or d["type"] == node_type]
    exact = [n for n in cands if strip_accents(n).lower() == q]
    if exact:
        return exact[0]
    part = [n for n in cands if q in strip_accents(n).lower()]
    return min(part, key=len) if part else None


QA_PATTERNS = [
    (r"where was (.+?) born", "born_in", "out"),
    (r"when was (.+?) born", "born_on", "out"),
    (r"where did (.+?) die", "died_in", "out"),
    (r"where did (.+?) (?:study|graduate)", "educated_at", "out"),
    (r"where did (.+?) work", "worked_at", "out"),
    (r"(?:which|what) (?:awards|prizes) did (.+?) (?:receive|win)", "award_received", "out"),
    (r"who shared the (?:nobel )?prize with (.+)", "shared_prize_with", "both"),
    (r"who (?:worked|works) at (.+)", "worked_at", "in"),
    (r"who (?:studied|graduated) (?:at|from) (.+)", "educated_at", "in"),
    (r"who (?:received|won) (?:the )?(.+)", "award_received", "in"),
    (r"who was born in (.+)", "born_in", "in"),
]


def answer(G, question):
    """Question-réponse par patrons : la question est traduite en requête sur le graphe."""
    q = question.lower().strip(" ?")
    for pat, rel, direction in QA_PATTERNS:
        m = re.match(pat, q)
        if not m:
            continue
        node = find_node(G, m.group(1))
        if node is None:
            return f"Entité inconnue dans le KG : {m.group(1)}"
        res = []
        if direction in ("out", "both"):
            res += [o for _, o, d in G.out_edges(node, data=True) if d["relation"] == rel]
        if direction in ("in", "both"):
            res += [s for s, _, d in G.in_edges(node, data=True) if d["relation"] == rel]
        return sorted(set(res)) or "Aucune réponse dans le KG"
    return "Question non reconnue"


def similar_people(G, person, k=5):
    """Recommandation : personnes partageant le plus d'organisations/prix (similarité de Jaccard)."""
    def feats(p):
        return {o for _, o, d in G.out_edges(p, data=True)
                if d["relation"] in ("worked_at", "educated_at", "award_received") and o != COMMON}
    base = feats(person)
    scores = []
    for n, d in G.nodes(data=True):
        if d["type"] == "Person" and n != person:
            f = feats(n)
            if f and base:
                j = len(base & f) / len(base | f)
                if j > 0:
                    scores.append((n, round(j, 2), sorted(base & f)))
    return sorted(scores, key=lambda x: -x[1])[:k]
