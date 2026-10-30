# Préparation de la présentation orale (~12 minutes + questions)

Support : `presentation/slides.pptx` (11 diapositives). Démo : notebook + `outputs/kg/kg_classic_interactive.html`.

## Déroulé minuté

| # | Diapositive | Durée | Message à faire passer |
|---|---|---|---|
| 1 | Titre | 0:30 | Sujet, corpus, idée générale : texte brut → graphe de connaissances |
| 2 | Problème et données | 1:00 | Pourquoi ce corpus : textes courts, faits vérifiables avec Wikidata, CoNLL-2003 pour évaluer la NER |
| 3 | Pipeline | 1:00 | Les 4 étapes A → D et les outils utilisés |
| 4 | A — Prétraitement et représentation | 1:15 | Deux niveaux de nettoyage ; TF-IDF vs MiniLM ; clustering faible car textes très homogènes |
| 5 | B — NER : résultats | 1:30 | CRF meilleur sur CoNLL ; règles précises mais rappel faible ; spaCy pénalisé par le schéma |
| 6 | C — Relations par dépendances | 1:15 | Remonter l'arbre jusqu'à un mot déclencheur ; coréférence heuristique |
| 7 | C — Évaluation et KG | 1:30 | Précision/rappel vs Wikidata ; taille du KG ; exports RDF/GraphML |
| 8 | Visualisation du KG | 1:00 | Ego-graphe et hubs (Cambridge, Manhattan Project…) ; montrer le HTML interactif |
| 9 | D — LLM vs classique | 1:30 | Tableau multi-critères : qualité, hallucinations, coût, explicabilité |
| 10 | D — Applications et discussion | 1:00 | QA, recommandation ; ambiguïtés (Columbia, Curie, received) |
| 11 | Conclusion | 0:30 | Les deux approches sont complémentaires ; limites et pistes |

## Chiffres à connaître par cœur

| Élément | Valeur |
|---|---|
| Corpus | 150 introductions Wikipedia, 21 542 mots, ~6 phrases par texte |
| Référence relations | 3 146 faits Wikidata |
| NER CoNLL (F1) | CRF 0,814 · règles 0,613 · spaCy small 0,595 (LLM 0,250 sur 100 phrases) |
| Accord CRF / spaCy sur Wikipedia | F1 0,46 (changement de domaine) |
| Relations classique (vs Wikidata) | P 0,753 · R 0,762 · F1 0,757 ; meilleures : born_on 0,98, award_received 0,94 |
| Relations LLM | P 0,544 · R 0,636 · F1 0,586 ; meilleur que nous sur educated_at (0,50 vs 0,30) |
| KG classique | 492 nœuds, 699 arcs, 1 682 triplets RDF |
| Ancrage dans le texte | classique 100 % · LLM 89 % |
| Temps (150 textes) | classique 2,5 s · LLM 72 min (~29 s/texte) |
| Recouvrement des deux KG | 411 triplets communs, Jaccard 0,37 |

## Démonstration (2 minutes max, si demandée)
1. Ouvrir `outputs/kg/kg_classic_interactive.html` : zoomer sur *University of Cambridge*, survoler un arc
   pour montrer la phrase source (provenance).
2. Dans le notebook, cellule D.4 : poser une question (`answer(G, "Who worked at CERN?")`).
3. Montrer une réponse brute du LLM (cellule D.1) et un triplet halluciné (cellule D.3).

## Questions probables et réponses

**Pourquoi évaluer la NER sur CoNLL et pas sur votre corpus ?**
Notre corpus n'a pas d'annotations d'entités, et annoter nous-mêmes aurait été long et subjectif.
CoNLL-2003 est la référence standard, avec un train pour le CRF et un test séparé. En contrepartie,
le CRF est évalué dans son propre domaine ; son accord avec spaCy sur Wikipedia montre qu'il se transfère mal.

**Pourquoi spaCy a-t-il un F1 plus faible que le CRF, alors que c'est un réseau de neurones ?**
Il n'a pas été entraîné sur CoNLL : son schéma (OntoNotes, 18 types) ne correspond pas exactement aux 4 types
de CoNLL, surtout MISC, et la projection perd de l'information. C'est aussi le plus petit modèle (`sm`).
Un modèle `en_core_web_trf` ou un fine-tuning sur CoNLL ferait mieux, au prix de calcul en plus.

**Comment fonctionne le CRF ?**
C'est un modèle probabiliste de séquence : il prédit toute la suite d'étiquettes en tenant compte des traits du mot
(majuscule, suffixe, POS, mots voisins) et des transitions entre étiquettes (I-PER ne peut pas suivre B-ORG).
Entraînement par L-BFGS avec régularisation L1/L2 (c1, c2).

**Pourquoi l'extraction de relations par règles plutôt que par apprentissage ?**
On n'a pas de données annotées en relations pour notre domaine. Les règles sur les dépendances sont
explicables (on garde la phrase et le mot déclencheur) et suffisent pour un schéma de 6 relations.
Le fine-tuning de BERT pour la RE était optionnel ; il faudrait d'abord annoter quelques centaines de phrases
(ou utiliser Wikidata en *distant supervision*).

**Votre précision est-elle fiable ?**
C'est une borne inférieure : un triplet absent de Wikidata est compté faux alors qu'il est souvent vrai
(ex. Bardeen chez Bell Labs). Le rappel n'est calculé que sur les faits Wikidata écrits dans le texte, pour ne
pas pénaliser l'extracteur sur des informations absentes.

**Comment est gérée la coréférence ?**
Par une heuristique : chaque article parle d'une personne, donc *he/she/who* et les mentions du nom de famille
sont rattachés à cette personne. Ça marche bien sur des biographies, pas sur des textes à plusieurs personnages.

**Pourquoi un si petit LLM ?**
Contrainte matérielle : CPU et 8 Go de RAM, sans API payante. Qwen2.5-1.5B en bfloat16 tient en ~3 Go.
Le code accepte aussi une API compatible OpenAI (Groq gratuit, Ollama) pour tester un modèle plus gros :
c'est la première piste d'amélioration.

**Qu'est-ce que l'« ancrage » (groundedness) ?**
La part des triplets dont l'objet apparaît dans le texte source. Un objet absent du texte est un indice
d'hallucination (le LLM « sait » ou invente une information). Par construction, l'approche classique est toujours ancrée.

**Neo4j ?**
On exporte en GraphML et en RDF/Turtle : le GraphML s'importe dans Neo4j (APOC `apoc.import.graphml`) ou Gephi,
le Turtle dans un triplestore (requêtes SPARQL). NetworkX a suffi pour la taille du graphe (quelques centaines de nœuds).

**Comment améliorer le système ?**
Liage d'entités vers Wikidata (fusionner « UC Berkeley » et « University of California, Berkeley »),
vraie résolution de coréférence, modèle NER plus gros ou fine-tuné, LLM plus gros avec vérification automatique
des triplets (ancrage + types), extraction des relations par *distant supervision*.

## Points d'attention
- Bien distinguer les **deux évaluations** : NER sur CoNLL (annoté), relations avec Wikidata (référence externe).
- Ne pas survendre le LLM ni l'approche classique : chacune a des forces différentes (tableau de la diapositive 9).
- Avoir le notebook déjà exécuté et ouvert (le LLM ne sera pas relancé en direct : cache).
