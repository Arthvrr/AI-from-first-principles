# AI from First Principles

> **Du gradient au LLM calibré et aux systèmes agentiques.**
>
> Une roadmap from scratch pour comprendre, implémenter et déployer les mécanismes internes de l'IA moderne — sans s'encombrer de l'historique devenu obsolète.

[![Progress](https://img.shields.io/badge/progress-0%2F15_chapters-lightgrey)](#progression)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

---

## 🎯 Objectif

Devenir **AI Engineer** avec une compréhension **interne** des modèles : non pas seulement les utiliser, mais savoir comment ils fonctionnent, pourquoi ils échouent, comment les évaluer, les calibrer et les déployer.

À l'issue du parcours : être capable de comprendre, implémenter, analyser, évaluer et déployer des modèles de Deep Learning, des LLMs et des systèmes agentiques.

---

## 📚 Légende

- 🟢 Fondations
- 🟡 Cœur du Deep Learning
- 🔴 LLM, Production & Agents
- 🧪 Implémentation from scratch

---

# PARTIE I — Fondations Mathématiques 🟢

## Chapitre 01 — Tenseurs et Calcul Différentiel

- **1.1** Algèbre pour le Deep Learning : scalaires, vecteurs, matrices, tenseurs et broadcasting.
- **1.2** Produit matriciel et opérations de base (NumPy pur) — *focus sur le broadcasting et le shaping.*
- **1.3** Calcul différentiel : dérivées partielles et concept de gradient.
- **1.4** Règle de la chaîne (*Chain Rule*) et graphes de calcul.

## Chapitre 02 — L'Optimisation de Base

- **2.1** Descente de gradient : le moteur de l'apprentissage.
- **2.2** Régression logistique et fonction d'activation Sigmoïde.
- **2.3** Entropie croisée binaire (*Binary Cross-Entropy*).
- **2.4** 🧪 Régression logistique *from scratch*.

## Chapitre 03 — Le Moteur d'Autograd (Projet Charnière)

- **3.1** Graphes de calcul dynamiques.
- **3.2** Forward pass et backward pass.
- **3.3** 🧪 Mini-moteur d'autograd (façon `micrograd`).
- **3.4** Vérification des gradients (*Gradient Checking*).

---

# PARTIE II — Fondations Deep Learning 🟡

## Chapitre 04 — Le Perceptron Multicouche (MLP)

- **4.1** Neurone artificiel et théorème d'approximation universelle.
- **4.2** Fonctions d'activation modernes : ReLU et GELU.
- **4.3** Softmax et Cross-Entropy multiclasse.
- **4.4** 🧪 MLP modulaire *from scratch* sur un dataset simple.

## Chapitre 05 — Entraîner des Réseaux Profonds

- **5.1** Initialisation des poids (Xavier, He) : explosion/disparition des gradients.
- **5.2** Optimiseurs modernes : Momentum → Adam / AdamW.
- **5.3** Régularisation : Dropout et Weight Decay.
- **5.4** Normalisation : LayerNorm et RMSNorm (préparation aux Transformers).

## Chapitre 06 — PyTorch Moderne et Pratiques d'Ingénierie *(court)*

- **6.1** `nn.Module`, `Dataset`/`DataLoader`, boucle d'entraînement idiomatique.
- **6.2** `autocast`, `torch.compile` (survol), device management.
- **6.3** 🧪 Reproduire le MLP du chapitre 04 en PyTorch pur — *comparer from scratch vs framework*.

---

# PARTIE III — Architecture Transformer & LLMs 🔴

## Chapitre 07 — Langage, Représentations et Tokenisation

- **7.1** Embeddings et espace latent — *lecture guidée, pas from scratch.*
- **7.2** Rôle de la tokenisation dans les LLMs modernes.
- **7.3** Algorithme Byte-Pair Encoding (BPE).
- **7.4** 🧪 Tokenizer BPE *from scratch*.

## Chapitre 08 — Le Mécanisme d'Attention

- **8.1** Attention comme recherche pondérée : Q, K, V.
- **8.2** *Scaled Dot-Product Attention* et masque causal.
- **8.3** Multi-Head Attention.
- **8.4** 🧪 Self-Attention et Masked Attention *from scratch*.

## Chapitre 09 — Construire un LLM (Type GPT)

- **9.1** Assemblage du bloc Transformer décodeur.
- **9.2** Encodages positionnels appris et RoPE.
- **9.3** 🧪 Architecture d'un mini-GPT de bout en bout.
- **9.4** Entraînement sur corpus miniature (prédiction du prochain token).

---

# PARTIE IV — Ingénierie, Déploiement et Production 🔴

## Chapitre 10 — Inférence et Optimisation

- **10.1** Génération autorégressive et décodage (température, top-p, top-k).
- **10.2** KV Cache : réduire la latence.
- **10.3** Quantification (INT4/INT8) et exécution locale (principes derrière Ollama / llama.cpp).
- **10.4** Fine-tuning économe : comprendre LoRA.

## Chapitre 11 — Évaluation des LLMs

- **11.1** Perplexité : ce qu'elle mesure (et ce qu'elle ne mesure pas).
- **11.2** Benchmarks standards (MMLU, HumanEval, GSM8K) et leurs limites.
- **11.3** Évaluation applicative : jeux de test métier, LLM-as-judge, éval humaine.
- **11.4** 🧪 Construire un petit pipeline d'évaluation reproductible pour un cas d'usage.

## Chapitre 12 — Systèmes RAG

- **12.1** Bases vectorielles : mécanique interne (façon ChromaDB).
- **12.2** Pipeline : découpage, embeddings, récupération sémantique.
- **12.3** 🧪 RAG en Python pur + évaluation maison (RAGAS en bonus).

## Chapitre 13 — Calibration, Fiabilité et Alignement

- **13.1** Calibration des probabilités : le modèle sait-il ce qu'il ne sait pas ?
- **13.2** Métriques : Expected Calibration Error (ECE), Brier Score.
- **13.3** Techniques correctives : Temperature Scaling, Platt Scaling.
- **13.4** Bases de l'alignement : SFT et intuition RLHF / DPO.

---

# PARTIE V — IA Agentique 🔴

## Chapitre 14 — Fondations des Agents LLM

- **14.1** Qu'est-ce qu'un agent ? Boucle *perception → raisonnement → action*.
- **14.2** Tool use et function calling : donner des outils au modèle.
- **14.3** Pattern ReAct (Reasoning + Acting) : raisonner avant d'agir.
- **14.4** Sorties structurées (JSON schema, grammar-constrained decoding).
- **14.5** 🧪 Mini-agent from scratch : un LLM qui appelle une calculatrice et une recherche web locale.

## Chapitre 15 — Systèmes Agentiques Avancés

- **15.1** Mémoire d'agent : court terme (contexte) vs long terme (vector store).
- **15.2** Patterns multi-agents (orchestrateur, débat, pipeline) — quand c'est utile, quand c'est du bruit.
- **15.3** MCP (Model Context Protocol) et interopérabilité des outils.
- **15.4** Guardrails, sandboxing et gestion des erreurs : un agent qui agit *safe*.
- **15.5** Évaluation d'agents : trajectoires, succès de tâche, coût par action.
- **15.6** 🧪 Construire un agent outillé de bout en bout (RAG + tool use + éval).

---

## 🏆 Objectif Final

Comprendre, implémenter, évaluer, calibrer et déployer des modèles de Deep Learning, des LLMs et des systèmes agentiques — avec la maîtrise des mécanismes internes comme différenciateur.

---

## 📊 Progression

Mise à jour au fil de l'avancement.

### Partie I — Fondations Mathématiques
- [ ] Ch. 01 — Tenseurs et Calcul Différentiel
- [ ] Ch. 02 — L'Optimisation de Base
- [ ] Ch. 03 — Le Moteur d'Autograd

### Partie II — Fondations Deep Learning
- [ ] Ch. 04 — Le Perceptron Multicouche (MLP)
- [ ] Ch. 05 — Entraîner des Réseaux Profonds
- [ ] Ch. 06 — PyTorch Moderne

### Partie III — Architecture Transformer & LLMs
- [ ] Ch. 07 — Langage, Représentations et Tokenisation
- [ ] Ch. 08 — Le Mécanisme d'Attention
- [ ] Ch. 09 — Construire un LLM (Type GPT)

### Partie IV — Ingénierie, Déploiement et Production
- [ ] Ch. 10 — Inférence et Optimisation
- [ ] Ch. 11 — Évaluation des LLMs
- [ ] Ch. 12 — Systèmes RAG
- [ ] Ch. 13 — Calibration, Fiabilité et Alignement

### Partie V — IA Agentique
- [ ] Ch. 14 — Fondations des Agents LLM
- [ ] Ch. 15 — Systèmes Agentiques Avancés

### Projets vitrines
- [ ] **MLP from scratch** (Partie II) — MNIST ou dataset tabulaire
- [ ] **Mini-GPT** (Partie III) — architecture + entraînement + génération
- [ ] **RAG calibré** (Partie IV) — RAG + éval + calibration
- [ ] **Agent outillé** (Partie V) — tool use + mémoire + éval

---

## 📄 Licence

MIT — voir [LICENSE](./LICENSE).