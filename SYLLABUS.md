# AI-from-first-principles : Syllabus complet

> Du neurone au LLM, et au-delà. Chaque chapitre = un dossier avec : `notes.md` (théorie), un ou plusieurs notebooks `.ipynb` (implémentation pas à pas), `exercises.ipynb` (exercices), `interview.md` (questions d'entretien).
> Légende : 🟢 fondations · 🟡 intermédiaire · 🔴 avancé · 🧪 notebook from scratch · 📄 papier à lire

---

## PARTIE I : Fondations 🟢

### Chapitre 01 : Qu'est-ce que l'IA ?
- 1.1 Histoire : Turing, Dartmouth, hivers de l'IA, renaissance du deep learning, ère des LLM
- 1.2 IA symbolique vs apprentissage automatique vs deep learning
- 1.3 Types d'apprentissage : supervisé, non supervisé, auto-supervisé, par renforcement
- 1.4 Anatomie d'un problème ML : données, modèle, perte, optimisation, évaluation
- 1.5 Vocabulaire de base (feature, label, paramètre, hyperparamètre, inférence)

### Chapitre 02 : Algèbre linéaire
- 2.1 Vecteurs, espaces vectoriels, combinaisons linéaires
- 2.2 Matrices, produit matriciel, transposée, inverse
- 2.3 Normes, produit scalaire, distance, similarité cosinus
- 2.4 Projections, orthogonalité, bases
- 2.5 Valeurs propres et vecteurs propres
- 2.6 Décomposition SVD, rang, approximation de faible rang
- 2.7 🧪 Implémenter la PCA avec la SVD
- 2.8 Tenseurs, broadcasting, `einsum`

### Chapitre 03 : Calcul différentiel et optimisation
- 3.1 Dérivées, dérivées partielles, gradient
- 3.2 Règle de la chaîne, jacobienne, hessienne
- 3.3 Différentiation automatique (mode avant et inverse)
- 3.4 Convexité, minima locaux, points selle
- 3.5 Descente de gradient : batch, stochastique, mini-batch
- 3.6 Taux d'apprentissage, conditionnement, convergence
- 3.7 🧪 Descente de gradient sur fonctions 1D/2D, visualisation

### Chapitre 04 : Probabilités et statistiques
- 4.1 Axiomes, probabilités conditionnelles, théorème de Bayes
- 4.2 Variables aléatoires, espérance, variance, covariance
- 4.3 Lois : Bernoulli, binomiale, catégorielle, gaussienne, multivariée
- 4.4 Loi des grands nombres, théorème central limite
- 4.5 Estimation : maximum de vraisemblance, MAP, estimation bayésienne
- 4.6 Théorie de l'information : entropie, entropie croisée, divergence KL, information mutuelle
- 4.7 Tests d'hypothèses, intervalles de confiance, bootstrap
- 4.8 Méthodes de Monte-Carlo, échantillonnage
- 4.9 🧪 MLE d'une gaussienne, lien entre entropie croisée et vraisemblance

### Chapitre 05 : Outils de travail
- 5.1 Python scientifique : NumPy, pandas, matplotlib
- 5.2 Vectorisation et performance
- 5.3 PyTorch : tenseurs, autograd, `nn.Module`
- 5.4 Reproductibilité : seeds, environnements, Docker
- 5.5 Git, tests, bonnes pratiques de code ML

---

## PARTIE II : Machine Learning classique 🟡

### Chapitre 06 : Principes de l'apprentissage
- 6.1 Risque empirique, risque réel, généralisation
- 6.2 Sur-apprentissage et sous-apprentissage
- 6.3 Compromis biais-variance
- 6.4 Régularisation : L1, L2, elastic net, vue bayésienne
- 6.5 Jeux d'entraînement / validation / test, validation croisée
- 6.6 Fuite de données (*data leakage*) et pièges courants
- 6.7 Théorie de l'apprentissage : PAC, dimension VC (intuition), no free lunch

### Chapitre 07 : Modèles linéaires
- 7.1 Régression linéaire : moindres carrés, solution fermée
- 7.2 🧪 Régression linéaire par descente de gradient
- 7.3 Régression logistique, sigmoïde, softmax
- 7.4 🧪 Régression logistique from scratch
- 7.5 Modèles linéaires généralisés
- 7.6 Régression polynomiale, fonctions de base

### Chapitre 08 : Méthodes non paramétriques et probabilistes
- 8.1 k plus proches voisins, malédiction de la dimension
- 8.2 Naive Bayes
- 8.3 Modèles discriminatifs vs génératifs
- 8.4 Processus gaussiens (intuition)

### Chapitre 09 : SVM et méthodes à noyau
- 9.1 Marge maximale, formulation primale et duale
- 9.2 Astuce du noyau, noyaux usuels
- 9.3 Marge souple
- 9.4 🧪 SVM linéaire avec descente de gradient

### Chapitre 10 : Arbres et ensembles
- 10.1 Arbres de décision : Gini, entropie, élagage
- 10.2 🧪 Arbre de décision from scratch
- 10.3 Bagging, forêts aléatoires
- 10.4 Boosting : AdaBoost, gradient boosting
- 10.5 XGBoost, LightGBM, CatBoost
- 10.6 Pourquoi les arbres dominent encore le tabulaire

### Chapitre 11 : Apprentissage non supervisé
- 11.1 k-means, k-means++
- 11.2 Clustering hiérarchique, DBSCAN
- 11.3 Mélanges de gaussiennes et algorithme EM
- 11.4 Réduction de dimension : PCA, t-SNE, UMAP
- 11.5 Détection d'anomalies

### Chapitre 12 : Évaluation et ingénierie des données
- 12.1 Métriques de classification : accuracy, précision, rappel, F1, ROC-AUC, PR-AUC
- 12.2 Métriques de régression
- 12.3 Calibration des probabilités
- 12.4 Déséquilibre de classes
- 12.5 Feature engineering, valeurs manquantes, encodage
- 12.6 Interprétabilité : importance des variables, SHAP, LIME
- 12.7 Pipelines scikit-learn
- 12.8 Projet : compétition tabulaire de bout en bout

---

## PARTIE III : Deep Learning 🟡

### Chapitre 13 : Réseaux de neurones
- 13.1 Neurone artificiel, perceptron, limites (XOR)
- 13.2 Perceptron multicouche, théorème d'approximation universelle
- 13.3 Fonctions d'activation : sigmoïde, tanh, ReLU, GELU, SwiGLU
- 13.4 Fonctions de perte : MSE, entropie croisée, hinge
- 13.5 Propagation avant

### Chapitre 14 : Rétropropagation
- 14.1 Rétropropagation à la main sur un petit réseau
- 14.2 Graphes de calcul
- 14.3 🧪 Mini moteur d'autograd (type micrograd)
- 14.4 🧪 MLP from scratch sur MNIST (NumPy)
- 14.5 Vérification des gradients (*gradient checking*)

### Chapitre 15 : Entraîner correctement
- 15.1 Initialisation : Xavier, He
- 15.2 Optimiseurs : momentum, RMSprop, Adam, AdamW, Lion
- 15.3 Planificateurs : warmup, cosine, décroissance
- 15.4 Normalisation : BatchNorm, LayerNorm, RMSNorm
- 15.5 Régularisation : dropout, weight decay, augmentation de données, early stopping
- 15.6 Gradients qui explosent/disparaissent, *gradient clipping*
- 15.7 Diagnostiquer les courbes de perte, débogage
- 15.8 Précision mixte, entraînement GPU, checkpoints
- 15.9 Suivi d'expériences

### Chapitre 16 : Réseaux convolutifs (CNN)
- 16.1 Convolution, padding, stride, champ réceptif
- 16.2 Pooling, hiérarchie de features
- 16.3 🧪 Convolution from scratch
- 16.4 Architectures : LeNet, AlexNet, VGG, ResNet, EfficientNet, ConvNeXt
- 16.5 Connexions résiduelles, pourquoi ça marche
- 16.6 Transfer learning et fine-tuning
- 16.7 Projet : CIFAR-10 > 90 %

### Chapitre 17 : Modèles séquentiels (avant les Transformers)
- 17.1 RNN, rétropropagation dans le temps
- 17.2 LSTM, GRU
- 17.3 Seq2seq, premières formes d'attention
- 17.4 Limites : parallélisation, dépendances longues
- 17.5 Aperçu des modèles à états (SSM, Mamba) 🔴

### Chapitre 18 : Représentations et modèles génératifs classiques
- 18.1 Autoencodeurs, débruitage
- 18.2 Autoencodeurs variationnels (VAE) : ELBO, astuce de reparamétrisation
- 18.3 GAN : jeu min-max, instabilités, mode collapse
- 18.4 Modèles à flux normalisants (aperçu)
- 18.5 Modèles à énergie (aperçu)

### Chapitre 19 : Apprentissage de représentations
- 19.1 Auto-supervision : tâches prétextes
- 19.2 Apprentissage contrastif : SimCLR, InfoNCE
- 19.3 Méthodes sans négatifs : BYOL, DINO
- 19.4 Masquage : MAE
- 19.5 Embeddings et espaces latents

---

## PARTIE IV : Transformers et LLM 🔴

### Chapitre 20 : Langage et tokenisation
- 20.1 Représenter du texte : bag-of-words, TF-IDF
- 20.2 Modèles de langage : n-grammes, perplexité
- 20.3 Embeddings : word2vec, GloVe
- 20.4 Tokenisation : BPE, WordPiece, Unigram, SentencePiece, tokenisation par octets
- 20.5 🧪 Tokenizer BPE from scratch
- 20.6 Impact de la tokenisation (multilingue, nombres, coût)

### Chapitre 21 : Le mécanisme d'attention
- 21.1 Attention comme recherche pondérée (requêtes, clés, valeurs)
- 21.2 *Scaled dot-product attention*
- 21.3 Attention multi-têtes
- 21.4 Masque causal, auto-attention vs attention croisée
- 21.5 🧪 Attention from scratch
- 21.6 Complexité quadratique en mémoire et en calcul

### Chapitre 22 : L'architecture Transformer
- 22.1 Encodeur, décodeur, blocs résiduels, LayerNorm (pre-norm vs post-norm)
- 22.2 Réseaux feed-forward, rôle des MLP
- 22.3 Encodages positionnels : sinusoïdal, appris, RoPE, ALiBi
- 22.4 📄 *Attention Is All You Need*
- 22.5 🧪 Transformer complet from scratch
- 22.6 Familles : BERT, GPT, T5
- 22.7 Interprétabilité mécaniste : circuits, têtes d'induction (introduction) 🔴

### Chapitre 23 : Construire un LLM
- 23.1 Objectif d'entraînement : prédiction du prochain token
- 23.2 🧪 GPT miniature entraîné sur un petit corpus (nanoGPT)
- 23.3 Données : collecte, filtrage, déduplication, qualité, mélange
- 23.4 Lois d'échelle, Chinchilla, compute-optimal
- 23.5 Entraînement distribué : data/tensor/pipeline parallelism, ZeRO, FSDP
- 23.6 Stabilité d'entraînement, instabilités de perte
- 23.7 Mixture of Experts
- 23.8 Contexte long : RoPE scaling, attention par fenêtre, FlashAttention
- 23.9 Alternatives à l'attention dense : attention linéaire, SSM, hybrides

### Chapitre 24 : Post-entraînement et alignement
- 24.1 Fine-tuning supervisé (SFT), formats de conversation
- 24.2 Méthodes économes : LoRA, QLoRA, adaptateurs
- 24.3 RLHF : modèle de récompense, PPO
- 24.4 DPO et méthodes d'optimisation de préférences
- 24.5 Apprentissage par renforcement avec récompenses vérifiables, modèles de raisonnement
- 24.6 Distillation de connaissances
- 24.7 Alignement : objectifs, limites, *reward hacking*
- 24.8 🧪 Fine-tuning LoRA d'un petit modèle avec évaluation avant/après

### Chapitre 25 : Inférence et déploiement des LLM
- 25.1 Génération autoregressive, stratégies de décodage (greedy, beam, top-k, top-p, température)
- 25.2 KV cache
- 25.3 Quantification : INT8, INT4, GPTQ, AWQ
- 25.4 Décodage spéculatif, batching continu
- 25.5 Moteurs d'inférence (vLLM, llama.cpp) : principes
- 25.6 Coûts, latence, débit

### Chapitre 26 : Applications des LLM
- 26.1 Prompt engineering structuré, sorties structurées
- 26.2 Embeddings et recherche sémantique, bases vectorielles
- 26.3 RAG : découpage, récupération, reranking, génération
- 26.4 🧪 Système RAG complet avec évaluation
- 26.5 Appel d'outils et protocoles (MCP)
- 26.6 Agents : boucles de raisonnement/action, planification, mémoire
- 26.7 Systèmes multi-agents
- 26.8 Évaluation : benchmarks, LLM-as-judge, contamination, évaluations maison
- 26.9 Hallucinations, calibration, incertitude

---

## PARTIE V : Autres domaines 🔴

### Chapitre 27 : Vision par ordinateur moderne
- 27.1 Vision Transformer (ViT)
- 27.2 CLIP, apprentissage contrastif image-texte
- 27.3 Détection : YOLO, DETR ; segmentation : U-Net, SAM
- 27.4 Modèles vision-langage multimodaux
- 27.5 Vidéo et 3D (aperçu)

### Chapitre 28 : Modèles de diffusion
- 28.1 Processus de bruitage et débruitage, DDPM
- 28.2 Score matching, vue SDE
- 28.3 Guidage (classifier-free), diffusion latente
- 28.4 Flow matching, rectified flows
- 28.5 🧪 Diffusion minimale sur un petit jeu de données
- 28.6 Génération audio et vidéo (aperçu)

### Chapitre 29 : Apprentissage par renforcement
- 29.1 MDP, politique, fonctions de valeur, équations de Bellman
- 29.2 Programmation dynamique, Monte-Carlo, TD learning
- 29.3 Q-learning, DQN
- 29.4 Policy gradient, REINFORCE, actor-critic
- 29.5 PPO, TRPO, SAC
- 29.6 Exploration vs exploitation, bandits
- 29.7 🧪 DQN et PPO sur Gymnasium
- 29.8 Lien avec le post-entraînement des LLM

### Chapitre 30 : Autres sujets importants
- 30.1 Systèmes de recommandation
- 30.2 Séries temporelles et prévision
- 30.3 Graphes et GNN
- 30.4 Modèles audio et parole (Whisper, TTS)
- 30.5 Apprentissage fédéré, apprentissage continu, méta-apprentissage
- 30.6 Inférence causale (introduction)

---

## PARTIE VI : IA en production et en société

### Chapitre 31 : MLOps
- 31.1 Cycle de vie d'un projet ML
- 31.2 Servir un modèle : FastAPI, batching, latence
- 31.3 Docker, bases de Kubernetes
- 31.4 CI/CD pour le ML, tests de données et de modèles
- 31.5 Versionnage des données et modèles, registre
- 31.6 Monitoring : dérive, qualité, coûts
- 31.7 Cloud et GPU : choix et coûts
- 31.8 SQL et entrepôts de données
- 31.9 Projet : déploiement bout en bout

### Chapitre 32 : IA responsable, sécurité et cadre légal
- 32.1 Biais, équité, métriques de fairness
- 32.2 Explicabilité
- 32.3 Vie privée : anonymisation, confidentialité différentielle
- 32.4 Sécurité : *prompt injection*, jailbreaks, empoisonnement de données, attaques adverses
- 32.5 RGPD et AI Act (niveaux de risque, obligations)
- 32.6 Impact environnemental
- 32.7 Sécurité de l'IA et alignement : enjeux de long terme

---

## PARTIE VII : Frontière de la recherche 🔴 (chapitres vivants)

> Cette partie évolue : on y ajoutera ensemble les nouveautés au fil du temps. Un fichier par sujet, avec date, papiers de référence, intuition, implémentation minimale si possible.

- 33 : Modèles de raisonnement et calcul au moment du test
- 34 : Agents autonomes et utilisation d'ordinateurs
- 35 : Architectures post-Transformer (SSM, hybrides, attention efficace)
- 36 : Modèles multimodaux natifs
- 37 : Interprétabilité mécaniste et contrôle des modèles
- 38 : Données synthétiques et auto-amélioration
- 39 : Évaluation et sécurité des modèles de pointe
- 40 : Veille : template pour ajouter une nouveauté (voir ci-dessous)

### Template d'une fiche "nouveauté"
1. Contexte : quel problème ?
2. Idée clé en 3 phrases
3. Maths / algorithme
4. 📄 Papier(s) de référence
5. 🧪 Implémentation minimale
6. Limites et critiques
7. Questions d'entretien associées

---

## Annexes

- **A. Glossaire** : tous les termes, définis simplement
- **B. Bibliographie** : livres, cours, papiers classés par chapitre
- **C. Cheat-sheets** : formules clés, dérivations, architectures
- **D. Préparation aux entretiens** : banque de questions par chapitre, ML system design, algorithmique et SQL
- **E. Projets portfolio** : mini-autograd, mini-GPT, RAG évalué, projet déployé
- **F. Plan du site web** : voir `docs/` (Jupyter Book / Quarto)

---

## Ordre de lecture conseillé

1. **Socle** : 01 → 05
2. **ML classique** : 06 → 12
3. **Deep learning** : 13 → 19
4. **Transformers et LLM** : 20 → 26 (le cœur pour un AI Engineer)
5. **Production** : 31 → 32 (à entamer dès la partie IV)
6. **Spécialisations** : 27 → 30 selon intérêt
7. **Frontière** : 33+ en continu

> Les parties I à IV représentent le chemin critique. Les parties V à VII s'abordent ensuite ou en parallèle.

---

## Conventions du repo

- Un chapitre = un dossier `NN-nom-du-chapitre/`
- Chaque notebook : titre, objectifs, prérequis, théorie courte, code pas à pas, exercices, résumé, références
- Le code réutilisable va dans `src/` (modules `.py` testés), les notebooks l'importent
- Notebooks nettoyés avant commit (`nbstripout`), exécutés en CI pour garantir qu'ils tournent
- Tout concept est expliqué en trois niveaux : intuition, maths, code
