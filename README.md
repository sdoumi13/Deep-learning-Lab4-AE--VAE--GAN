# Deep-learning-Lab4-AE, VAE, GAN


# 📘 Rapport de TP : Modèles Génératifs Profonds (AE, VAE, GAN)

##  Introduction
Ce projet explore trois architectures fondamentales de l'apprentissage profond **non supervisé et génératif**, appliquées au dataset **MNIST** (chiffres manuscrits).

L'objectif est de :
- Réduire la dimensionnalité des données
- Structurer un espace latent
- Générer de nouvelles données synthétiques

Les modèles implémentés sont :

- **AutoEncoder (AE)** : Compression et reconstruction  
- **Variational AutoEncoder (VAE)** : Modélisation probabiliste de l'espace latent  
- **Generative Adversarial Network (GAN)** : Génération par compétition (jeu Min-Max)

---

## 🛠️ 1. AutoEncoder (AE)

L’AutoEncoder est un réseau de neurones qui apprend une **représentation compressée** des données d’entrée, puis tente de les reconstruire.

### 🔧 Architecture
Dans cette implémentation, la compression est volontairement **extrême**, passant de **784 dimensions** (image MNIST 28×28) à **2 dimensions**.

- **Encoder** :  
  `784 → 256 → 128 → 2`

- **Bottleneck (Goulot d’étranglement)** :  
  Vecteur latent de taille **2** contenant l’information essentielle.

- **Decoder** :  
  `2 → 128 → 256 → 784`

###  Analyse des Résultats
L’entraînement sur **20 époques** montre une convergence stable :

- **Début** : Loss ≈ `0.0603`
- **Fin** : Loss ≈ `0.0362`

 **Interprétation**  
La diminution progressive de la **MSELoss (Mean Squared Error)** indique que le modèle parvient à reconstruire les chiffres avec une erreur de plus en plus faible, malgré la perte d’information imposée par un espace latent très réduit.

---

##  2. Variational AutoEncoder (VAE)

Contrairement à l’AE classique, le **VAE** n’encode pas une image en un point fixe, mais en une **distribution de probabilité** définie par :

- une moyenne **μ**
- une variance **σ²**

###  Astuce de Reparamétrisation
Cette technique permet la rétropropagation du gradient à travers un échantillonnage aléatoire.

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Bruit aléatoire
    return mu + eps * std

 **Sans cette astuce, l’entraînement du VAE serait impossible.**

---

##  Fonction de Perte (Loss)

La **loss totale du VAE** est composée de deux termes opposés :

- **Perte de reconstruction**  
  → L’image reconstruite doit être proche de l’originale

- **Divergence de Kullback-Leibler (KL)**  
  → Force la distribution latente à suivre une loi normale  
  **𝒩(0,1)**

---

##  Analyse des Logs

- **Loss totale** diminue (`6438 → 4399`)  
  → Amélioration de la reconstruction

- **KL Divergence** augmente légèrement puis se stabilise  
  → L’espace latent devient structuré et exploitable

---

##  3. Generative Adversarial Network (GAN)

Le **GAN** repose sur un jeu à somme nulle entre deux réseaux :

- **Générateur (G)** : produit des images  
- **Discriminateur (D)** : distingue les vraies images des fausses

---

##  Architecture Corrigée (MNIST)

Suite à une erreur de dimension, l’architecture a été adaptée :

### 🔹 Générateur
- **Entrée** : vecteur de bruit `z ∈ ℝ¹⁰⁰`
- **Sortie** : image de taille `784` (MNIST)

### 🔹 Discriminateur
- **Entrée** : image `784`
- **Sortie** : probabilité *(réelle ou fausse)*

---

##  Dynamique d’Entraînement

- **Loss_D** : capacité du discriminateur à détecter les faux
- **Loss_G** : capacité du générateur à tromper le discriminateur

---

##  Objectif

Atteindre un **équilibre de Nash** :

- Le générateur produit des images réalistes
- Le discriminateur prédit avec une probabilité ≈ `0.5`

---

## 📊 Conclusion Technique

Ce TP met en évidence la progression de la complexité des modèles génératifs :

| Modèle | Avantages | Limites |
|------|----------|--------|
| **AE** | Simple, stable, efficace | Génération limitée |
| **VAE** | Espace latent continu et structuré | Images parfois floues |
| **GAN** | Images nettes et réalistes | Entraînement instable |

---

##  Prérequis pour l’Exécution

```bash
pip install torch torchvision matplotlib numpy
