---
license: mit
language:
- fr
pretty_name: Aricate
---


🌊 Documentation : Architecture Aricate 🩵

# 🚀 Aricate : Un Framework SLM Léger pour le Q\&A, la completion de texte et bien d'autres 

![Aricate](http://www.image-heberg.fr/files/17641405502571958061.png)

Le Framework **Aricate** est une architecture de *Small Language Model* (SLM) conçue pour être **légère**, **rapide à entraîner** et **facile à déployer**. Il est idéal pour les tâches de Question/Réponse (Q/A) sur des jeux de données spécifiques et de petite à moyenne taille.

## 🎯 But du Framework

Aricate vise à démontrer qu'une architecture d'apprentissage profond personnalisée, combinant des concepts fondamentaux, peut être entièrement développée et publiée sur le Hub Hugging Face (modèle, *tokenizer* et configuration).

| Caractéristique | Valeur |
| :--- | :--- |
| **Type** | Sequence-to-Sequence (Génération de Séquence) |
| **Architectures** | GRU + Attention Additive (Bahdanau) |
| **Format de Déploiement** | Poids en Safetensors, Tokenizer JSON personnalisé |
| **Idéal pour** | Chatbots Q/A sur des domaines spécialisés |

-----

## 🏗️ Architecture et Fonctionnement d'Aricate

Aricate repose sur deux composants principaux qui travaillent de concert pour comprendre la question et générer la réponse.

### 1\. Le Cerveau Séquentiel (GRU)

Le modèle utilise une *Gated Recurrent Unit* (**GRU**), une forme de réseau de neurones récurrents, pour lire la séquence d'entrée (la question suivie du *token* `<sep>`).

  * Chaque mot est transformé en un vecteur numérique (*embedding*).
  * La GRU traite ces vecteurs un par un et génère un **état caché** à chaque étape. Cet état est la "mémoire" du modèle à l'instant $t$.
  * Un paramètre ajustable important est le **`num_layers`** (nombre de couches GRU) et la **`hidden_dim`** (la taille de la mémoire).

[Image of simple neural network]

### 2\. Le Mécanisme d'Attention Additive (Bahdanau)

L'**Attention** est le mécanisme clé qui permet à Aricate de générer des réponses pertinentes :

  * **Le Problème :** Lorsque la GRU finit de lire la séquence, l'état caché final a souvent "oublié" les premiers mots de la question.
  * **La Solution :** La couche d'Attention analyse l'état caché final et le compare à **tous** les états cachés intermédiaires générés par la GRU.
  * Elle calcule des **poids d'attention** pour déterminer quelles parties de la séquence d'entrée (les mots de la question) sont les plus importantes pour générer le prochain mot de la réponse.
  * Ces poids sont utilisés pour créer un **vecteur de contexte** qui met en évidence les informations cruciales.
  * **Génération :** La couche finale utilise le **vecteur de contexte combiné à l'état caché final** pour prédire le mot de la réponse le plus probable.

-----

## 📚 Comment les Modèles Aricate Apprennent

L'entraînement d'un modèle Aricate se fait par **prédiction du mot suivant** :

1.  **Préparation des Données :** Les paires Question/Réponse sont transformées en longues séquences d'entraînement : `[Question] <sep> [Réponse] <eos>`.
2.  **Formation des Paires :** Le modèle est entraîné à prédire le mot $W_{i+1}$ en se basant sur la séquence $W_1, W_2, ..., W_i$ (par exemple : prédire "la" après avoir vu "Quel est \<sep\>").
3.  **Fonction de Perte :** Nous utilisons la `nn.CrossEntropyLoss` pour mesurer l'écart entre la prédiction du modèle et le mot réel.
4.  **Amélioration :** L'optimiseur (comme Adam) ajuste les poids du modèle (embeddings, poids GRU, poids d'Attention) pour minimiser cette erreur sur des centaines d'époques.

-----

## ⚙️ Inférence et Techniques de Génération

Pour que le modèle génère des phrases complètes, il doit "boucler" la prédiction mot par mot jusqu'à ce qu'il prédise le *token* de fin de séquence (`<eos>`).

### Stratégie : Beam Search (Recherche en Faisceau)

Les modèles Aricate publiés utilisent par défaut la **Beam Search** pour garantir une haute qualité de génération :

  * Au lieu de choisir un seul mot le plus probable à chaque étape, l'algorithme garde en mémoire les **$K$ chemins de phrases les plus prometteurs** (où $K$ est la taille du faisceau, ex. 3).
  * Il évalue la probabilité cumulée de ces $K$ chemins.
  * Ceci permet d'éviter de choisir un mot très probable au début qui mènerait à une mauvaise phrase par la suite.

### Améliorer la Créativité

Si le modèle génère des réponses trop répétitives (surapprentissage), les développeurs sont encouragés à utiliser des techniques comme :

  * **Top-K Sampling :** Introduire de l'aléatoire dans la sélection du mot suivant pour encourager la diversité.
  * **Temperature (T \> 1.0) :** Rendre la distribution de probabilité des mots plus "plate" pour donner une chance aux mots moins probables.

-----

## 🛠️ Utilisation et Exemples de Code

### Installation

```bash
pip install torch huggingface_hub safetensors datasets
```

### Chargement et Inférence

Pour utiliser un modèle Aricate publié (comme `Clemylia/lam-2`), vous devez recharger la classe `AricateModel`, son `WordTokenizer` et le script `generate_sequence_beam`.

*(**Astuce :** Le code source complet avec les classes nécessaires pour l'inférence est disponible sur notre [Lien vers le dépôt GitHub de l'architecture Aricate].)*

```python
# Exemple de chargement (Similaire à hf_hub_download)
# ... (Code de la fonction load_lam2_model) ...

# Exemple d'inférence directe
lam2_model, lam2_tokenizer, max_len = load_lam2_model("Clemylia/lam-2")

question = "Quel est le rôle du framework Aricate ?"

generate_sequence_beam(
    model=lam2_model, 
    tokenizer=lam2_tokenizer, 
    question=question, 
    max_length=15, 
    max_len_input=max_len 
)
```

**Développé par :**  (Clemylia)
**Dataset d'Entraînement :** (`Clem27sey/Nacid`)

# 🦋 Exemples de Codes d'utilisations (Entraînement, inference, quantification)

🍪 **Entraînement Q/A from scratch (pour petits SLM):**

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # 💡 NOUVELLE IMPORTATION
import collections
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin, HfApi, notebook_login
import os
import time
import json
import heapq
from safetensors.torch import save_file as save_safetensors_file

# --- A. WordTokenizer (Inchangé) ---
class WordTokenizer:
    """Tokenizer simple pour l'architecture Aricate."""
    # ... (votre code WordTokenizer) ...
    def __init__(self, texts):
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)

        word_counts = collections.Counter(all_words)
        sorted_words = [word for word, count in word_counts.most_common()]

        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<sep>': 3,
        }

        self.word_to_id = self.special_tokens.copy()
        next_id = len(self.special_tokens)

        for word in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = next_id
                next_id += 1

        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        print(f"Tokenisation effectuée. Taille du vocabulaire : {self.vocab_size}")

    def encode(self, text, add_eos=False):
        words = text.lower().split()
        if add_eos:
            words.append('<eos>')

        ids = [self.word_to_id.get(word, self.word_to_id['<unk>']) for word in words]
        return ids

    def decode(self, ids):
        words = [self.id_to_word.get(id, '<unk>') for id in ids]
        return " ".join(word for word in words if word not in ['<pad>', '<unk>', '<eos>', '<sep>'])

# --- B. AricateAttentionLayer (Inchangé) ---
class AricateAttentionLayer(nn.Module):
    """Couche d'Attention Additive (Bahdanau)."""
    def __init__(self, hidden_dim):
        super(AricateAttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, rnn_outputs, last_hidden):
        last_hidden_expanded = last_hidden.unsqueeze(1)
        energy = torch.tanh(self.W(rnn_outputs) + self.U(last_hidden_expanded))
        attention_weights_raw = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention_weights_raw, dim=1)
        context_vector = torch.sum(rnn_outputs * attention_weights.unsqueeze(2), dim=1)
        return context_vector

# --- C. AricateModel V4 (Inchangé) ---
class AricateModel(nn.Module, PyTorchModelHubMixin):
    """Architecture Aricate V4. Hérite de PyTorchModelHubMixin pour la sauvegarde et la publication."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, config: dict = None):
        super(AricateModel, self).__init__()

        if config is not None:
             vocab_size = config.get("vocab_size", vocab_size)
             embedding_dim = config.get("embedding_dim", embedding_dim)
             hidden_dim = config.get("hidden_dim", hidden_dim)
             num_layers = config.get("num_layers", num_layers)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AricateAttentionLayer(hidden_dim)
        self.hidden_to_vocab = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_words):
        embeds = self.word_embeddings(input_words)
        rnn_out, hn = self.rnn(embeds)
        last_hidden = hn[-1]
        context_vector = self.attention(rnn_out, last_hidden)
        combined_features = torch.cat((context_vector, last_hidden), dim=1)
        logits = self.hidden_to_vocab(combined_features)
        return logits

# --- D. Fonctions de Préparation et de Génération (Inchangé) ---
def generate_sequence_beam(model, tokenizer, question, max_length, max_len_input, beam_size=3):
    """Génère la réponse en utilisant la Beam Search."""
    model.eval()

    sep_id = tokenizer.special_tokens['<sep>']
    eos_id = tokenizer.special_tokens['<eos>']

    question_ids = tokenizer.encode(question)
    initial_sequence = question_ids + [sep_id]

    beam = [(-0.0, initial_sequence)]
    finished_sequences = []

    print(f"\n--- Q/A Génération (Beam Search, K={beam_size}) ---")
    print(f"Question: '{question}'")

    with torch.no_grad():
        for _ in range(max_length):
            candidates = []
            for neg_log_prob_prev, sequence in beam:
                input_ids_to_pad = sequence[-max_len_input:] if len(sequence) > max_len_input else sequence
                padding_needed = max_len_input - len(input_ids_to_pad)
                input_ids_padded = [tokenizer.special_tokens['<pad>']] * padding_needed + input_ids_to_pad
                input_tensor = torch.tensor(input_ids_padded).unsqueeze(0)

                # Assurez-vous que le modèle est sur le bon appareil (CPU ou GPU)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)

                logits = model(input_tensor)
                log_probabilities = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = torch.topk(log_probabilities, beam_size)

                for log_prob_next, predicted_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_score = neg_log_prob_prev + (-log_prob_next)
                    new_sequence = sequence + [predicted_id]
                    candidates.append((new_score, new_sequence))

            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_size]

            new_beam = []
            for score, sequence in beam:
                last_word_id = sequence[-1]
                if last_word_id == eos_id:
                    finished_sequences.append((score, sequence))
                else:
                    new_beam.append((score, sequence))

            beam = new_beam
            if not beam and finished_sequences:
                break
            if not beam and not finished_sequences:
                break

    if finished_sequences:
        finished_sequences.sort(key=lambda x: x[0])
        best_sequence = finished_sequences[0]
    elif beam:
        beam.sort(key=lambda x: x[0])
        best_sequence = beam[0]
    else:
        print("Génération terminée : aucune séquence valide trouvée.")
        return "Je n'ai pas pu générer une réponse cohérente."

    final_ids = best_sequence[1]
    try:
        sep_index = final_ids.index(sep_id)
        response_ids = [id for id in final_ids[sep_index+1:] if id != eos_id]
    except ValueError:
        response_ids = final_ids

    final_response = tokenizer.decode(response_ids)

    print(f"Meilleur score (-logP): {best_sequence[0]:.4f}")
    print(f"Réponse générée: '{final_response}'")
    print("-" * 40)

    return final_response


# --- Nouvelle Classe PyTorch Dataset ---
class AricateDataset(Dataset):
    """Dataset personnalisé pour PyTorch."""
    def __init__(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- E. Fonction Principale (AVEC BATCHING) ---
def train_lam2_and_publish():
    # --- Configuration ---
    DATASET_NAME = "toughdata/quora-question-answer-dataset"
    REPO_ID = "Clemylia/quora-english"
    MODEL_NAME = "has-english"

    # ⬆️ PARAMÈTRES D'ENTRAÎNEMENT AJUSTÉS ⬆️
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    NUM_EPOCHS = 9
    BATCH_SIZE = 128 # 💡 NOUVEAU PARAMÈTRE ! Taille du mini-lot (ajustez si Colab plante encore)

    LEARNING_RATE = 0.005
    MAX_GENERATION_LENGTH = 15
    BEAM_SIZE = 3

    # --- Configuration de l'appareil (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil d'entraînement sélectionné: {device}")


    # --- Connexion Hugging Face ---
    # ... (inchangé) ...
    print("\n" + "="*50)
    print(">>> CONNEXION HUGGING FACE REQUISE POUR LA PUBLICATION <<<")
    notebook_login(new_session=False)
    print("Connexion établie (Vérifiez si le token a la permission 'Write').")
    print("="*50)

    print(f"--- Lancement de l'Entraînement du SLM {MODEL_NAME} (Aricate) ---")

    # 1. Préparation des données (Inchangé)
    DATASET_SPLIT = 'train'
    print(f"Chargement de la dataset '{DATASET_NAME}' (split '{DATASET_SPLIT}')...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    corpus_raw = [f"{ex['question']} <sep> {ex['answer']}" for ex in dataset]
    tokenizer = WordTokenizer(corpus_raw)

    train_data_X = []
    train_data_Y = []

    for item in dataset:
        q = item['question']
        r = item['reponse']
        full_seq_ids = tokenizer.encode(f"{q} <sep> {r}", add_eos=True)
        for i in range(1, len(full_seq_ids)):
            X = full_seq_ids[:i]
            Y = full_seq_ids[i]
            train_data_X.append(X)
            train_data_Y.append(Y)

    max_len = max(len(x) for x in train_data_X)
    padded_X = []
    for x in train_data_X:
        padding_needed = max_len - len(x)
        padded_X.append([tokenizer.special_tokens['<pad>']] * padding_needed + x)

    X_train_tensor = torch.tensor(padded_X)
    Y_train_tensor = torch.tensor(train_data_Y)
    VOCAB_SIZE = tokenizer.vocab_size

    print(f"Dataset chargée. Nombre de paires d'entraînement: {len(Y_train_tensor)}")
    print(f"Taille du vocabulaire total: {VOCAB_SIZE}")
    print(f"Longueur maximale d'entrée (max_len): {max_len}")

    # 💡 Création du DataLoader pour le Batching 💡
    aricate_dataset = AricateDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(
        dataset=aricate_dataset,
        batch_size=BATCH_SIZE, # Utilisation du nouveau paramètre BATCH_SIZE
        shuffle=True,          # Mélanger les données à chaque époque
        num_workers=2          # Permet de charger les données en parallèle (optionnel, mais efficace)
    )
    print(f"Nombre de batches par époque : {len(train_loader)}")

    # 2. Initialisation du Modèle
    model_config = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
    model = AricateModel(**model_config).to(device) # 💡 ENVOI DU MODÈLE SUR L'APPAREIL
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Entraînement avec Batching
    print(f"\nDébut de l'entraînement pour {NUM_EPOCHS} époques avec un BATCH_SIZE de {BATCH_SIZE}...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        # 💡 BOUCLE SUR LES BATCHES 💡
        for batch_X, batch_Y in train_loader:
            # 💡 ENVOI DES BATCHES SUR L'APPAREIL 💡
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_function(logits, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0) # Accumulation de la perte

        avg_loss = total_loss / len(aricate_dataset)

        if (epoch + 1) % 5 == 0: # Affichage un peu plus fréquent pour le batching
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss Moyenne: {avg_loss:.4f}')

    end_time = time.time()
    print(f"\nEntraînement terminé ! Durée: {(end_time - start_time):.2f}s 🎉")

    # 4. Phase de Test (Génération)
    # ... (inchangé) ...
    print("\n" + "="*50)
    print(">>> TEST FINAL DE LAM-2 (Aricate) <<<")
    print("="*50)
    # 💡 ENVOI DU MODÈLE EN MODE EVALUATION SUR L'APPAREIL 💡
    model.to(device) 

    dataset_test = load_dataset(DATASET_NAME, split='train[:3]')

    for i, item in enumerate(dataset_test):
        if i >= 3: break
        question = item['question']
        generate_sequence_beam(model, tokenizer, question, MAX_GENERATION_LENGTH, max_len, BEAM_SIZE)


    # 5. Sauvegarde et Publication sur Hugging Face
    # 💡 ENVOI DU MODÈLE SUR LE CPU POUR LA SAUVEGARDE (bonne pratique) 💡
    model.to("cpu")
    print("\n" + "="*50)
    print(">>> SAUVEGARDE ET PUBLICATION SUR HUGGING FACE <<<")
    print("="*50)

    save_directory = f"./{MODEL_NAME}_local_save"
    os.makedirs(save_directory, exist_ok=True)

    model.save_pretrained(save_directory)
    print(f"Modèle sauvegardé localement dans: {save_directory}")

    tokenizer_path = os.path.join(save_directory, "aricate_tokenizer.txt")
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.word_to_id, f, ensure_ascii=False)
    print(f"Tokenizer (vocabulaire) sauvegardé dans: {tokenizer_path}")

    try:
        model.push_to_hub(
            repo_id=REPO_ID,
            commit_message=f"Update: Lam-2 V2, capacity increased, trained with batching.",
            config=model_config
        )
        HfApi().upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="aricate_tokenizer.txt",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update Aricate custom tokenizer vocabulary."
        )

        print(f"\n✅ Publication réussie ! Le modèle Lam-2 est disponible sur : https://huggingface.co/{REPO_ID}")

    except Exception as e:
        print(f"\n❌ ERREUR DE PUBLICATION. Le modèle est sauvegardé localement mais l'envoi a échoué.")
        print(f"Détail de l'erreur: {e}")


# Lancement de la fonction principale
if __name__ == '__main__':
    train_lam2_and_publish()
```

🔥 **Entraînement Q/A (no crash, pour LLM et Grands SLM)** :

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # 💡 NOUVELLE IMPORTATION
import collections
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin, HfApi, notebook_login
import os
import time
import json
import heapq
from safetensors.torch import save_file as save_safetensors_file

# --- A. WordTokenizer (Inchangé) ---
class WordTokenizer:
    """Tokenizer simple pour l'architecture Aricate."""
    # ... (votre code WordTokenizer) ...
    def __init__(self, texts):
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)

        word_counts = collections.Counter(all_words)
        sorted_words = [word for word, count in word_counts.most_common()]

        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<sep>': 3,
        }

        self.word_to_id = self.special_tokens.copy()
        next_id = len(self.special_tokens)

        for word in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = next_id
                next_id += 1

        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        print(f"Tokenisation effectuée. Taille du vocabulaire : {self.vocab_size}")

    def encode(self, text, add_eos=False):
        words = text.lower().split()
        if add_eos:
            words.append('<eos>')

        ids = [self.word_to_id.get(word, self.word_to_id['<unk>']) for word in words]
        return ids

    def decode(self, ids):
        words = [self.id_to_word.get(id, '<unk>') for id in ids]
        return " ".join(word for word in words if word not in ['<pad>', '<unk>', '<eos>', '<sep>'])

# --- B. AricateAttentionLayer (Inchangé) ---
class AricateAttentionLayer(nn.Module):
    """Couche d'Attention Additive (Bahdanau)."""
    def __init__(self, hidden_dim):
        super(AricateAttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, rnn_outputs, last_hidden):
        last_hidden_expanded = last_hidden.unsqueeze(1)
        energy = torch.tanh(self.W(rnn_outputs) + self.U(last_hidden_expanded))
        attention_weights_raw = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention_weights_raw, dim=1)
        context_vector = torch.sum(rnn_outputs * attention_weights.unsqueeze(2), dim=1)
        return context_vector

# --- C. AricateModel V4 (Inchangé) ---
class AricateModel(nn.Module, PyTorchModelHubMixin):
    """Architecture Aricate V4. Hérite de PyTorchModelHubMixin pour la sauvegarde et la publication."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, config: dict = None):
        super(AricateModel, self).__init__()

        if config is not None:
             vocab_size = config.get("vocab_size", vocab_size)
             embedding_dim = config.get("embedding_dim", embedding_dim)
             hidden_dim = config.get("hidden_dim", hidden_dim)
             num_layers = config.get("num_layers", num_layers)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AricateAttentionLayer(hidden_dim)
        self.hidden_to_vocab = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_words):
        embeds = self.word_embeddings(input_words)
        rnn_out, hn = self.rnn(embeds)
        last_hidden = hn[-1]
        context_vector = self.attention(rnn_out, last_hidden)
        combined_features = torch.cat((context_vector, last_hidden), dim=1)
        logits = self.hidden_to_vocab(combined_features)
        return logits

# --- D. Fonctions de Préparation et de Génération (Inchangé) ---
def generate_sequence_beam(model, tokenizer, question, max_length, max_len_input, beam_size=3):
    """Génère la réponse en utilisant la Beam Search."""
    model.eval()

    sep_id = tokenizer.special_tokens['<sep>']
    eos_id = tokenizer.special_tokens['<eos>']

    question_ids = tokenizer.encode(question)
    initial_sequence = question_ids + [sep_id]

    beam = [(-0.0, initial_sequence)]
    finished_sequences = []

    print(f"\n--- Q/A Génération (Beam Search, K={beam_size}) ---")
    print(f"Question: '{question}'")

    with torch.no_grad():
        for _ in range(max_length):
            candidates = []
            for neg_log_prob_prev, sequence in beam:
                input_ids_to_pad = sequence[-max_len_input:] if len(sequence) > max_len_input else sequence
                padding_needed = max_len_input - len(input_ids_to_pad)
                input_ids_padded = [tokenizer.special_tokens['<pad>']] * padding_needed + input_ids_to_pad
                input_tensor = torch.tensor(input_ids_padded).unsqueeze(0)

                # Assurez-vous que le modèle est sur le bon appareil (CPU ou GPU)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)

                logits = model(input_tensor)
                log_probabilities = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = torch.topk(log_probabilities, beam_size)

                for log_prob_next, predicted_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_score = neg_log_prob_prev + (-log_prob_next)
                    new_sequence = sequence + [predicted_id]
                    candidates.append((new_score, new_sequence))

            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_size]

            new_beam = []
            for score, sequence in beam:
                last_word_id = sequence[-1]
                if last_word_id == eos_id:
                    finished_sequences.append((score, sequence))
                else:
                    new_beam.append((score, sequence))

            beam = new_beam
            if not beam and finished_sequences:
                break
            if not beam and not finished_sequences:
                break

    if finished_sequences:
        finished_sequences.sort(key=lambda x: x[0])
        best_sequence = finished_sequences[0]
    elif beam:
        beam.sort(key=lambda x: x[0])
        best_sequence = beam[0]
    else:
        print("Génération terminée : aucune séquence valide trouvée.")
        return "Je n'ai pas pu générer une réponse cohérente."

    final_ids = best_sequence[1]
    try:
        sep_index = final_ids.index(sep_id)
        response_ids = [id for id in final_ids[sep_index+1:] if id != eos_id]
    except ValueError:
        response_ids = final_ids

    final_response = tokenizer.decode(response_ids)

    print(f"Meilleur score (-logP): {best_sequence[0]:.4f}")
    print(f"Réponse générée: '{final_response}'")
    print("-" * 40)

    return final_response


# --- Nouvelle Classe PyTorch Dataset ---
class AricateDataset(Dataset):
    """Dataset personnalisé pour PyTorch."""
    def __init__(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- E. Fonction Principale (AVEC BATCHING) ---
def train_lam2_and_publish():
    # --- Configuration ---
    DATASET_NAME = "Clemylia/Pikachu-language"
    REPO_ID = "Clemylia/Pikachu-Aricate"
    MODEL_NAME = "BabyLaya"

    # ⬆️ PARAMÈTRES D'ENTRAÎNEMENT AJUSTÉS ⬆️
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_EPOCHS = 30
    BATCH_SIZE = 128 # 💡 NOUVEAU PARAMÈTRE ! Taille du mini-lot (ajustez si Colab plante encore)

    LEARNING_RATE = 0.005
    MAX_GENERATION_LENGTH = 15
    BEAM_SIZE = 3

    # --- Configuration de l'appareil (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil d'entraînement sélectionné: {device}")


    # --- Connexion Hugging Face ---
    # ... (inchangé) ...
    print("\n" + "="*50)
    print(">>> CONNEXION HUGGING FACE REQUISE POUR LA PUBLICATION <<<")
    notebook_login(new_session=False)
    print("Connexion établie (Vérifiez si le token a la permission 'Write').")
    print("="*50)

    print(f"--- Lancement de l'Entraînement du SLM {MODEL_NAME} (Aricate) ---")

    # 1. Préparation des données (Inchangé)
    DATASET_SPLIT = 'train'
    print(f"Chargement de la dataset '{DATASET_NAME}' (split '{DATASET_SPLIT}')...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    corpus_raw = [f"{ex['question']} <sep> {ex['reponse']}" for ex in dataset]
    tokenizer = WordTokenizer(corpus_raw)

    train_data_X = []
    train_data_Y = []

    for item in dataset:
        q = item['question']
        r = item['reponse']
        full_seq_ids = tokenizer.encode(f"{q} <sep> {r}", add_eos=True)
        for i in range(1, len(full_seq_ids)):
            X = full_seq_ids[:i]
            Y = full_seq_ids[i]
            train_data_X.append(X)
            train_data_Y.append(Y)

    max_len = max(len(x) for x in train_data_X)
    padded_X = []
    for x in train_data_X:
        padding_needed = max_len - len(x)
        padded_X.append([tokenizer.special_tokens['<pad>']] * padding_needed + x)

    X_train_tensor = torch.tensor(padded_X)
    Y_train_tensor = torch.tensor(train_data_Y)
    VOCAB_SIZE = tokenizer.vocab_size

    print(f"Dataset chargée. Nombre de paires d'entraînement: {len(Y_train_tensor)}")
    print(f"Taille du vocabulaire total: {VOCAB_SIZE}")
    print(f"Longueur maximale d'entrée (max_len): {max_len}")

    # 💡 Création du DataLoader pour le Batching 💡
    aricate_dataset = AricateDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(
        dataset=aricate_dataset,
        batch_size=BATCH_SIZE, # Utilisation du nouveau paramètre BATCH_SIZE
        shuffle=True,          # Mélanger les données à chaque époque
        num_workers=2          # Permet de charger les données en parallèle (optionnel, mais efficace)
    )
    print(f"Nombre de batches par époque : {len(train_loader)}")

    # 2. Initialisation du Modèle
    model_config = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
    model = AricateModel(**model_config).to(device) # 💡 ENVOI DU MODÈLE SUR L'APPAREIL
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Entraînement avec Batching
    print(f"\nDébut de l'entraînement pour {NUM_EPOCHS} époques avec un BATCH_SIZE de {BATCH_SIZE}...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        # 💡 BOUCLE SUR LES BATCHES 💡
        for batch_X, batch_Y in train_loader:
            # 💡 ENVOI DES BATCHES SUR L'APPAREIL 💡
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_function(logits, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0) # Accumulation de la perte

        avg_loss = total_loss / len(aricate_dataset)

        if (epoch + 1) % 5 == 0: # Affichage un peu plus fréquent pour le batching
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss Moyenne: {avg_loss:.4f}')

    end_time = time.time()
    print(f"\nEntraînement terminé ! Durée: {(end_time - start_time):.2f}s 🎉")

    # 4. Phase de Test (Génération)
    # ... (inchangé) ...
    print("\n" + "="*50)
    print(">>> TEST FINAL DE LAM-2 (Aricate) <<<")
    print("="*50)
    # 💡 ENVOI DU MODÈLE EN MODE EVALUATION SUR L'APPAREIL 💡
    model.to(device) 

    dataset_test = load_dataset(DATASET_NAME, split='train[:3]')

    for i, item in enumerate(dataset_test):
        if i >= 3: break
        question = item['question']
        generate_sequence_beam(model, tokenizer, question, MAX_GENERATION_LENGTH, max_len, BEAM_SIZE)


    # 5. Sauvegarde et Publication sur Hugging Face
    # 💡 ENVOI DU MODÈLE SUR LE CPU POUR LA SAUVEGARDE (bonne pratique) 💡
    model.to("cpu")
    print("\n" + "="*50)
    print(">>> SAUVEGARDE ET PUBLICATION SUR HUGGING FACE <<<")
    print("="*50)

    save_directory = f"./{MODEL_NAME}_local_save"
    os.makedirs(save_directory, exist_ok=True)

    model.save_pretrained(save_directory)
    print(f"Modèle sauvegardé localement dans: {save_directory}")

    tokenizer_path = os.path.join(save_directory, "aricate_tokenizer.txt")
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.word_to_id, f, ensure_ascii=False)
    print(f"Tokenizer (vocabulaire) sauvegardé dans: {tokenizer_path}")

    try:
        model.push_to_hub(
            repo_id=REPO_ID,
            commit_message=f"Update: Lam-2 V2, capacity increased, trained with batching.",
            config=model_config
        )
        HfApi().upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="aricate_tokenizer.txt",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update Aricate custom tokenizer vocabulary."
        )

        print(f"\n✅ Publication réussie ! Le modèle Lam-2 est disponible sur : https://huggingface.co/{REPO_ID}")

    except Exception as e:
        print(f"\n❌ ERREUR DE PUBLICATION. Le modèle est sauvegardé localement mais l'envoi a échoué.")
        print(f"Détail de l'erreur: {e}")


# Lancement de la fonction principale
if __name__ == '__main__':
    train_lam2_and_publish()
```

⚡ **Inference Q/A** :

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import collections
import heapq
# Importations des librairies nécessaires pour le chargement
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors_file

# --- A. AricateAttentionLayer (Inchangé) ---
class AricateAttentionLayer(nn.Module):
    # ... (code inchangé) ...
    """Couche d'Attention Additive (Bahdanau)."""
    def __init__(self, hidden_dim):
        super(AricateAttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, rnn_outputs, last_hidden):
        last_hidden_expanded = last_hidden.unsqueeze(1)
        energy = torch.tanh(self.W(rnn_outputs) + self.U(last_hidden_expanded))
        attention_weights_raw = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention_weights_raw, dim=1)
        context_vector = torch.sum(rnn_outputs * attention_weights.unsqueeze(2), dim=1)
        return context_vector

# --- B. AricateModel (Inchangé) ---
class AricateModel(nn.Module):
    # ... (code inchangé) ...
    """Architecture Aricate V4, adaptée pour le rechargement."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, config: dict = None):
        super(AricateModel, self).__init__()

        if config is not None:
             vocab_size = config.get("vocab_size", vocab_size)
             embedding_dim = config.get("embedding_dim", embedding_dim)
             hidden_dim = config.get("hidden_dim", hidden_dim)
             num_layers = config.get("num_layers", num_layers)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AricateAttentionLayer(hidden_dim)
        self.hidden_to_vocab = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_words):
        embeds = self.word_embeddings(input_words)
        rnn_out, hn = self.rnn(embeds)
        last_hidden = hn[-1]
        context_vector = self.attention(rnn_out, last_hidden)
        combined_features = torch.cat((context_vector, last_hidden), dim=1)
        logits = self.hidden_to_vocab(combined_features)
        return logits

# --- C. WordTokenizer (Inchangé) ---
class WordTokenizer:
    # ... (code inchangé) ...
    """Tokenizer Aricate adapté pour recharger à partir du vocabulaire publié."""
    def __init__(self, word_to_id: dict):
        self.word_to_id = word_to_id
        self.id_to_word = {id: word for word, id in word_to_id.items()}
        self.vocab_size = len(word_to_id)
        self.special_tokens = {
            '<pad>': word_to_id['<pad>'],
            '<unk>': word_to_id['<unk>'],
            '<eos>': word_to_id['<eos>'],
            '<sep>': word_to_id['<sep>'],
        }

    def encode(self, text, add_eos=False):
        words = text.lower().split()
        if add_eos:
            words.append('<eos>')
        ids = [self.word_to_id.get(word, self.word_to_id['<unk>']) for word in words]
        return ids

    def decode(self, ids):
        words = [self.id_to_word.get(id, '<unk>') for id in ids]
        return " ".join(word for word in words if word not in ['<pad>', '<unk>', '<eos>', '<sep>'])

# --- D. Fonction de Génération (MODIFIÉE pour Top-K Sampling et Temperature) ---
def generate_sequence(model, tokenizer, question, max_length, max_len_input, temperature=1.0, top_k=None):
    """
    Génère la réponse en utilisant Top-K Sampling et Temperature.
    
    Args:
        temperature (float): Ajuste la créativité (T > 1.0) ou la prudence (T < 1.0).
        top_k (int/None): Limite le choix aux K mots les plus probables pour l'échantillonnage.
    """
    model.eval()
    
    sep_id = tokenizer.special_tokens['<sep>']
    eos_id = tokenizer.special_tokens['<eos>']

    question_ids = tokenizer.encode(question)
    current_sequence = question_ids + [sep_id]
    
    print(f"\n--- Q/A Génération (Sampling | T={temperature:.2f} | K={top_k if top_k else 'désactivé'}) ---")
    print(f"Question: '{question}'")

    with torch.no_grad():
        for _ in range(max_length):
            
            # Préparer l'entrée
            input_ids_to_pad = current_sequence[-max_len_input:] if len(current_sequence) > max_len_input else current_sequence
            padding_needed = max_len_input - len(input_ids_to_pad)
            input_ids_padded = [tokenizer.special_tokens['<pad>']] * padding_needed + input_ids_to_pad
            input_tensor = torch.tensor(input_ids_padded).unsqueeze(0)

            # 1. Obtention des logits
            logits = model(input_tensor).squeeze(0)

            # 2. Application de la Temperature
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # 3. Application du Top-K
            if top_k is not None:
                # Filtrer les logits pour ne garder que le top_k
                values, indices = torch.topk(logits, k=top_k)
                
                # Créer un masque (tensor rempli de -inf)
                mask = torch.ones_like(logits) * float('-inf')
                
                # Mettre à jour le masque avec les valeurs filtrées
                logits = torch.scatter(mask, dim=0, index=indices, src=values)

            # 4. Convertir en probabilités et échantillonner
            probabilities = F.softmax(logits, dim=-1)
            
            # S'assurer que les probabilités somment à 1
            if top_k is not None:
                probabilities = probabilities.div(probabilities.sum())
            
            predicted_id = torch.multinomial(probabilities, num_samples=1).item()

            # 5. Mettre à jour la séquence
            current_sequence.append(predicted_id)

            if predicted_id == eos_id:
                break

    # 6. Décodage
    try:
        sep_index = current_sequence.index(sep_id)
        response_ids = [id for id in current_sequence[sep_index+1:] if id != eos_id]
    except ValueError:
        response_ids = current_sequence

    final_response = tokenizer.decode(response_ids)
    
    # Dans le sampling, on n'a pas de score de log-probabilité unique comme dans Beam Search.
    print(f"Réponse générée: '{final_response}'")
    print("-" * 40)
    
    return final_response

# --- E. Fonction de Chargement du Modèle Lam-2 (Inchangée) ---
def load_lam2_model(repo_id: str):
    # ... (code inchangé) ...
    """
    Télécharge et charge le modèle Lam-2 et son tokenizer depuis Hugging Face.
    """
    print(f"--- Chargement de Lam-2 depuis {repo_id} ---")

    # 1. Télécharger le tokenizer
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="aricate_tokenizer.txt")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        word_to_id = json.load(f)
    tokenizer = WordTokenizer(word_to_id)
    print(f"Tokenizer chargé. Taille du vocabulaire: {tokenizer.vocab_size}")

    # 2. Télécharger la configuration
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    print("Configuration du modèle chargée.")

    # 3. Initialiser le modèle
    model = AricateModel(
        vocab_size=model_config['vocab_size'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        config=model_config
    )

    # 4. Télécharger et charger les poids Safetensors
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_safetensors_file(weights_path)

    model.load_state_dict(state_dict)
    print("Poids du modèle Safetensors chargés avec succès.")

    MAX_LEN_INPUT_DEFAULT = 30

    print("-" * 40)
    return model, tokenizer, MAX_LEN_INPUT_DEFAULT

# --- F. Bloc principal d'exécution (MISE À JOUR) ---
if __name__ == '__main__':

    LAM2_REPO_ID = "Clemylia/Pikachu-Aricate"
    MAX_GENERATION_LENGTH = 15
    
    # 🚨 NOUVEAUX PARAMÈTRES POUR LE TEST 🚨
    TEST_TEMPERATURE = 0.8 # > 1.0 pour plus de créativité/aléatoire
    TEST_TOP_K = 10         # Limite le choix aux 10 mots les plus probables

    test_questions = [
        "Qui es-tu ?",
        "Comment l'amitié peut-elle être une source d'**inspiration scientifique** ou de découverte ?",
        "Quel est ton nom ?",
        "Comment l'identité de Charlotte pourrait-elle être utilisée dans la **gestion de crise** ou le soutien post-traumatique ?",
        "Qui t'a créé ?",
    ]

    try:
        # 1. Chargement du modèle
        lam2_model, lam2_tokenizer, max_len_input = load_lam2_model(LAM2_REPO_ID)

        print(f"\n>>> TEST D'INFÉRENCE LAM-2 EN MODE CRÉATIF (T={TEST_TEMPERATURE}, K={TEST_TOP_K}) <<<")

        # 2. Infèrence (Appel à la nouvelle fonction)
        for question in test_questions:
            generate_sequence( # Remplacement de generate_sequence_beam
                model=lam2_model,
                tokenizer=lam2_tokenizer,
                question=question,
                max_length=MAX_GENERATION_LENGTH,
                max_len_input=max_len_input,
                temperature=TEST_TEMPERATURE,
                top_k=TEST_TOP_K
            )

    except Exception as e:
        print(f"\n❌ Une erreur est survenue lors du chargement ou de l'inférence.")
        print(f"Détail de l'erreur: {e}")
        print("Vérifiez l'installation des dépendances et le REPO_ID.")
```

🌸 **Quantification** :

La quantification des SLM crée avec Aricate, se font au format : .arica 

Code de quantification :

```
import torch
import json
import struct
import numpy as np
from safetensors import safe_open
from tqdm import tqdm
import os
from collections import OrderedDict
from huggingface_hub import hf_hub_download

# =================================================================
# I. CONFIGURATION HUGGING FACE ET ARICA
# =================================================================

ARICA_MAGIC = b'ARIC' # Identifiant du format Arica
ARICA_VERSION = 1
OUTPUT_FILE = "pikachu_quantized.arica"

# --- CONFIGURATION HUGGING FACE (Chargement de LAM-2 / Aricate V4) ---
HF_MODEL_ID = "Clemylia/Pikachu-Aricate"
FILES_TO_DOWNLOAD = [
    "model.safetensors",
    "config.json",
    "aricate_tokenizer.txt" # Fichier de vocabulaire spécifique Aricate
]

# --- Fonction de Quantification Q4_0 ---

def quantize_q4_0(tensor_data: np.ndarray):
    """
    Quantification Q4_0 (4-bit sans zéro-point).
    Retourne (scale, poids_quantifies_4bit) ou (None, None) si le tenseur est trop petit.
    """
    # Éviter la quantification sur de très petits tenseurs (ex: biais)
    if tensor_data.dtype != np.float32 or tensor_data.size < 16:
        return None, None

    data_flat = tensor_data.flatten()
    abs_max = np.max(np.abs(data_flat))
    
    # Échelle pour Q4_0: divise par 8 pour mapper sur [-8, 7]
    SCALE_DIV = 8.0
    scale = abs_max / SCALE_DIV

    if scale == 0:
        return np.float32(0), np.zeros(data_flat.size // 2, dtype=np.uint8)

    # Quantification: Q = round(D / scale) et clamp [-8, 7]
    quant_data = np.round(data_flat / scale).astype(np.int8)
    quant_data = np.clip(quant_data, -8, 7)

    num_elements = quant_data.size
    
    # Padding si le nombre d'éléments est impair
    if num_elements % 2 != 0:
        quant_data = np.concatenate((quant_data, np.array([0], dtype=np.int8)))
        num_elements += 1
        
    quant_bytes = np.zeros(num_elements // 2, dtype=np.uint8)
    
    for i in range(0, num_elements, 2):
        # Conversion de [-8, 7] à [0, 15] pour l'empaquetage
        low_bits = quant_data[i] + 8      
        high_bits = quant_data[i+1] + 8   
        
        # Empaquetage des 4 bits bas et 4 bits hauts dans un seul octet (uint8)
        quant_bytes[i // 2] = (high_bits << 4) | low_bits

    return np.float32(scale), quant_bytes

# =================================================================
# II. FONCTIONS UTILITAIRES D'ÉCRITURE BINAIRE
# =================================================================

def write_data(f, data, format_string):
    """Écrit des données binaires (little-endian) avec un format struct."""
    f.write(struct.pack(format_string, data))

def write_string(f, s):
    """Écrit la longueur de la chaîne (Int 32-bit) puis la chaîne encodée en UTF-8."""
    s_bytes = s.encode('utf-8')
    write_data(f, len(s_bytes), '<I') 
    f.write(s_bytes) 

# =================================================================
# III. FONCTIONS DE TÉLÉCHARGEMENT ET CONVERSION
# =================================================================

def download_model_files(repo_id, files_list):
    """Télécharge les fichiers nécessaires du Hub HF et retourne leurs chemins locaux."""
    
    print(f"--- 📥 TÉLÉCHARGEMENT DES FICHIERS {repo_id} DE HUGGING FACE ---")
    local_paths = {}
    
    for filename in files_list:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            local_paths[filename] = path
            print(f"  ✅ {filename} téléchargé vers {path}")
        except Exception as e:
            print(f"  ❌ ERREUR: Impossible de télécharger {filename}. Vérifiez le nom dans le dépôt.")
            print(f"  Détail: {e}")
            return None
            
    return local_paths

def write_arica_file(paths):
    """Lit les fichiers LAM-2 (Aricate), quantifie les poids et écrit le fichier ARICA."""
    
    safetensors_path = paths["model.safetensors"]
    config_path = paths["config.json"]
    tokenizer_path = paths["aricate_tokenizer.txt"] 
    
    print(f"\n--- ⚙️ CONVERSION LAM-2 (ARICATE) VERS LE FORMAT ARICA ({OUTPUT_FILE}) ---")
    
    # 1. Lecture des fichiers de configuration et de VOCABULAIRE
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # ⚠️ Lecture du .TXT en tant que JSON
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            print(f"  Lecture du tokenizer Aricate (.txt) en tant que JSON...")
            tokenizer_vocab = json.load(f) 
            
    except FileNotFoundError as e:
        print(f"ERREUR Critique: Fichier local manquant après le téléchargement: {e.filename}")
        return
    except json.JSONDecodeError:
        print("\n🚨 ERREUR CRITIQUE DE FORMAT : Le contenu du fichier 'aricate_tokenizer.txt'")
        print("doit être un dictionnaire JSON valide, même avec l'extension .txt.")
        print("Conversion annulée.")
        return

    # --- 2. Ouverture du fichier binaire ARICA ---
    with open(OUTPUT_FILE, 'wb') as f:
        
        # --- A. EN-TÊTE ---
        f.write(ARICA_MAGIC)
        write_data(f, ARICA_VERSION, '<I')
        
        # --- B. HYPERPARAMÈTRES (JSON) ---
        print("Écriture des hyperparamètres (config.json)...")
        config_json_bytes = json.dumps(config, indent=2).encode('utf-8')
        write_data(f, len(config_json_bytes), '<I')
        f.write(config_json_bytes)
        
        # --- C. VOCABULAIRE ARICATE ---
        print(f"Écriture du vocabulaire Aricate ({len(tokenizer_vocab)} entrées)...")
        write_data(f, len(tokenizer_vocab), '<I') 
        
        # Écriture de chaque mot et de son ID
        sorted_vocab = sorted(tokenizer_vocab.items(), key=lambda item: item[1])
        
        for word, idx in sorted_vocab:
            write_string(f, word) # Longueur + Mot
            write_data(f, idx, '<I') # ID
            
        # --- D. TENSEURS QUANTIFIÉS (Poids) ---
        print("Début de la quantification des poids (Q4_0)...")
        
        try:
            with safe_open(safetensors_path, framework="pt") as st:
                tensors = st.keys()
                write_data(f, len(tensors), '<I') # Nombre total de tenseurs

                for tensor_name in tqdm(tensors, desc="Conversion des tenseurs"):
                    tensor = st.get_tensor(tensor_name).cpu().numpy().astype(np.float32)
                    
                    # 1. Quantification
                    scale, quantized_bytes = quantize_q4_0(tensor)
                    
                    quantization_type = 1 if scale is not None else 0 # 1=Q4_0, 0=Float32

                    # 2. Écriture des Métadonnées
                    write_string(f, tensor_name)
                    
                    write_data(f, len(tensor.shape), '<I') 
                    for dim in tensor.shape:
                         write_data(f, dim, '<I')
                    
                    write_data(f, quantization_type, '<I') 

                    # 3. Écriture des Poids
                    if quantization_type == 1:
                        # Q4_0
                        f.write(scale.tobytes()) 
                        write_data(f, len(quantized_bytes), '<I') 
                        f.write(quantized_bytes.tobytes()) 
                    else:
                        # Float32 (Non quantifié)
                        data_bytes = tensor.tobytes()
                        write_data(f, len(data_bytes), '<I') 
                        f.write(data_bytes) 
                        
        except Exception as e:
            print(f"ERREUR lors de la lecture des safetensors: {e}")
            return
        
    print(f"\n✅ CONVERSION TERMINÉE ! Fichier ARICA '{OUTPUT_FILE}' créé avec succès.")

# =================================================================
# IV. BLOC PRINCIPAL D'EXÉCUTION
# =================================================================

if __name__ == '__main__':
    
    # 1. Téléchargement des fichiers du Hub HF
    local_files = download_model_files(HF_MODEL_ID, FILES_TO_DOWNLOAD)
    
    if local_files:
        # 2. Conversion en Arica
        write_arica_file(local_files)
    else:
        print("\n❌ CONVERSION AVORTÉE.")
```

🎉 **Inference avec votre SLM Aricate quantifier (fichier arica)** :

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import json
from tqdm import tqdm
import os

# =================================================================
# I. CONFIGURATION ET LOGIQUE ARICA
# =================================================================

ARICA_MAGIC = b'ARIC'
ARICA_VERSION = 1
ARICA_FILE = "pikachu_quantized.arica"

# Définition des types de quantification
Q_TYPE_FLOAT32 = 0
Q_TYPE_Q4_0 = 1

DEVICE = torch.device("cpu") # Utilisation du CPU pour simuler un environnement simple

def dequantize_q4_0(scale, quant_bytes, original_shape):
    """
    Déquantification simple Q4_0 (inverse du processus de conversion).
    """

    # 1. Décompression des octets
    num_bytes = len(quant_bytes)
    data_int8 = np.zeros(num_bytes * 2, dtype=np.int8)

    for i in range(num_bytes):
        byte = quant_bytes[i]

        # Poids bas (bits 0-3)
        low_bits = byte & 0xF
        data_int8[i * 2] = low_bits - 8 # [-8, 7]

        # Poids haut (bits 4-7)
        high_bits = byte >> 4
        data_int8[i * 2 + 1] = high_bits - 8 # [-8, 7]

    # Suppression du padding potentiel
    total_elements = np.prod(original_shape)
    data_int8 = data_int8[:total_elements]

    # 2. Déquantification: Valeur = Poids_quantifié * Scale
    data_float32 = data_int8.astype(np.float32) * scale

    # 3. Remodelage (Reshape)
    return torch.from_numpy(data_float32.reshape(original_shape)).to(DEVICE)


class AricaModelRuntime(nn.Module):
    """
    Classe pour charger et utiliser les poids Arica déquantifiés.
    Ceci est une simulation de l'architecture Aricate (GRU + Attention).
    """
    def __init__(self, config):
        super(AricaModelRuntime, self).__init__()
        # Initialisation minimaliste des paramètres de l'architecture Aricate
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']

        # Les modules sont créés mais n'ont pas encore de poids.
        # Les poids seront chargés et injectés manuellement plus tard.

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # Note: GRU/RNN complexes ont des structures de poids multiples (ih, hh)
        # Pour la simplicité, nous initialisons GRU et nous injecterons les Tenseurs déquantifiés.
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # La couche Attention n'est pas nécessaire si nous simplifions la lecture de 'rnn_out'
        # Pour la fidélité, nous créons des placeholders pour les poids de l'Attention et du Head
        self.attn_w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_u = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_v = nn.Linear(self.hidden_dim, 1, bias=False)

        self.hidden_to_vocab = nn.Linear(self.hidden_dim * 2, self.vocab_size)

        # Conteneur pour le vocabulaire (chargé depuis le fichier Arica)
        self.id_to_word = {}
        self.word_to_id = {}

    def attention_forward(self, rnn_outputs, last_hidden):
        """Simulation de la couche d'Attention additive Aricate."""
        last_hidden_expanded = last_hidden.unsqueeze(1)
        # Utilise les poids injectés (self.attn_w.weight, self.attn_w.bias, etc.)
        energy = torch.tanh(self.attn_w(rnn_outputs) + self.attn_u(last_hidden_expanded))
        attention_weights_raw = self.attn_v(energy).squeeze(2)
        attention_weights = F.softmax(attention_weights_raw, dim=1)
        context_vector = torch.sum(rnn_outputs * attention_weights.unsqueeze(2), dim=1)
        return context_vector

    def forward(self, input_ids):
        """Passage avant simplifié pour l'inférence."""
        if input_ids.size(1) == 0:
             return torch.zeros(input_ids.size(0), self.vocab_size, device=input_ids.device)

        embeds = self.word_embeddings(input_ids)
        rnn_out, hn = self.rnn(embeds)
        last_hidden = hn[-1]

        context_vector = self.attention_forward(rnn_out, last_hidden)
        combined_features = torch.cat((context_vector, last_hidden), dim=1)

        logits = self.hidden_to_vocab(combined_features)
        return logits

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7):
        """Génération séquentielle simple (greedy sampling)."""
        self.eval()

        if not self.word_to_id:
             return "Erreur: Vocabulaire non chargé."

        pad_id = self.word_to_id.get('<pad>', 0)
        eos_id = self.word_to_id.get('<eos>', 2)
        sep_id = self.word_to_id.get('<sep>', 3)
        sys_id = self.word_to_id.get('<sys>', 4)
        unk_id = self.word_to_id.get('<unk>', 1)

        # Tokenisation simple du prompt
        prompt_ids = [sys_id] + [self.word_to_id.get(w.lower(), unk_id) for w in prompt.split()] + [sep_id]

        # Préparation de l'entrée
        current_input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        output_ids = prompt_ids[:]

        with torch.no_grad():
            for _ in range(max_length):

                # Le modèle Arica/Aricate est entraîné pour prédire le jeton suivant
                logits = self(current_input_ids)

                if temperature == 0.0:
                    next_token_id = torch.argmax(logits, dim=-1).item()
                else:
                    probabilities = F.softmax(logits / temperature, dim=-1).squeeze(0)
                    next_token_id = torch.multinomial(probabilities, num_samples=1).item()

                output_ids.append(next_token_id)

                if next_token_id == eos_id:
                    break

                # Le nouvel input pour l'étape suivante inclut le jeton généré
                current_input_ids = torch.tensor(output_ids).unsqueeze(0).to(DEVICE)

        # Décodage (on retire les jetons système/séparateurs pour la réponse)
        response_words = []
        in_response_mode = False

        for id in output_ids:
            if id == sep_id:
                in_response_mode = True
                continue
            if id == eos_id:
                break

            if in_response_mode:
                word = self.id_to_word.get(id, '<unk>')
                if word not in ['<pad>', '<unk>', '<sys>', '<sep>']:
                    response_words.append(word)

        return " ".join(response_words)

# =================================================================
# II. LOGIQUE DE CHARGEMENT BINAIRE
# =================================================================

def read_data(f, format_string):
    """Lit les données binaires et les déballe."""
    size = struct.calcsize(format_string)
    data = f.read(size)
    if not data or len(data) != size: raise EOFError("Fin de fichier inattendue lors de la lecture des données structurées.")
    return struct.unpack(format_string, data)[0]

def read_string(f):
    """Lit la longueur d'une chaîne (Int 32-bit) puis lit la chaîne encodée en UTF-8."""
    length = read_data(f, '<I')
    s_bytes = f.read(length)
    if not s_bytes or len(s_bytes) != length: raise EOFError("Fin de fichier inattendue lors de la lecture de la chaîne.")
    return s_bytes.decode('utf-8')

def load_arica_model(arica_file_path):
    """Charge l'intégralité du modèle Arica depuis le fichier binaire."""

    print(f"Chargement du modèle Arica depuis : {arica_file_path}")

    with open(arica_file_path, 'rb') as f:
        # --- A. EN-TÊTE ---
        magic = f.read(4)
        if magic != ARICA_MAGIC:
            raise ValueError(f"Fichier invalide. Magic number attendu '{ARICA_MAGIC.decode()}', trouvé '{magic.decode()}'")
        version = read_data(f, '<I')
        if version != ARICA_VERSION:
            print(f"Attention: Version Arica attendue {ARICA_VERSION}, trouvée {version}.")

        # --- B. HYPERPARAMÈTRES (CONFIG) ---
        config_len = read_data(f, '<I')
        config_json = f.read(config_len).decode('utf-8')
        config = json.loads(config_json)

        model = AricaModelRuntime(config).to(DEVICE)

        # --- C. VOCABULAIRE ARICATE ---
        vocab_size = read_data(f, '<I')
        word_to_id = {}
        id_to_word = {}

        for _ in range(vocab_size):
            word = read_string(f)
            idx = read_data(f, '<I')
            word_to_id[word] = idx
            id_to_word[idx] = word

        model.word_to_id = word_to_id
        model.id_to_word = id_to_word
        print(f"  ✅ Vocabulaire chargé ({vocab_size} mots).")

        # --- D. TENSEURS QUANTIFIÉS (Poids) ---
        num_tensors = read_data(f, '<I')
        loaded_state_dict = {}

        print(f"  Début du chargement et de la déquantification des {num_tensors} tenseurs...")

        for _ in tqdm(range(num_tensors), desc="Déquanfication des tenseurs"):
            tensor_name = read_string(f)

            # Dimensions
            num_dims = read_data(f, '<I')
            shape = tuple(read_data(f, '<I') for _ in range(num_dims))

            quant_type = read_data(f, '<I')

            # Lecture des données
            if quant_type == Q_TYPE_Q4_0:
                scale = read_data(f, '<f')
                data_len = read_data(f, '<I')
                quant_bytes = f.read(data_len)

                # Déquantification et conversion en Tenseur PyTorch
                tensor = dequantize_q4_0(scale, quant_bytes, shape)

            elif quant_type == Q_TYPE_FLOAT32:
                data_len = read_data(f, '<I')
                data_bytes = f.read(data_len)
                data_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
                tensor = torch.from_numpy(data_np).to(DEVICE)

            else:
                raise ValueError(f"Type de quantification inconnu: {quant_type}")

            loaded_state_dict[tensor_name] = tensor

        # 3. Injection des poids déquantifiés dans le modèle PyTorch
        # Renommage des clés pour correspondre aux noms des modules dans AricaModelRuntime
        state_dict = {}
        key_mapping = {
            "attention.W.weight": "attn_w.weight",
            "attention.W.bias": "attn_w.bias",
            "attention.U.weight": "attn_u.weight",
            "attention.U.bias": "attn_u.bias",
            "attention.V.weight": "attn_v.weight",
        }

        for k, v in loaded_state_dict.items():
            if k in key_mapping:
                state_dict[key_mapping[k]] = v
            else:
                state_dict[k] = v

        model.load_state_dict(state_dict, strict=True)
        print("  ✅ Tenseurs chargés et injectés.")

        return model

# =================================================================
# IV. EXÉCUTION
# =================================================================

if __name__ == '__main__':
    if not os.path.exists(ARICA_FILE):
        print(f"ERREUR: Fichier Arica '{ARICA_FILE}' non trouvé. Exécutez le script de conversion d'abord.")
    else:
        try:
            # 1. Chargement et Déquantification
            loaded_model = load_arica_model(ARICA_FILE)

            # 2. Utilisation pour la Génération
            test_prompt = "Why is the capital of France important?"

            print("\n--- 🧠 Début de la Génération ---")
            print(f"Prompt: {test_prompt}")

            # Utilisation de la génération simple
            response = loaded_model.generate(
                prompt=test_prompt,
                max_length=80,
                temperature=0.8
            )

            print("\nRéponse Arica:")
            print("-" * 50)
            print(response)
            print("-" * 50)

        except (EOFError, ValueError) as e:
            print(f"\nERREUR FATALE LORS DE LA LECTURE DU FICHIER ARICA : {e}")
```