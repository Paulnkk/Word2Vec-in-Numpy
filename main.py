import os
import urllib.request
import zipfile

import numpy as np

from model import Word2VecNumPy
from preprocessing import TextProcessor


def load_text8(max_words=50000):
    """Download, unzip, and return a slice of the text8 dataset."""
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = "text8.zip"

    # 1) Download if needed.
    if not os.path.exists(zip_path):
        print(f"Downloading Text8 (31MB) from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    # 2) Extract and read.
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        raw_data = zip_file.read(zip_file.namelist()[0]).decode("utf-8")

    # 3) Split and slice.
    words = raw_data.split()
    print(f"Dataset loaded. Total words in Text8: {len(words)}")
    return words[:max_words]


def get_similarity(word1, word2, processor, model):
    """Cosine similarity between two words using input embeddings."""
    if word1 not in processor.word2id or word2 not in processor.word2id:
        raise KeyError(f"At least one word is out-of-vocabulary: '{word1}', '{word2}'")
    id1, id2 = processor.word2id[word1], processor.word2id[word2]
    v1, v2 = model.W_in[id1], model.W_in[id2]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    # Optional env overrides for faster experiments.
    max_words = int(os.getenv("MAX_WORDS", "50000"))
    embed_dim = int(os.getenv("EMBED_DIM", "50"))
    epochs = int(os.getenv("EPOCHS", "2"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.01"))
    window_size = int(os.getenv("WINDOW_SIZE", "2"))
    neg_samples = int(os.getenv("NEGATIVE_SAMPLES", "5"))

    # 1) Get data (automatic download on first run).
    words = load_text8(max_words=max_words)
    sentences = [" ".join(words)]  # TextProcessor expects a list of strings.

    # 2) Build vocabulary and target-context pairs.
    processor = TextProcessor(window_size=window_size, negative_samples=neg_samples)
    processor.build_vocab(sentences)
    pairs = processor.get_target_context_pairs(sentences)
    print(f"Vocabulary size: {processor.vocab_size}")
    print(f"Training pairs: {len(pairs)}")

    # 3) Initialize model.
    model = Word2VecNumPy(
        vocab_size=processor.vocab_size,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
    )

    # 4) Train.
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(pairs)
        for target, pos_context in pairs:
            negative_ids = processor.get_negative_samples(processor.neg_samples)
            total_loss += model.train_step(target, pos_context, negative_ids)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # 5) Interactive similarity query loop.
    print("\nTraining complete. Enter word pairs to check similarity.")
    print(f"Sample vocabulary: {list(processor.word2id.keys())[:30]}")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("Enter two words (space-separated): ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        parts = user_input.split()
        if len(parts) != 2:
            print("Please enter exactly two words separated by a space.")
            continue
        w1, w2 = parts
        try:
            score = get_similarity(w1, w2, processor, model)
            print(f"Similarity ({w1}, {w2}): {score:.4f}")
        except KeyError as e:
            print(e)


if __name__ == "__main__":
    main()
