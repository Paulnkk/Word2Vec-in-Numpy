import numpy as np


class Word2VecNumPy:
    def __init__(self, vocab_size, embed_dim=100, learning_rate=0.01):
        self.v_size = vocab_size
        self.d = embed_dim
        self.lr = learning_rate

        # Initialize weights uniformly between -0.5/embed_dim and 0.5/embed_dim
        self.W_in = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def train_step(self, target_id, pos_context_id, neg_indices):
        # 1. Forward pass
        u_t = self.W_in[target_id]  # (D,)

        # Extract all context vectors (positive + negatives).
        all_context_ids = np.append(pos_context_id, neg_indices)
        v_contexts = self.W_out[all_context_ids]  # (1+K, D)

        # Scores: dot product of target with each context.
        scores = np.dot(v_contexts, u_t)  # (1+K,)
        predictions = self.sigmoid(scores)

        # Labels: 1 for first (positive), 0 for negatives.
        labels = np.zeros(len(all_context_ids))
        labels[0] = 1.0

        # 2. Gradients
        errors = predictions - labels  # (1+K,)
        grad_W_out = np.outer(errors, u_t)
        grad_W_in = np.dot(errors, v_contexts)  # (D,)

        # 3. SGD update
        self.W_out[all_context_ids] -= self.lr * grad_W_out
        self.W_in[target_id] -= self.lr * grad_W_in

        # Binary cross-entropy for reporting.
        return -np.sum(
            labels * np.log(predictions + 1e-10) + (1 - labels) * np.log(1 - predictions + 1e-10)
        )
