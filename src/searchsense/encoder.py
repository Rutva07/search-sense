"""A corpus-trained semantic context encoder; no pretrained downloads required."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ContextEncoder(nn.Module):
    def __init__(self, size: int, dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(size, dim, padding_idx=0)
        self.projection = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, size)

    def forward(self, tokens):
        mask = (tokens != 0).unsqueeze(-1)
        mean = (self.embedding(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.tanh(self.projection(mean))


class Encoder:
    def __init__(self, vocabulary: list[str], model: ContextEncoder):
        self.words = vocabulary
        self.ids = {word: i for i, word in enumerate(vocabulary)}
        self.model = model.eval()
        with torch.inference_mode():
            self.vectors = F.normalize(model.output.weight, dim=1).numpy().astype("float32")
            self.output_weights = model.output.weight.numpy().copy()
            self.output_bias = model.output.bias.numpy().copy()

    def encode(self, texts: list[str]) -> np.ndarray:
        ids = [[self.ids.get(word, 1) for word in text.split()[-16:]] or [1] for text in texts]
        tokens = torch.zeros((len(ids), max(map(len, ids))), dtype=torch.long)
        for i, row in enumerate(ids):
            tokens[i, :len(row)] = torch.tensor(row)
        with torch.inference_mode():
            return self.model(tokens).numpy().astype("float32")

    def save(self, directory: Path):
        torch.save(self.model.state_dict(), directory / "encoder.pt")
        (directory / "vocabulary.json").write_text(json.dumps(self.words))

    @classmethod
    def load(cls, directory: Path):
        words = json.loads((directory / "vocabulary.json").read_text())
        state = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        model = ContextEncoder(len(words), state["embedding.weight"].shape[1])
        model.load_state_dict(state)
        return cls(words, model)


def train_encoder(event_factory, directory: Path, epochs=5, vocab_size=8000, dim=64, seed=42, max_pairs=200000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frequencies = Counter()
    for _, _, query in event_factory():
        frequencies.update(query.split())
    words = ["<pad>", "<unk>"] + [w for w, n in frequencies.most_common(vocab_size - 2) if n >= 2]
    model = ContextEncoder(len(words), dim)
    ids = {w: i for i, w in enumerate(words)}
    # Bounded uniform reservoir over all training prefix/next-word pairs.
    pairs = []
    seen = 0
    for _, _, query in event_factory():
        tokens = query.split()
        for j in range(1, len(tokens)):
            if tokens[j] not in ids:
                continue
            item = ([ids.get(w, 1) for w in tokens[max(0, j - 16):j]], ids[tokens[j]])
            seen += 1
            if len(pairs) < max_pairs:
                pairs.append(item)
            else:
                k = int(rng.integers(seen))
                if k < max_pairs:
                    pairs[k] = item
    if not pairs:
        raise ValueError("No usable next-word training pairs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for offset in range(0, len(pairs), 512):
            if offset == 0:
                order = rng.permutation(len(pairs))
            batch = [pairs[i] for i in order[offset:offset + 512]]
            tokens = torch.zeros((len(batch), max(len(x[0]) for x in batch)), dtype=torch.long)
            for i, (context, _) in enumerate(batch):
                tokens[i, :len(context)] = torch.tensor(context)
            target = torch.tensor([item[1] for item in batch])
            loss = F.cross_entropy(model.output(model(tokens)), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(pairs))
        print(json.dumps({"encoder_epoch": epoch + 1, "loss": losses[-1]}), flush=True)
    encoder = Encoder(words, model)
    encoder.save(directory)
    return encoder, {"vocabulary_size": len(words), "training_pairs": len(pairs), "all_usable_pairs": seen,
                     "epochs": epochs, "losses": losses, "dimension": dim, "seed": seed}
