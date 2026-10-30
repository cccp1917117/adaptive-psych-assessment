"""Synthetic-data prototype: masked VAE + fixed questionnaire item selection.

Not a validated psychological scale, diagnostic tool, or adaptive testing policy.
Run from this directory: python demo.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MaskedVAE(nn.Module):
    """Encode observed response values AND their mask; decode item categories."""

    def __init__(self, items: int, categories: int = 5, latent: int = 4):
        super().__init__()
        if items < 2 or categories < 2 or latent < 1:
            raise ValueError("Need items >= 2, categories >= 2, latent >= 1.")
        self.items, self.categories = items, categories
        self.encoder = nn.Sequential(nn.Linear(2 * items, 64), nn.ReLU())
        self.mu, self.logvar = nn.Linear(64, latent), nn.Linear(64, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, items * categories)
        )

    def forward(self, answers: torch.Tensor, observed: torch.Tensor):
        # A true score of zero must remain distinguishable from a missing answer.
        values = answers.float() / (self.categories - 1)
        h = self.encoder(torch.cat((values * observed, observed.float()), dim=1))
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decoder(z).reshape(-1, self.items, self.categories), mu, logvar


def synthetic_responses(n: int = 600, items: int = 18, seed: int = 42) -> np.ndarray:
    """Generate artificial 0-4 responses, NOT responses to any named real scale."""
    if n < 30 or items < 3:
        raise ValueError("Use at least 30 participants and 3 items.")
    rng = np.random.default_rng(seed)
    traits = rng.normal(size=(n, 3))
    loadings = rng.normal(0, 0.12, size=(3, items))
    loadings[np.arange(items) % 3, np.arange(items)] += 1.0
    continuous = traits @ loadings + rng.normal(0, 0.4, size=(n, items))
    continuous += rng.normal(0, 0.2, size=(1, items))
    return np.digitize(continuous, [-0.8, -0.25, 0.25, 0.8]).astype(np.int64)


def split_indices(n: int, seed: int):
    """Participant-disjoint 60/20/20 train/selection-validation/test split."""
    if n < 30:
        raise ValueError("Need at least 30 rows.")
    order = np.random.default_rng(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return order[:a], order[a:b], order[b:]


def random_mask(rows: int, items: int) -> torch.Tensor:
    """Each row has at least one observed and one unobserved item."""
    ranks = torch.rand(rows, items).argsort(dim=1).argsort(dim=1)
    counts = torch.randint(1, items, (rows, 1))
    return ranks < counts


def fit(model: MaskedVAE, train: torch.Tensor, epochs: int, beta: float = 0.02):
    """Masked-denoising reconstruction + beta-weighted KL; CPU reference code."""
    if epochs < 1 or beta < 0:
        raise ValueError("epochs must be positive and beta nonnegative.")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(train)).split(64):
            batch = train[idx]
            logits, mu, logvar = model(batch, random_mask(len(batch), model.items))
            reconstruction = F.cross_entropy(logits.transpose(1, 2), batch.long())
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            loss = reconstruction + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append({"epoch": epoch + 1, "loss": total / len(train)})
    model.eval()
    return history


@torch.no_grad()
def reconstruct(model: MaskedVAE, answers: torch.Tensor, selected: list[int]):
    """Plug-in posterior-mean prediction; preserve genuinely observed answers."""
    if len(selected) != len(set(selected)) or any(i < 0 or i >= model.items for i in selected):
        raise ValueError("Selected item indices must be unique and in range.")
    model.eval()
    observed = torch.zeros_like(answers, dtype=torch.bool)
    observed[:, selected] = True
    logits, _, _ = model(answers, observed)
    scores = torch.arange(model.categories, dtype=logits.dtype, device=logits.device)
    expected = (logits.softmax(-1) * scores).sum(-1)
    return torch.where(observed, answers.float(), expected)


def select_items(model: MaskedVAE, validation: torch.Tensor, count: int) -> list[int]:
    """Choose ONE fixed short form, using validation only, never test responses."""
    if not 1 <= count < model.items:
        raise ValueError("short-form size must be in [1, number of items - 1].")
    selected = []
    for _ in range(count):
        candidates = [i for i in range(model.items) if i not in selected]
        errors = [
            (reconstruct(model, validation, selected + [i]) - validation).square().mean().item()
            for i in candidates
        ]
        selected.append(candidates[int(np.argmin(errors))])
    return selected


def metrics(prediction: torch.Tensor, truth: torch.Tensor, selected: list[int]):
    omitted = [i for i in range(truth.shape[1]) if i not in selected]
    if not omitted:
        raise ValueError("Need unadministered items for evaluation.")
    return {
        "unadministered_item_rmse": (prediction[:, omitted] - truth[:, omitted]).square().mean().sqrt().item(),
        "all_item_rmse_including_observed": (prediction - truth).square().mean().sqrt().item(),
        # Mean-score agreement is only a synthetic diagnostic, NOT construct validity.
        "full_mean_score_rmse": (prediction.mean(1) - truth.mean(1)).square().mean().sqrt().item(),
    }


def run_demo(n: int = 600, items: int = 18, short: int = 6, epochs: int = 20, seed: int = 42):
    if not 1 <= short < items:
        raise ValueError("short must be between 1 and items - 1.")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    data = torch.from_numpy(synthetic_responses(n, items, seed))
    indices = split_indices(n, seed + 1)
    train, validation, test = (data[i] for i in indices)
    model = MaskedVAE(items)
    history = fit(model, train, epochs)
    selected = select_items(model, validation, short)
    random_items = np.random.default_rng(seed + 2).choice(items, short, replace=False).tolist()
    results = {}
    means = train.float().mean(0)
    for name, subset in (("greedy", selected), ("random", random_items)):
        results[f"vae_{name}"] = metrics(reconstruct(model, test, subset), test.float(), subset)
        baseline = means.expand(len(test), -1).clone()
        baseline[:, subset] = test[:, subset].float()
        results[f"train_mean_{name}"] = metrics(baseline, test.float(), subset)
    return {
        "status": "SYNTHETIC DEMO ONLY; no real-scale or clinical validation",
        "config": {"n": n, "items": items, "short": short, "epochs": epochs, "seed": seed},
        "split_sizes": {key: len(i) for key, i in zip(("train", "validation", "test"), indices)},
        "selected_item_indices_zero_based": selected,
        "random_item_indices_zero_based": random_items,
        "training_history": history,
        "test_metrics": results,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--items", type=int, default=18)
    parser.add_argument("--short", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_demo.json"))
    args = parser.parse_args()
    try:
        result = run_demo(args.n, args.items, args.short, args.epochs, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"])
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved run record to {args.out}")


if __name__ == "__main__":
    main()
