#!/usr/bin/env python3
"""Paired 20-seed XOR activation reliability experiment.

Uses NumPy because PyTorch is not installed in the bundled runtime.
"""
import json
import math
from pathlib import Path

import numpy as np


X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])
LR = 0.01
STEPS = 200
SEEDS = list(range(20))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def bce_with_logits(logits, targets):
    # Stable mean BCE: max(x,0) - x*y + log(1 + exp(-abs(x)).
    return float(np.mean(np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))))


def init_model(seed):
    rng = np.random.RandomState(seed)
    # Identical parameter shapes and initialization procedure for both activations.
    w1 = rng.randn(2, 4) * math.sqrt(2.0 / 2.0)
    b1 = np.zeros((1, 4))
    w2 = rng.randn(4, 1) * math.sqrt(2.0 / 4.0)
    b2 = np.zeros((1, 1))
    return [w1, b1, w2, b2]


def run(seed, activation):
    p = init_model(seed)
    m = [np.zeros_like(v) for v in p]
    v = [np.zeros_like(v) for v in p]
    losses = []
    first_success = None
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, STEPS + 1):
        w1, b1, w2, b2 = p
        z1 = X @ w1 + b1
        if activation == "relu":
            h = np.maximum(z1, 0.0)
            dh = (z1 > 0.0).astype(float)
        else:
            h = np.tanh(z1)
            dh = 1.0 - h * h
        logits = h @ w2 + b2
        probs = sigmoid(logits)
        loss = bce_with_logits(logits, Y)
        losses.append(loss)

        dlogits = (probs - Y) / X.shape[0]
        dw2 = h.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        dhidden = (dlogits @ w2.T) * dh
        dw1 = X.T @ dhidden
        db1 = np.sum(dhidden, axis=0, keepdims=True)
        grads = [dw1, db1, dw2, db2]

        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
            mh = m[i] / (1.0 - beta1 ** step)
            vh = v[i] / (1.0 - beta2 ** step)
            p[i] -= LR * mh / (np.sqrt(vh) + eps)

        # First success is measured after the completed optimizer update.
        w1, b1, w2, b2 = p
        z1 = X @ w1 + b1
        h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
        final_probs = sigmoid(h @ w2 + b2).reshape(-1)
        preds = (final_probs >= 0.5).astype(int)
        acc = float(np.mean(preds == Y.reshape(-1)))
        if acc == 1.0 and first_success is None:
            first_success = step

    w1, b1, w2, b2 = p
    z1 = X @ w1 + b1
    h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
    logits = h @ w2 + b2
    probs = sigmoid(logits).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    final_loss = bce_with_logits(logits, Y)
    final_acc = float(np.mean(preds == Y.reshape(-1)))
    return {
        "seed": seed,
        "activation": activation,
        "final_loss": final_loss,
        "final_probabilities": [float(x) for x in probs],
        "final_predictions": [int(x) for x in preds],
        "final_accuracy": final_acc,
        "final_success": bool(final_acc == 1.0),
        "first_success_step": first_success,
        "losses_every_step": losses,
    }


def main():
    rows = []
    for seed in SEEDS:
        # Each activation resets to the same seed immediately before construction.
        rows.append(run(seed, "relu"))
        rows.append(run(seed, "tanh"))

    by = {a: [r for r in rows if r["activation"] == a] for a in ("relu", "tanh")}
    summary = {}
    for a, rs in by.items():
        successes = [r for r in rs if r["final_success"]]
        first_steps = [r["first_success_step"] for r in successes]
        losses = [r["final_loss"] for r in rs]
        summary[a] = {
            "seeds_reaching_100_by_step_200": len(successes),
            "fraction_reaching_100_by_step_200": len(successes) / len(rs),
            "seeds_ending_at_100": len(successes),
            "fraction_ending_at_100": len(successes) / len(rs),
            "mean_final_loss": float(np.mean(losses)),
            "median_final_loss": float(np.median(losses)),
            "mean_final_accuracy": float(np.mean([r["final_accuracy"] for r in rs])),
            "median_first_success_step_successful_only": int(np.median(first_steps)) if first_steps else None,
        }

    pairs = []
    for seed in SEEDS:
        rr = next(r for r in by["relu"] if r["seed"] == seed)
        tt = next(r for r in by["tanh"] if r["seed"] == seed)
        pairs.append({
            "seed": seed,
            "relu_final_success": rr["final_success"],
            "tanh_final_success": tt["final_success"],
            "relu_final_accuracy": rr["final_accuracy"],
            "tanh_final_accuracy": tt["final_accuracy"],
            "relu_final_loss": rr["final_loss"],
            "tanh_final_loss": tt["final_loss"],
            "relu_first_success_step": rr["first_success_step"],
            "tanh_first_success_step": tt["first_success_step"],
        })
    win_counts = {
        "only_tanh": sum((not p["relu_final_success"]) and p["tanh_final_success"] for p in pairs),
        "only_relu": sum(p["relu_final_success"] and (not p["tanh_final_success"]) for p in pairs),
        "both": sum(p["relu_final_success"] and p["tanh_final_success"] for p in pairs),
        "neither": sum((not p["relu_final_success"]) and (not p["tanh_final_success"]) for p in pairs),
    }
    result = {
        "experiment": "XOR ReLU vs tanh reliability",
        "runtime": "numpy_fallback_no_pytorch",
        "dataset": {"inputs": X.tolist(), "labels": Y.reshape(-1).astype(int).tolist()},
        "architecture": "2->4->1",
        "loss": "BCEWithLogitsLoss-equivalent-stable-formula",
        "optimizer": {"name": "Adam", "learning_rate": LR, "steps": STEPS, "full_batch": True},
        "seeds": SEEDS,
        "summary": summary,
        "paired_win_counts": win_counts,
        "paired_rows": pairs,
        "runs": rows,
    }
    out = Path(__file__).with_name("metrics.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_win_counts": win_counts}, indent=2))


if __name__ == "__main__":
    main()

