#!/usr/bin/env python3
"""Continuous 2000-update paired XOR budget experiment."""
import json
import math
from pathlib import Path

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])
SEEDS = list(range(200))
CHECKPOINTS = [25, 50, 100, 200, 500, 1000, 2000]
STEPS = 2000
LR = 0.01


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def bce_logits(logits):
    return float(np.mean(np.maximum(logits, 0.0) - logits * Y + np.log1p(np.exp(-np.abs(logits)))))


def init_params(seed, scheme):
    rng = np.random.RandomState(seed)
    if scheme == "he":
        s1, s2 = math.sqrt(2.0 / 2.0), math.sqrt(2.0 / 4.0)
    elif scheme == "xavier":
        s1, s2 = math.sqrt(2.0 / (2.0 + 4.0)), math.sqrt(2.0 / (4.0 + 1.0))
    else:
        raise ValueError(scheme)
    return [rng.randn(2, 4) * s1, np.zeros((1, 4)), rng.randn(4, 1) * s2, np.zeros((1, 1))]


def evaluate(p):
    z1 = X @ p[0] + p[1]
    h = np.maximum(z1, 0.0) if p[4] == "relu" else np.tanh(z1)
    logits = h @ p[2] + p[3]
    probs = sigmoid(logits).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return bce_logits(logits), float(np.mean(preds == Y.reshape(-1))), probs


def run(seed, scheme, activation):
    p = init_params(seed, scheme)
    # Keep activation as metadata outside numeric parameter arrays.
    m = [np.zeros_like(a) for a in p]
    v = [np.zeros_like(a) for a in p]
    checkpoints = {}
    first_success = None
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        w1, bias1, w2, bias2 = p
        z1 = X @ w1 + bias1
        if activation == "relu":
            h, dh = np.maximum(z1, 0.0), (z1 > 0.0).astype(float)
        else:
            h = np.tanh(z1)
            dh = 1.0 - h * h
        logits = h @ w2 + bias2
        probs = sigmoid(logits)
        dlogits = (probs - Y) / len(X)
        dhidden = (dlogits @ w2.T) * dh
        grads = [X.T @ dhidden, np.sum(dhidden, axis=0, keepdims=True), h.T @ dlogits, np.sum(dlogits, axis=0, keepdims=True)]
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1.0 - b1) * g
            v[i] = b2 * v[i] + (1.0 - b2) * g * g
            mh = m[i] / (1.0 - b1 ** step)
            vh = v[i] / (1.0 - b2 ** step)
            p[i] -= LR * mh / (np.sqrt(vh) + eps)
        # Check first success after every completed update (not only checkpoints).
        z1_now = X @ p[0] + p[1]
        h_now = np.maximum(z1_now, 0.0) if activation == "relu" else np.tanh(z1_now)
        probs_now = sigmoid(h_now @ p[2] + p[3]).reshape(-1)
        if first_success is None and float(np.mean((probs_now >= 0.5) == Y.reshape(-1))) == 1.0:
            first_success = step
        # Evaluate only after completed updates at requested checkpoints.
        if step in CHECKPOINTS:
            z1 = X @ p[0] + p[1]
            h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
            logits = h @ p[2] + p[3]
            cp = sigmoid(logits).reshape(-1)
            acc = float(np.mean((cp >= 0.5) == Y.reshape(-1)))
            checkpoints[str(step)] = {"loss": bce_logits(logits), "accuracy": acc, "reached_100": acc == 1.0}
    z1 = X @ p[0] + p[1]
    h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
    probs = sigmoid(h @ p[2] + p[3]).reshape(-1)
    final_acc = float(np.mean((probs >= 0.5) == Y.reshape(-1)))
    return {"seed": seed, "initialization": scheme, "activation": activation, "first_success_step": first_success, "checkpoints": checkpoints, "final_2000_probabilities": [float(x) for x in probs], "final_2000_loss": bce_logits(h @ p[2] + p[3]), "final_2000_accuracy": final_acc}


def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def exact_mcnemar(only_tanh, only_relu):
    n = only_tanh + only_relu
    if n == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * binom_cdf(min(only_tanh, only_relu), n))
    return {"discordant_total": n, "two_sided_p": p, "definition": "exact two-sided binomial test on discordant pairs, null p=0.5"}


def main():
    rows = [run(seed, scheme, activation) for scheme in ("he", "xavier") for seed in SEEDS for activation in ("relu", "tanh")]
    summaries, paired = [], {}
    for scheme in ("he", "xavier"):
        for activation in ("relu", "tanh"):
            rs = [r for r in rows if r["initialization"] == scheme and r["activation"] == activation]
            by_cp = {}
            for cp in CHECKPOINTS:
                vals = [r["checkpoints"][str(cp)] for r in rs]
                firsts = [r["first_success_step"] for r in rs if r["first_success_step"] is not None and r["first_success_step"] <= cp]
                by_cp[str(cp)] = {"success_count": sum(v["reached_100"] for v in vals), "success_fraction": sum(v["reached_100"] for v in vals) / 200.0, "mean_accuracy": float(np.mean([v["accuracy"] for v in vals])), "mean_loss": float(np.mean([v["loss"] for v in vals])), "median_loss": float(np.median([v["loss"] for v in vals])), "median_first_success_step_among_successful_by_checkpoint": int(np.median(firsts)) if firsts else None, "final_accuracy_distribution": {str(x): sum(v["accuracy"] == x for v in vals) for x in (0.5, 0.75, 1.0)}}
            summaries.append({"initialization": scheme, "activation": activation, "checkpoints": by_cp, "never_success_by_2000": sum(r["first_success_step"] is None for r in rs)})
        rr = {r["seed"]: r for r in rows if r["initialization"] == scheme and r["activation"] == "relu"}
        tt = {r["seed"]: r for r in rows if r["initialization"] == scheme and r["activation"] == "tanh"}
        cp_tests = {}
        for cp in CHECKPOINTS:
            only_tanh = sum((not rr[i]["checkpoints"][str(cp)]["reached_100"]) and tt[i]["checkpoints"][str(cp)]["reached_100"] for i in SEEDS)
            only_relu = sum(rr[i]["checkpoints"][str(cp)]["reached_100"] and (not tt[i]["checkpoints"][str(cp)]["reached_100"]) for i in SEEDS)
            both = sum(rr[i]["checkpoints"][str(cp)]["reached_100"] and tt[i]["checkpoints"][str(cp)]["reached_100"] for i in SEEDS)
            neither = sum((not rr[i]["checkpoints"][str(cp)]["reached_100"]) and (not tt[i]["checkpoints"][str(cp)]["reached_100"]) for i in SEEDS)
            cp_tests[str(cp)] = {"only_tanh": only_tanh, "only_relu": only_relu, "both": both, "neither": neither, "success_rate_difference_tanh_minus_relu": (sum(tt[i]["checkpoints"][str(cp)]["reached_100"] for i in SEEDS) - sum(rr[i]["checkpoints"][str(cp)]["reached_100"] for i in SEEDS)) / 200.0, "mcnemar_exact": exact_mcnemar(only_tanh, only_relu)}
        paired[scheme] = cp_tests
    result = {"experiment": "XOR activation training-budget dependence", "runtime": "numpy_fallback_no_pytorch", "dataset": {"inputs": X.tolist(), "labels": Y.reshape(-1).astype(int).tolist()}, "architecture": "2->4->1", "optimizer": {"name": "Adam", "learning_rate": LR, "full_batch": True, "continuous_steps": STEPS}, "seeds": SEEDS, "checkpoints": CHECKPOINTS, "summaries": summaries, "paired_tests": paired, "runs": rows}
    Path(__file__).with_name("metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired_tests": paired}, indent=2))


if __name__ == "__main__":
    main()

