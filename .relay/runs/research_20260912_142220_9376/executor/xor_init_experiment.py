#!/usr/bin/env python3
"""200-seed paired XOR experiment under shared He and Xavier initialization."""
import json
import math
from pathlib import Path

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])
SEEDS = list(range(200))
STEPS = 200
LR = 0.01
TRACE_SEEDS = set(range(10))


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


def run(seed, scheme, activation, trace=False):
    p = [a.copy() for a in init_params(seed, scheme)]
    m = [np.zeros_like(a) for a in p]
    v = [np.zeros_like(a) for a in p]
    losses = [] if trace else None
    first_success = None
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        w1, bias1, w2, bias2 = p
        z1 = X @ w1 + bias1
        if activation == "relu":
            h, dh = np.maximum(z1, 0.0), (z1 > 0.0).astype(float)
        else:
            h, dh = np.tanh(z1), None
            dh = 1.0 - h * h
        logits = h @ w2 + bias2
        probs = sigmoid(logits)
        loss = bce_logits(logits)
        if trace:
            losses.append(loss)
        dlogits = (probs - Y) / len(X)
        grads = [X.T @ ((dlogits @ w2.T) * dh), np.sum((dlogits @ w2.T) * dh, axis=0, keepdims=True), h.T @ dlogits, np.sum(dlogits, axis=0, keepdims=True)]
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1.0 - b1) * g
            v[i] = b2 * v[i] + (1.0 - b2) * g * g
            mh = m[i] / (1.0 - b1 ** step)
            vh = v[i] / (1.0 - b2 ** step)
            p[i] -= LR * mh / (np.sqrt(vh) + eps)
        # Evaluate after the completed update for first-success semantics.
        z1 = X @ p[0] + p[1]
        h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
        post_probs = sigmoid(h @ p[2] + p[3]).reshape(-1)
        post_acc = float(np.mean((post_probs >= 0.5) == Y.reshape(-1)))
        if post_acc == 1.0 and first_success is None:
            first_success = step
    z1 = X @ p[0] + p[1]
    h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
    logits = h @ p[2] + p[3]
    probs = sigmoid(logits).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    final_acc = float(np.mean(preds == Y.reshape(-1)))
    out = {
        "seed": seed,
        "initialization": scheme,
        "activation": activation,
        "final_loss": bce_logits(logits),
        "final_accuracy": final_acc,
        "reached_100_by_step_200": first_success is not None,
        "ended_at_100": final_acc == 1.0,
        "first_success_step": first_success,
        "final_probabilities": [float(x) for x in probs],
    }
    if trace:
        out["losses_every_step"] = losses
    return out


def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return {"discordant_total": 0, "two_sided_p": 1.0, "definition": "exact two-sided binomial test on discordant pairs"}
    p = min(1.0, 2.0 * binom_cdf(min(b, c), n))
    return {"discordant_total": n, "two_sided_p": p, "definition": "exact two-sided binomial test on discordant pairs, null p=0.5"}


def summarize(rows, scheme, activation):
    rs = [r for r in rows if r["initialization"] == scheme and r["activation"] == activation]
    succ = [r for r in rs if r["ended_at_100"]]
    first = [r["first_success_step"] for r in succ if r["first_success_step"] is not None]
    losses = [r["final_loss"] for r in rs]
    accs = [r["final_accuracy"] for r in rs]
    counts = {str(x): sum(r["final_accuracy"] == x for r in rs) for x in (0.5, 0.75, 1.0)}
    return {
        "initialization": scheme,
        "activation": activation,
        "n": len(rs),
        "success_count": len(succ),
        "success_fraction": len(succ) / len(rs),
        "ended_at_100_count": len(succ),
        "mean_final_loss": float(np.mean(losses)),
        "median_final_loss": float(np.median(losses)),
        "mean_final_accuracy": float(np.mean(accs)),
        "median_first_success_step_successful_only": int(np.median(first)) if first else None,
        "final_accuracy_distribution": counts,
    }


def main():
    rows = []
    for scheme in ("he", "xavier"):
        for seed in SEEDS:
            # Both activations receive independently copied, identical parameters per seed/scheme.
            rows.append(run(seed, scheme, "relu", trace=seed in TRACE_SEEDS))
            rows.append(run(seed, scheme, "tanh", trace=seed in TRACE_SEEDS))
    summaries = [summarize(rows, s, a) for s in ("he", "xavier") for a in ("relu", "tanh")]
    paired = []
    paired_tests = {}
    for scheme in ("he", "xavier"):
        rr = {r["seed"]: r for r in rows if r["initialization"] == scheme and r["activation"] == "relu"}
        tt = {r["seed"]: r for r in rows if r["initialization"] == scheme and r["activation"] == "tanh"}
        only_tanh = sum((not rr[i]["ended_at_100"]) and tt[i]["ended_at_100"] for i in SEEDS)
        only_relu = sum(rr[i]["ended_at_100"] and (not tt[i]["ended_at_100"]) for i in SEEDS)
        both = sum(rr[i]["ended_at_100"] and tt[i]["ended_at_100"] for i in SEEDS)
        neither = sum((not rr[i]["ended_at_100"]) and (not tt[i]["ended_at_100"]) for i in SEEDS)
        paired_tests[scheme] = {
            "only_tanh": only_tanh,
            "only_relu": only_relu,
            "both": both,
            "neither": neither,
            "success_rate_difference_tanh_minus_relu": sum(tt[i]["ended_at_100"] for i in SEEDS) / 200.0 - sum(rr[i]["ended_at_100"] for i in SEEDS) / 200.0,
            "mcnemar_exact": mcnemar_exact(only_tanh, only_relu),
        }
        for i in SEEDS:
            paired.append({
                "initialization": scheme,
                "seed": i,
                "relu_final_loss": rr[i]["final_loss"],
                "tanh_final_loss": tt[i]["final_loss"],
                "relu_final_accuracy": rr[i]["final_accuracy"],
                "tanh_final_accuracy": tt[i]["final_accuracy"],
                "relu_ended_at_100": rr[i]["ended_at_100"],
                "tanh_ended_at_100": tt[i]["ended_at_100"],
                "relu_first_success_step": rr[i]["first_success_step"],
                "tanh_first_success_step": tt[i]["first_success_step"],
            })
    result = {
        "experiment": "XOR ReLU vs tanh initialization robustness",
        "runtime": "numpy_fallback_no_pytorch",
        "dataset": {"inputs": X.tolist(), "labels": Y.reshape(-1).astype(int).tolist()},
        "architecture": "2->4->1",
        "loss": "BCEWithLogitsLoss-equivalent-stable-formula",
        "optimizer": {"name": "Adam", "learning_rate": LR, "steps": STEPS, "full_batch": True},
        "seeds": SEEDS,
        "trace_seeds": sorted(TRACE_SEEDS),
        "summaries": summaries,
        "paired_tests": paired_tests,
        "paired_rows": paired,
        "runs": rows,
    }
    Path(__file__).with_name("metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired_tests": paired_tests}, indent=2))


if __name__ == "__main__":
    main()

