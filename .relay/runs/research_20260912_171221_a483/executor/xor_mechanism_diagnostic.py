#!/usr/bin/env python3
"""Instrumented paired XOR diagnostic for ReLU failure mechanisms."""
import json
import math
from pathlib import Path

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])
SEEDS = list(range(200))
CHECKPOINTS = [0, 25, 50, 100, 200, 500, 1000, 2000]
STEPS = 2000
LR = 0.01


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def bce(logits):
    return float(np.mean(np.maximum(logits, 0.0) - logits * Y + np.log1p(np.exp(-np.abs(logits)))))


def init_params(seed, scheme):
    rng = np.random.RandomState(seed)
    if scheme == "he":
        s1, s2 = math.sqrt(2 / 2), math.sqrt(2 / 4)
    else:
        s1, s2 = math.sqrt(2 / 6), math.sqrt(2 / 5)
    return [rng.randn(2, 4) * s1, np.zeros((1, 4)), rng.randn(4, 1) * s2, np.zeros((1, 1))]


def forward(p, activation):
    z1 = X @ p[0] + p[1]
    h = np.maximum(z1, 0.0) if activation == "relu" else np.tanh(z1)
    logits = h @ p[2] + p[3]
    probs = sigmoid(logits).reshape(-1)
    acc = float(np.mean((probs >= 0.5) == Y.reshape(-1)))
    return z1, h, logits, probs, bce(logits), acc


def gradients(p, activation):
    z1, h, logits, probs, loss, acc = forward(p, activation)
    dh = (z1 > 0.0).astype(float) if activation == "relu" else 1.0 - h * h
    dlogits = (probs.reshape(-1, 1) - Y) / len(X)
    dhidden = (dlogits @ p[2].T) * dh
    return [X.T @ dhidden, np.sum(dhidden, axis=0, keepdims=True), h.T @ dlogits, np.sum(dlogits, axis=0, keepdims=True)], (z1, h, logits, probs, loss, acc)


def diagnostics(p, initial, activation, grad_norms, update_norms, step):
    z1, h, logits, probs, loss, acc = forward(p, activation)
    out = {
        "loss": loss,
        "accuracy": acc,
        "gradient_norms": {"W1": grad_norms[0], "b1": grad_norms[1], "W2": grad_norms[2], "b2": grad_norms[3]},
        "update_norms": {"W1": update_norms[0], "b1": update_norms[1], "W2": update_norms[2], "b2": update_norms[3]},
        "W1_displacement": float(np.linalg.norm(p[0] - initial[0])),
        "W2_displacement": float(np.linalg.norm(p[2] - initial[2])),
        "hidden_output_variance_per_unit": [float(x) for x in np.var(h, axis=0)],
        "hidden_output_variance_mean": float(np.mean(np.var(h, axis=0))),
    }
    if activation == "relu":
        bits = (z1 > 0.0).astype(int)
        signatures = ["".join(str(int(v)) for v in row) for row in bits]
        out.update({
            "activation_pattern": bits.tolist(),
            "dead_unit_count": int(np.sum(np.all(bits == 0, axis=0))),
            "always_active_count": int(np.sum(np.all(bits == 1, axis=0))),
            "distinct_activation_signature_count": len(set(signatures)),
            "activation_signatures": signatures,
        })
    return out


def run(seed, scheme, activation):
    p = init_params(seed, scheme)
    initial = [a.copy() for a in p]
    m, v = [np.zeros_like(a) for a in p], [np.zeros_like(a) for a in p]
    zero = [0.0] * 4
    grads, state = gradients(p, activation)
    grad0 = [float(np.linalg.norm(g)) for g in grads]
    cps = {"0": diagnostics(p, initial, activation, grad0, zero, 0)}
    first_success = 0 if state[5] == 1.0 else None
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        # grads/state are the values immediately before this optimizer update.
        grads, state = gradients(p, activation)
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            delta = LR * mh / (np.sqrt(vh) + eps)
            p[i] -= delta
        # First success is checked after every completed update.
        post = forward(p, activation)
        if first_success is None and post[5] == 1.0:
            first_success = step
        if step in CHECKPOINTS:
            # Recompute gradients at the post-update checkpoint state; these are
            # the norms before the next optimizer update. The update norm is the
            # actual Adam displacement just applied at this step.
            gcp, _ = gradients(p, activation)
            un = [float(np.linalg.norm(LR * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps))) for i in range(4)]
            cps[str(step)] = diagnostics(p, initial, activation, [float(np.linalg.norm(g)) for g in gcp], un, step)
    final = cps[str(STEPS)]
    return {"seed": seed, "initialization": scheme, "activation": activation, "first_success_step": first_success, "ended_at_100": final["accuracy"] == 1.0, "checkpoints": cps, "final_probabilities": [float(x) for x in forward(p, activation)[3]]}


def aggregate_group(rows, scheme, activation, group_seeds):
    rs = [r for r in rows if r["initialization"] == scheme and r["activation"] == activation and r["seed"] in group_seeds]
    out = {"n": len(rs)}
    for cp in CHECKPOINTS:
        vals = [r["checkpoints"][str(cp)] for r in rs]
        d = {"mean_accuracy": float(np.mean([x["accuracy"] for x in vals])), "mean_loss": float(np.mean([x["loss"] for x in vals])), "median_loss": float(np.median([x["loss"] for x in vals])), "mean_hidden_output_variance": float(np.mean([x["hidden_output_variance_mean"] for x in vals])), "median_W1_gradient_norm": float(np.median([x["gradient_norms"]["W1"] for x in vals])), "mean_W1_gradient_norm": float(np.mean([x["gradient_norms"]["W1"] for x in vals])), "median_W1_update_norm": float(np.median([x["update_norms"]["W1"] for x in vals])), "median_W1_displacement": float(np.median([x["W1_displacement"] for x in vals])), "median_W2_displacement": float(np.median([x["W2_displacement"] for x in vals]))}
        if activation == "relu":
            d.update({"mean_dead_unit_count": float(np.mean([x["dead_unit_count"] for x in vals])), "median_dead_unit_count": float(np.median([x["dead_unit_count"] for x in vals])), "fraction_any_dead": float(np.mean([x["dead_unit_count"] > 0 for x in vals])), "mean_always_active_count": float(np.mean([x["always_active_count"] for x in vals])), "mean_distinct_activation_signatures": float(np.mean([x["distinct_activation_signature_count"] for x in vals]))})
        out[str(cp)] = d
    return out


def conditional_failure(rows, scheme):
    rr = [r for r in rows if r["initialization"] == scheme and r["activation"] == "relu"]
    out = {}
    for cp in (0, 25):
        key = "dead_unit_count" if cp == 0 or cp == 25 else "dead_unit_count"
        vals = {}
        for k in sorted(set(r["checkpoints"][str(cp)][key] for r in rr)):
            sub = [r for r in rr if r["checkpoints"][str(cp)][key] == k]
            vals[str(k)] = {"n": len(sub), "failure_probability": float(np.mean([not r["ended_at_100"] for r in sub]))}
        out[f"failure_given_dead_count_step_{cp}"] = vals
        if cp in (0, 25):
            sigvals = {}
            for k in sorted(set(r["checkpoints"][str(cp)]["distinct_activation_signature_count"] for r in rr)):
                sub = [r for r in rr if r["checkpoints"][str(cp)]["distinct_activation_signature_count"] == k]
                sigvals[str(k)] = {"n": len(sub), "failure_probability": float(np.mean([not r["ended_at_100"] for r in sub]))}
            out[f"failure_given_signature_count_step_{cp}"] = sigvals
    return out


def main():
    rows = [run(seed, scheme, activation) for scheme in ("he", "xavier") for seed in SEEDS for activation in ("relu", "tanh")]
    summaries, conditionals, discordant = {}, {}, {}
    for scheme in ("he", "xavier"):
        good = {r["seed"] for r in rows if r["initialization"] == scheme and r["activation"] == "relu" and r["ended_at_100"]}
        bad = set(SEEDS) - good
        summaries[scheme] = {"relu_success_group": aggregate_group(rows, scheme, "relu", good), "relu_failure_group": aggregate_group(rows, scheme, "relu", bad), "tanh_reference_on_relu_success_group": aggregate_group(rows, scheme, "tanh", good), "tanh_reference_on_relu_failure_group": aggregate_group(rows, scheme, "tanh", bad)}
        conditionals[scheme] = conditional_failure(rows, scheme)
        tanh_good = {r["seed"] for r in rows if r["initialization"] == scheme and r["activation"] == "tanh" and r["ended_at_100"]}
        discordant_seeds = sorted(tanh_good - good)
        discordant[scheme] = {"count": len(discordant_seeds), "seeds": discordant_seeds, "relu_group": aggregate_group(rows, scheme, "relu", set(discordant_seeds)), "tanh_group": aggregate_group(rows, scheme, "tanh", set(discordant_seeds))}
    result = {"experiment": "XOR ReLU mechanism diagnosis", "runtime": "numpy_fallback_no_pytorch", "dataset": {"inputs": X.tolist(), "labels": Y.reshape(-1).astype(int).tolist()}, "architecture": "2->4->1", "optimizer": {"name": "Adam", "learning_rate": LR, "steps": STEPS, "full_batch": True}, "seeds": SEEDS, "checkpoints": CHECKPOINTS, "summaries_by_initialization": summaries, "conditional_failure": conditionals, "discordant_tanh_success_relu_failure": discordant, "runs": rows}
    Path(__file__).with_name("metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "relu_success": {s: summaries[s]["relu_success_group"]["n"] for s in summaries}, "relu_failure": {s: summaries[s]["relu_failure_group"]["n"] for s in summaries}, "discordant": {s: discordant[s]["count"] for s in discordant}}, indent=2))


if __name__ == "__main__":
    main()

