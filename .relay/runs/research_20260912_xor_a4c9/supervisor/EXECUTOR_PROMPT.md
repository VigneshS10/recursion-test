# Executor Prompt

You are the executor for relay research run `research_20260912_xor_a4c9` in repository `VigneshS10/recursion-test` on branch `main`.

Before doing any experiment, read these authoritative files from the repository:

- `.relay/runs/research_20260912_xor_a4c9/supervisor/TASK.md`
- `.relay/runs/research_20260912_xor_a4c9/supervisor/task_manifest.json`

Verify that the manifest `run_id` is exactly `research_20260912_xor_a4c9` and copy its `task_nonce` exactly into your final handoff. Treat `TASK.md` as authoritative if anything in this prompt is less specific.

Implement and run the XOR activation experiment described in `TASK.md`. Keep it minimal and reproducible. Use the four XOR points, a `2 -> 4 -> 1` MLP, compare ReLU versus tanh, Adam with learning rate 0.01, full-batch training for exactly 200 steps, and paired seeds 0 through 19. Keep every experimental choice matched except the hidden activation. Do not tune after seeing results.

For every seed and activation, measure final loss, predictions/probabilities on all four points, final accuracy, whether 100% accuracy was reached, and the first step at which 100% accuracy was reached. Preserve a lightweight loss trajectory so optimization behavior can be inspected. Compute aggregate success fraction, mean/median final loss, mean final accuracy, and median first-success step among successful runs.

Commit your evidence under exactly:

`.relay/runs/research_20260912_xor_a4c9/executor/`

At minimum commit:

- `RESULT.md` — what you implemented, exact settings, aggregate results, important per-seed observations, whether the hypothesis was supported/rejected/unresolved, and any unexpected findings.
- `metrics.json` — machine-readable aggregate and per-seed metrics.
- `xor_experiment.py` — the exact experiment code that produced the result.
- `handoff.json` — relay handoff metadata.

If you generate a small plot, also commit it in the same executor directory. Do not write research evidence outside this run's executor directory.

Generate a fresh `evidence_nonce` yourself. `handoff.json` should contain at least:

```json
{
  "version": 1,
  "run_id": "research_20260912_xor_a4c9",
  "task_nonce": "<copy exactly from supervisor/task_manifest.json>",
  "evidence_nonce": "<fresh random token>",
  "status": "complete",
  "result": "RESULT.md",
  "artifacts": ["metrics.json", "xor_experiment.py"]
}
```

Add any generated plot filenames to `artifacts`.

When finished, push the executor evidence to branch `main`. Then reply to me with the executor commit SHA and a very short summary only. Do not ask me to trust the chat summary: the committed files are authoritative, and I will read them directly from GitHub for the review stage.
