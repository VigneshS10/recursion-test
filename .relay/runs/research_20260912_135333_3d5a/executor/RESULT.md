# Research Run Result

RUN_ID: research_20260912_135333_3d5a

## Status

STRUCTURAL LIMITATION / BLOCKED: the requested global-versus-local Lipschitz verifier study could not be executed from the fetched repository state.

## Manifest verification

- Manifest: `.relay/runs/research_20260912_135333_3d5a/supervisor/task_manifest.json`
- `run_id` verified exactly as `research_20260912_135333_3d5a`.
- `task_nonce`: `LIP-9f3a7c2d_20260912`
- Fetched branch: `main`
- Fetched HEAD before executor write: `dd4c888`

## Repository capability audit

The tracked repository state contains 23 files, consisting of relay/protocol/test artifacts and the prior run's limitation report. It contains:

- 0 neural-policy or certificate checkpoints (`.pt`, `.pth`, `.ckpt`, `.onnx`)
- 0 Python/notebook/config implementation files (`.py`, `.ipynb`, `.yaml`, `.yml`, `.toml`)
- 0 tracked files matching `L_lip`, `tau_K`, verifier, or certificate implementation terms
- no environment/layout or verified-domain definition
- no boxwise policy, transition over-approximation, or MLP certificate
- no global Lipschitz implementation, sound local bound routine, formal verifier logs, or diagnostic Jacobian samples

Consequently, the same fixed policy, abstraction, partition, certificate checkpoint, verifier equations, and tolerances required by the packet are unavailable. Running the four bound variants would require inventing or changing experimental components, which is out of scope.

## Conditions and measurements

| condition | evaluated | formal verifier runs | boxes | result |
|---|---:|---:|---:|---|
| global-bound baseline | no | 0 | 0 | unavailable |
| existing tightened-global control | no | 0 | 0 | unavailable |
| sound box-local bound | no | 0 | 0 | unavailable |
| empirical gradient diagnostic | no | 0 | 0 | unavailable |

No global/local `L_lip`, `tau_K`, verifier margins, pass/fail transitions, runtimes, bound ratios, violation decompositions, or plots are reported because their required inputs are absent. No sampled gradient was used as a proof bound.

## Verification evidence

`git pull --ff-only origin main` completed successfully and fetched the supervisor packet. Manifest parsing and the run-ID assertion completed with exit code 0 and no error.

Tracked inventory audit at HEAD `dd4c888` returned: `tracked_count=23`, `checkpoint_count=0`, `implementation_count=0`, `lip_terms=0`.

## Scientific conclusion

This run supports a structural limitation finding only. It does not establish whether global `L_lip` is conservative, whether sound local bounds improve certifiability, or whether `tau_K` is the dominant bottleneck. A valid comparison requires the existing difficult reach-avoid instance, fixed policy and boxwise abstraction, at least one MLP certificate checkpoint, verifier implementation with the current `tau_K + L_lip` expression, and its formal/diagnostic logs. No claim about local-bound improvement or global-bound adequacy is made.

## Evidence files

- `metrics.json`: machine-readable capability audit and zero-run counts.
- `handoff.json`: required handoff with copied task nonce and generated evidence nonce.

