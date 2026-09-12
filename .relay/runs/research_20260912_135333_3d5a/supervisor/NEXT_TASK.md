# Next Research Task: Materialize and replay one canonical reach-avoid verification baseline

## Research question
Can the executor locate and freeze the actual existing reach-avoid policy/certificate/verifier pipeline used in the ongoing research, then replay one known baseline verification result exactly enough that subsequent ablations are scientifically meaningful and reproducible?

## Hypothesis
The experimental pipeline already exists in the broader research workspace but is not present in `VigneshS10/recursion-test`. If the canonical implementation revision and its fixed artifacts are identified and materialized without reconstruction from prose, one previously successful baseline should be replayable with matching policy behavior, verifier equations, and certification outcome. If no such implementation/artifact source is accessible, that absence must be established explicitly before further ablations are attempted.

## Experiment / implementation request
Do **not** perform a new Lipschitz, D20, repair, or certificate ablation in this task. The sole objective is to establish one immutable, executable scientific baseline that later tasks can vary one factor at a time.

1. Search the executor's available research workspace/repositories for the canonical existing implementation corresponding to the conversation's reach-avoid work. Prefer, in order:
   - the current double-integrator neural-policy -> boxwise-policy -> neural-certificate pipeline with `tau_K + L_lip`, if a known successful instance exists;
   - otherwise the previously successful continuous-discrete MiniGrid P0-P4 verifier pipeline;
   - otherwise the strongest accessible known-working reach-avoid certificate instance that uses the same current theory.
2. Do not reimplement missing theory-sensitive components from the supervisor prose. Locate the real source files and artifacts. Record the exact repository/path and source revision (commit SHA if under version control) for every component used.
3. Freeze a minimal reproducible experiment bundle containing or unambiguously referencing:
   - environment/layout and verified-domain definition;
   - policy architecture and trained policy checkpoint;
   - boxwise/discontinuous policy abstraction and partition definition, if applicable;
   - transition/next-state over-approximation used by the verifier;
   - certificate architecture and checkpoint or deterministic training configuration needed to recreate it;
   - exact certificate conditions and verifier implementation;
   - exact `tau_K`, `L_lip`, and bound computation used by the baseline, if applicable;
   - verifier tolerances, numerical precision, random seeds, and relevant dependency versions;
   - the command/script/config required to replay the baseline.
4. Replay the baseline **without changing its scientific configuration**. The purpose is provenance and reproducibility, not improvement. If harmless compatibility changes are needed to run old code, document them line-by-line and do not alter certificate semantics, policy, partition, verifier equations, or tolerances.
5. Compare the replayed result against the original known result/log if available. Check at minimum:
   - neural-policy empirical reach-avoid success;
   - boxwise-policy success if applicable;
   - formal verifier pass/fail;
   - minimum verifier margins or violation counts;
   - `tau_K` and `L_lip` values if used;
   - partition/box count;
   - certificate checkpoint identity/hash.
6. Produce an explicit provenance manifest that hashes or uniquely identifies all scientific inputs so later experiments can declare exactly which baseline they modify.
7. If no canonical implementation or required artifact can be located, stop rather than rebuilding it. Produce a dependency report listing exactly what was searched, what was found, what is missing, and the smallest user action/source access needed to unblock the work.

The task succeeds by establishing a trustworthy baseline, not by obtaining a new positive research result.

## Controls
- No changes to the baseline environment geometry, initial/goal/unsafe sets, policy, partition, certificate theory, verifier equations, bound formulas, or tolerances.
- Use the exact stored checkpoint/configuration when available rather than retraining.
- If retraining is absolutely required because only a deterministic training recipe exists, use the recorded seed/config and clearly distinguish reproduced versus original checkpoint results.
- Preserve the original source revision. Do not silently patch scientific code in place.
- Any compatibility patch must be minimal, separately recorded, and shown not to change the mathematical computation.
- Do not infer `tau_K`, `L_lip`, verifier formulas, or certificate conditions from conversation text when the actual implementation is unavailable.

## Expected measurements
Record:

- Canonical source repository/path and exact revision.
- Inventory of all baseline scientific inputs and cryptographic hashes where practical.
- Environment/layout identifier and verified-domain definition.
- Policy/certificate checkpoint identifiers.
- Partition resolution and total box count if applicable.
- Neural-policy and boxwise-policy reach-avoid success over the baseline's standard evaluation budget; use at least 1000 rollouts if that is compatible with the existing setup.
- Formal verifier pass/fail and minimum condition margins or violation counts.
- Exact `tau_K`, `L_lip`, and combined bound contribution where present.
- Baseline runtime if readily available.
- Original-versus-replay differences for every comparable metric.
- Dependency/environment versions needed for replay.
- Any compatibility modifications and evidence that they are non-scientific.

## Completion criteria
The task is complete under either of two outcomes:

**Reproducible baseline established:** one canonical known-working reach-avoid experiment is replayed from identified source/artifacts, the formal result is reproduced within explained numerical tolerance, and a frozen provenance manifest identifies all scientific inputs needed for later matched ablations.

**Precisely blocked:** after a documented search of all accessible research sources, one or more indispensable components remain unavailable. The executor must identify the exact missing components and their expected source/location and must not substitute newly invented implementations.

Do not proceed to the global-vs-local Lipschitz comparison until the first outcome is achieved.

## Artifacts or results to preserve
- A concise `BASELINE_REPRO.md` describing the canonical source, replay command, original result, replay result, and any discrepancies.
- Machine-readable provenance manifest with source revisions, paths, hashes, seeds, dependency versions, and scientific configuration.
- Exact configs/scripts needed to invoke the replay, or precise references to immutable existing files.
- Copies or stable references for policy, boxwise abstraction/partition, certificate, verifier configuration, and environment/layout.
- Replay logs and machine-readable metrics.
- If blocked, a structured dependency/missing-artifact report including all searched locations and the minimal unblock requirement.
