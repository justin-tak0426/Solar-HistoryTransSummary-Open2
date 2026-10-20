# Approved Solar Open 2 Training Design

Status: design approved for implementation and smoke validation only.

Date: 2026-07-27

Production training is not approved by this document. A second explicit approval is
required after the implementation, deterministic replay checks, and GPU smoke gates
described below have passed.

## Scope

This design replaces the 4,000-step, 3,072-token sampling plan with one continuous
run over a deterministic globally shuffled augmented deck, with a maximum horizon
of two epochs and a fail-closed early-stopping policy. It preserves the existing
Solar Open 2 LoRA, loss, prompt, output-only, and sealed evaluation contracts unless
this document explicitly changes them.

The fixed optimization settings are:

- LoRA rank 32 and alpha 64
- peak learning rate `1e-4`
- cosine schedule over the full 18,631-step horizon with minimum ratio `0.1`
  (learning-rate floor `1e-5`)
- 3% warmup, derived as 558 optimizer steps
- effective batch size 16 occurrences
- maximum sequence length 9,600 tokens
- hard-term target-reading token loss weight 2.5
- expert-corrected data represented eight times per epoch
- NEFTune alpha 5
- a maximum horizon of exactly two augmented-deck epochs

The 1.0- and 1.5-epoch checkpoints are validation candidates from this single
18,631-step cosine horizon. They are not equivalent to independent shorter runs
whose cosine schedules end at 1.0 or 1.5 epochs. The first authorized stopping
boundary is step 13,973; step 9,316 supplies the minimum-epoch full-validation
baseline but is not itself a stopping boundary.

The scheduler is stateless: the learning rate for each one-indexed update is
derived from the durably completed optimizer-step count, rather than advanced
through mutable scheduler state. A resume therefore recomputes the next update's
position from `completed_steps`. Every data-bearing update receives a positive
learning rate; after warmup, cosine decay is bounded below by `1e-5`.

## Dataset Arithmetic

The pinned training set contains 142,713 unique rows.

| Task | Silver rows | Expert rows | Unique rows | Augmented occurrences per epoch |
| --- | ---: | ---: | ---: | ---: |
| Translation | 96,304 | 740 | 97,044 | 102,224 |
| Summarization | 45,504 | 165 | 45,669 | 46,824 |
| Total | 141,808 | 905 | 142,713 | 149,048 |

Each expert row occurs eight times in each epoch: its original occurrence plus
seven additional occurrences. Each silver row occurs once per epoch. Across two
epochs, every silver row therefore occurs twice and every expert row occurs 16
times.

The two-epoch totals are:

- translation: 204,448 occurrences
- summarization: 93,648 occurrences
- total: 298,096 occurrences
- optimizer steps: `298096 / 16 = 18631`

Exact deck coverage determines the natural training mixture:

- translation: 68.5846170361%
- summarization: 31.4153829639%

The former 65/35 values are not training sampling probabilities. They remain only
as the separately named validation selection weights. Enforcing 65/35 during
training would require repeating or dropping occurrences and would violate the
approved exact-deck contract.

## Deterministic Deck Contract

Each epoch is one deterministic global permutation of all 149,048 augmented
occurrences. Occurrences must have stable identities containing at least the row
ID, task, quality, and expert occurrence ordinal. Epoch permutations must be
derived deterministically from the pinned seed and epoch number.

The resumable sampler state must bind:

- state-contract version
- dataset and eligibility hashes
- deck-construction policy and counts
- epoch seed, epoch index, and position
- current deck or a reproducible deck hash
- total consumed occurrences by task and quality

An uninterrupted 298,096-draw simulation and simulations resumed at arbitrary
positions must produce the same occurrence-sequence hash. A run that reaches the
maximum horizon must assert 204,448 translation occurrences, 93,648 summarization
occurrences, all 142,713 unique rows covered, each expert row seen 16 times, and
zero skipped occurrences. An early-stopped run must instead prove that every
occurrence in the deterministic prefix through its durable stopping step was
consumed exactly once, with zero skipped occurrences.

The existing draw-then-skip behavior is incompatible with this contract. Before
deck construction, every pinned training row must be tokenized with the production
prompt and tokenizer. Any row longer than 9,600 tokens fails the gate. The same
eligibility audit applies to validation: either all rows are eligible or the run
must not describe the result as full validation. Changing the approved occurrence
counts requires another design approval.

## Optimizer And Epoch Boundaries

One epoch is not divisible by the effective batch:

`149048 / 16 = 9315.5`

Similarly, 1.5 epochs lands at optimizer step 13,973.25. Exact 1.0- and 1.5-epoch
model snapshots are therefore impossible while simultaneously preserving batch
size 16 and exact deck coverage.

The approved nearest-boundary candidates are:

| Target | Target occurrences | Step | Consumed occurrences | Offset |
| --- | ---: | ---: | ---: | ---: |
| 1.0 epoch | 149,048 | 9,316 | 149,056 | +8 |
| 1.5 epochs | 223,572 | 13,973 | 223,568 | -4 |
| 2.0 epochs | 298,096 | 18,631 | 298,096 | 0 |

At an exact half-step tie, the later boundary is used, which explains step 9,316.
Reports and artifacts must record the target epoch, actual consumed occurrences,
and offset. Only step 18,631 may be described as an exact epoch boundary.

## Dynamic Microbatch Target

Every optimizer update always represents 16 consecutive deck occurrences.
Length bucketing may reorder or group work only within that fixed 16-occurrence
optimizer group. It must not pull rows from later optimizer groups, because doing
so would change the deterministic deck and resume contract.

The target microbatch sizes are:

| Token length | Target microbatch size |
| --- | ---: |
| 1-1,024 | 8 |
| 1,025-2,048 | 4 |
| 2,049-4,096 | 2 |
| 4,097-9,600 | 1 |

Gradients must be normalized by the 16 occurrences in the optimizer group, not by
the number of dynamically formed microbatches. Grouping and execution order must
remain deterministic after resume.

These sizes are targets, not an authorization to run production. Every bucket must
pass the production-model GPU smoke. A bucket that fails must fall back to a lower
microbatch size, the resulting policy must be recorded in a new smoke receipt, and
the effective batch must remain 16.

## Validation And Best Selection

A fixed stratified validation subset is evaluated every 1,000 optimizer steps.
Full validation is evaluated only at steps 9,316, 13,973, and 18,631. Explicit
candidate steps are part of the validation schedule even though they are not
multiples of 1,000.

The selection loss is the fixed task-level weighted mean:

`0.65 * translation_loss + 0.35 * summarization_loss`

These weights are independent of the natural training-deck mixture. Every completed
candidate must use the same pinned validation rows, row order, eligibility set,
prompt, tokenizer, maximum length, and loss implementation.

The exact early-stopping contract is:

- monitor `selection_loss` in `min` mode
- no stopping before 1.0 epoch or optimizer step 9,316
- fixed-subset baseline: the last fixed-subset result before the minimum step
  (step 9,000 under the pinned 1,000-step schedule)
- subset patience: three evaluations
- subset material improvement: the larger of absolute `0.001` and relative
  `0.0025 * best`
- a mandatory full-validation confirmation at step 13,973
- full material improvement: the larger of absolute `0.0005` and relative
  `0.001 * best`
- continue when the composite full loss or either individual task loss materially
  improves
- earliest stop step 13,973 and latest stop step 18,631
- restore the best completed full-validation candidate, with an exact tie resolved
  in favor of the earlier optimizer step
- sealed test and human-gold data never act as monitors

Subset patience can be exhausted before step 13,973, but it cannot stop training
without the mandatory full-validation confirmation. If confirmation shows no
material composite or per-task improvement, training stops at step 13,973. This
saves 4,658 of the maximum 18,631 updates, or approximately 25.0013%. Otherwise it
continues to the step-18,631 maximum.

After training reaches either authorized terminal boundary, `best_adapter` is
selected by minimum selection loss over the completed full-validation candidates.
An exact loss tie selects the earlier optimizer step. The selected candidate,
candidate losses, offsets, validation hashes, adapter file hashes, stopping reason,
and terminal step must be recorded.

`final_adapter` proves the actual terminal step. It proves exact two-epoch
completion only when the terminal step is 18,631. `best_adapter` is the only tuned
adapter used by sealed test and human-gold evaluation. Test and human-gold results
must never participate in monitoring, stopping, or checkpoint selection.

## Checkpoint And Resume Contract

Recovery is due at whichever happens first:

- 100 completed optimizer steps since the last recovery generation
- four elapsed hours since the last recovery generation
- `SIGUSR1`

Each recovery generation is an immutable pair containing the adapter and the
matching optimizer, durably completed step used as the stateless scheduler
position, RNG, deterministic deck, validation history, and validation-progress
state. A small pointer may be atomically advanced only after the complete pair is
durable. Keep the newest two complete recovery generations. Retain every completed
candidate adapter regardless of normal recovery pruning.

Never delete and replace an adapter directory referenced by resumable state. A
repeated save at the same optimizer step must verify and reuse the immutable
adapter. Validation progress and history may be committed separately and
atomically.

Full validation must be resumable by candidate identity, adapter hash, validation
data/order hash, next row index, and task-level running sums and counts. Commit
progress in bounded row chunks. A preemption must resume the same validation
candidate rather than silently skip it or restart indefinitely from row zero.
Exactly one completed history record is allowed per candidate.

Best selection runs only after an authorized terminal boundary is durably
completed. Candidate checkpoints and validation history are the source of truth,
so creation of `best_adapter` must be atomic, idempotent, and reconstructable after
interruption.

## Runtime Audit And Gate

The new run has at most 18,631 optimizer steps, 4.65775 times the former
4,000-step count.
The maximum length is 3.125 times the former 3,072-token cap. Those ratios are not
a runtime estimate because actual sequence-length distribution and attention
behavior determine throughput.

Measured evidence from the prior 3,072-token smoke is retained only as a baseline:

- longest eligible training microbatch: approximately 12.574 seconds
- longest eligible validation row: approximately 3.175 seconds
- prior fixed-workload safety projection: approximately 307.3 hours

At the prior longest-validation-row rate, one 17,850-row full validation would
already take about 15.7 hours. The maximum horizon performs three full validations,
or 53,550 validation forwards, in addition to fixed-subset monitoring. The earliest
authorized stop performs two full validations. Partial validation resume is
therefore mandatory in either case.

Observed artifact sizes are approximately 533 MB per adapter and 1.067 GB per
resume state. One complete recovery generation is about 1.6 GB; two retained
generations require about 3.2 GB, excluding candidate, final, and best adapters.
Retention and free-space checks must be part of preflight.

Before a second approval, a fresh three-GPU, 9,600-token smoke must prove:

- absolute maximum row and over-limit counts for train and validation
- forward, backward, finite loss, finite gradients, and a positive-LR update
- hard-term 2.5-weight supervision and output-only masking
- target microbatch size for every length bucket, including boundary lengths
- effective batch normalization across mixed buckets
- partial-optimizer rollback on signal
- immutable recovery save, reload, and deterministic deck replay
- interrupted and resumed full validation
- all explicit candidate schedule paths, including a valid step-13,973 stop
- atomic reconstruction of `best_adapter`
- measured runtime by actual length bucket
- separate empirical p50/p90 projections for the step-13,973 earliest stop and
  step-18,631 maximum, with exact subset/full-validation event counts, saved
  updates/fraction, allocation count, and headroom
- peak memory on every visible GPU

Runtime evidence must exclude cold shape setup from steady-state estimates. Warm each
training and validation bucket once, time three synchronized repeats, and use their
median. Separately warm two deterministic deck updates and time ten updates sampled
across both epochs, reporting p50 and p90. Validation cost must be projected from the
exact subset/full-validation microbatch plans, not from longest-row time multiplied
by row count. The benchmark must restore and verify RNG, LoRA parameters, frozen
parameters, optimizer state, and gradients. Allocation atomicity passing is a
technical gate only; the resulting total-duration estimate still requires the
second explicit production approval.

Smoke and certifier constants that still encode 3,072 tokens, 4,000 steps, one
final full validation, or fixed-final-step selection must be updated and reviewed.
Evaluation and W&B upload entry points must require the hashed `best_adapter` and
the new two-epoch training run identity before production.

## Sealed Evaluation

The existing evaluation configuration remains unchanged:

- datasets: full test and human gold
- temperature 0 and top-p 1
- model context 16,384
- translation generation limit 8,192
- summarization generation limit 6,144
- corpus BLEU, mean sentence BLEU, mean ROUGE-L F1, and mean BERTScore F1

Only validation loss selects `best_adapter`. Base and tuned inference must preserve
the established translation-or-summary-body-only output contract, and test and
human-gold metrics must be reported separately.

## Approval Boundary

This document approves the design and configuration update only. It does not
approve production submission, continuation of an old 4,000-step checkpoint, or
reuse of its W&B run identity. The implementation must use a new output directory,
new state-contract version, and new W&B run identity.

The sealed implementation identities are:

- recipe `solar_open2_hardterm_fancy_v3_earlystop`
- recovery state contract v4
- runtime projection schema v5
- production output `train_2epoch_earlystop_step18631_len9600_v2`
- production W&B run ID
  `solar_open2_hardterm_2epoch_earlystop_20260727_v2`
- evaluation output `eval_2epoch_earlystop_validation_best_v2`
- evaluation W&B run ID `solar_open2_hardterm_eval_earlystop_20260727_v2`

The `step18631` substring identifies the fixed scheduler maximum, not a claim that
every conforming run reaches that step. The manifest records the actual terminal
step, stopping reason, completed epochs, and selected candidate.

The smoke entry point is `sbatch/03_train_smoke.sbatch` and writes below
`gpu_smoke_v7_9600_earlystop`. Its base and tuned inference smoke use a
16,384-token model context. A v7 receipt records
`technical_production_gate_passed=true`,
`authorization_scope=technical_smoke_gate_only`,
`human_production_approval_required=true`, and
`human_production_approval_recorded=false`. A technical PASS is not human
production approval. `sbatch/04_train_production.sbatch` is a reviewed template
only; it must not be submitted before the second approval.

No production job may be submitted until all gates above pass and a second
explicit approval is recorded.
