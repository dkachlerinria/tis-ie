# Influence-Spearman Data Partitioning

**Critical:** All data partitioning is defined by config variables in `runs/influence_spearman/config_influence.sh`. This document verifies **zero data leakage** between eval and training sets.

## Eval Set (Used for Final Scoring)

**BBH Anchors:** `[0:NUM_ANCHORS]` (e.g., `[0:100]`)
- Used by ALL methods: ground_truth, LESS, embedding, LoGRA, random, influcoder
- Fixed, identical across methods ✓

**Tulu Train Pool:** `[0:END_INDEX]` (e.g., `[0:1000]`)
- Used by ALL methods for training labels
- Fixed, identical across methods ✓

## Method-Specific Training Data (Disjoint from Eval)

### Influcoder Gradient Stocking
**BBH Training Anchors:** `[NUM_ANCHORS : NUM_ANCHORS+INFLUCODER_N_TRAIN_A]`
- e.g., `[100:2100]` with default 2000 training anchors
- **Disjoint from eval** ✓

**BBH Eval Anchors (for encoder validation during training):** `[NUM_ANCHORS+INFLUCODER_N_TRAIN_A : NUM_ANCHORS+INFLUCODER_N_TRAIN_A+INFLUCODER_N_EVAL_A]`
- e.g., `[2100:2600]` with default 500 eval anchors
- **Disjoint from eval** ✓

**Tulu Training Pool:** `[END_INDEX : END_INDEX+INFLUCODER_N_TRAIN_P]`
- e.g., `[1000:3000]` with default 2000 training samples
- **Disjoint from eval** ✓

**Tulu Eval Pool (for encoder training labels):** `[END_INDEX+INFLUCODER_N_TRAIN_P : END_INDEX+INFLUCODER_N_TRAIN_P+INFLUCODER_N_EVAL_P]`
- e.g., `[3000:5000]` with default 2000 eval samples
- **Critical:** Used DURING encoder training as labels/validation set
- **Disjoint from final eval** ✓

### Ground Truth, LESS, Embedding, LoGRA, Random
These methods train fresh LoRA adapters on the eval set itself, so they have no separate training data.
All use `[0:NUM_ANCHORS]` anchors and `[0:END_INDEX]` train samples.

## Verification Checklist

```python
# Config values (example from config_influence_tiny.sh)
NUM_ANCHORS = 100                    # eval anchors
END_INDEX = 1000                     # eval train samples
INFLUCODER_N_TRAIN_A = 2000          # influcoder train anchors
INFLUCODER_N_EVAL_A = 500            # influcoder eval anchors
INFLUCODER_N_TRAIN_P = 4000          # influcoder train pool
INFLUCODER_N_EVAL_P = 2000           # influcoder eval pool

# Ranges
eval_anchors_bbh = [0, 100]                                    # ✓
eval_train_tulu = [0, 1000]                                    # ✓
influcoder_train_anchors = [100, 2100]                         # ✓ disjoint from eval_anchors_bbh
influcoder_eval_anchors = [2100, 2600]                         # ✓ disjoint from eval_anchors_bbh
influcoder_train_pool = [1000, 5000]                           # ✓ disjoint from eval_train_tulu
influcoder_eval_pool = [5000, 7000]                            # ✓ disjoint from eval_train_tulu

# Invariants
assert eval_anchors_bbh[1] <= influcoder_train_anchors[0]      # [0:100] < [100:∞)
assert eval_train_tulu[1] <= influcoder_train_pool[0]          # [0:1000] < [1000:∞)
assert influcoder_train_pool[1] <= influcoder_eval_pool[0]     # [1000:5000] < [5000:∞)
```

## Why This Matters

If influcoder's `eval_pool` overlapped with the final eval set, the encoder would be optimized on labels (`true_scores_eval`) computed from samples it will later score. This inflates correlation metrics.

**Before fix:** eval_pool = `[0:2000]` → overlaps [0:1000] → data leakage
**After fix:** eval_pool = `[1000:3000]` → disjoint from [0:1000] → ✓ clean

## Ranges by Variable

| Variable | BBH Range | Tulu Range | Purpose |
|----------|-----------|-----------|---------|
| `NUM_ANCHORS` | [0, NUM_ANCHORS] | — | Final eval anchors (all methods) |
| `END_INDEX` | — | [0, END_INDEX] | Final eval train pool (all methods) |
| `INFLUCODER_N_TRAIN_A` | [NUM_ANCHORS, NUM_ANCHORS+N] | — | Influcoder encoder training anchors |
| `INFLUCODER_N_EVAL_A` | [NUM_ANCHORS+N, NUM_ANCHORS+N+M] | — | Influcoder encoder validation anchors |
| `INFLUCODER_N_TRAIN_P` | — | [END_INDEX, END_INDEX+N] | Influcoder encoder training pool |
| `INFLUCODER_N_EVAL_P` | — | [END_INDEX+N, END_INDEX+N+M] | Influcoder encoder validation pool |
