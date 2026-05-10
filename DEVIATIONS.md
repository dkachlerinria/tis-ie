# Deviations from the Paper (arXiv:2602.14696, Llama 2 7B experiment)

This file tracks where our setup intentionally or practically differs from the paper's
exact Llama 2 7B configuration (Figure 7 / Table 3). These should be resolved or
re-evaluated before any final publication comparison.

---

## 1. LoRA for final SFT (practical)

**Paper:** Full fine-tuning of Llama 2 7B (no LoRA), ~42GB VRAM required.
**Ours:** LoRA (r=128, α=512, dropout=0.1, all-linear, merged before saving) to fit
on A100 40GB.
**Impact:** Absolute accuracy numbers will differ. Relative ordering (LESS > random)
is expected to hold since both methods see the same model and training procedure.

---

## 2. Candidate pool size: END_INDEX=10000 (practical)

**Paper:** Full 197K tulu-v2 candidate pool.
**Ours:** First 10K samples only.
**Impact:** Selection diversity is limited. At budget=2500 the signal should still be
visible. Extend END_INDEX toward 197K once we confirm the pipeline works.

---

## 3. Warmup: 5K samples × 1 epoch, 1 checkpoint (compute trade-off)

**Paper:** 10K samples × 4 epochs = ~312 warmup gradient steps, 4 checkpoints,
LR-weighted multi-checkpoint averaging.
**Ours:** 5K samples × 1 epoch = ~39 warmup gradient steps, 1 checkpoint, no
averaging (weight trivially 1.0 — mathematically clean).

**Rationale:** With a single checkpoint the warmup duration matters directly. The
final SFT also runs ~39 gradient steps (2500 samples × 2 epochs / eff. batch 128),
so 5K×1ep puts the warmup reference point at a comparable training depth.
The multi-checkpoint logic remains in compute_less_similarity.py and can be
activated by increasing WARMUP_EPOCHS and passing additional CKPT_STEPS.

---

## 4. Budget: 2500 only (time)

**Paper:** Five budget levels — 500, 1000, 2500, 5000, 10000.
**Ours:** 2500 only for now. Extend via the --sizes flag in select_less.sh.

---

## 5. Seed: 0 (now matching paper)

Changed from 42 → 0 to match the paper. ✓
