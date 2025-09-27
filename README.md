## Session 6 Notebooks: Scenario 1 → 2 → 3

<!-- OLD: Intro without explicit final target -->
This document explains what changed across the three MNIST scenarios and why those changes improved accuracy. It also provides quick steps to run the notebooks and a compact comparison table.

**Final target**: Achieve ≥99.4% test accuracy in < 15 epochs and maintain ≥99.4% consistently through the 15th epoch, with model size under 8k parameters.

### What’s here
- `notebooks/erav4_session6_scenario_1.ipynb`
- `notebooks/erav4_session6_scenario_2.ipynb`
- `notebooks/erav4_session6_scenario_3.ipynb`

All three scenarios use MNIST with light data augmentation and a compact CNN trained end-to-end using a Global Average Pooling (GAP) based classifier head.

## Dataset and preprocessing
- **Transforms (train)**: RandomApply(CenterCrop(22), p=0.1) → Resize(28,28) → RandomRotation(±15°) → ToTensor → Normalize(mean=0.1307, std=0.3081)
- **Transforms (test)**: ToTensor → Normalize
- Minor differences:
  - Scenario 1 uses `fill=0` for rotations; Scenarios 2/3 use `fill=(1,)` which slightly changes border pixels when rotating.

## Common training/evaluation setup
- Loss: CrossEntropyLoss on raw logits (Scenarios 2/3 explicitly return logits; Scenario 1 returns `log_softmax` but still uses CE in training cell).
- Metrics: Accuracy and average loss on the test set.

## Scenario-by-scenario deep dive

### Scenario 1: Strong baseline with Adam + StepLR (~98.9%)
- **Model**:
  - 6 convolutional blocks with BatchNorm and Dropout.
  - Two `MaxPool2d(2,2)` layers for spatial downsampling.
  - GAP to produce 10 channels → flatten to 10 classes.
  - Uses `F.log_softmax` in `forward`.
- **Regularization**: Dropout around 0.05–0.10; modest augmentation.
- **Training**:
  - Optimizer: Adam (lr=0.001)
  - Scheduler: StepLR(step_size=15, gamma=0.1)
  - Epochs: 20, Batch size: 512
- **Outcome**: Test accuracy climbs to about **98.9%** (peaks ~98.98%).
- **Observations**:
  - Large batch and a coarse StepLR schedule converge stably but may land in a flatter-yet-not-optimal region.
  - Parameter count is higher than necessary for MNIST given the target, leaving room to tighten the architecture.

### Scenario 2: Parameter-efficient CNN + OneCycle (SGD) (≈99.2%)
- **Design goals** (notebook notes):
  1) Add a layer after GAP, 2) Reduce parameter count under ~8k.
- **Architecture changes**:
  - Narrower channels (8→10→16) with 1×1 transition layers to control capacity.
  - Structured downsampling via two `MaxPool2d(2,2)` layers.
  - Classification head simplified to GAP → flatten to 10 logits (the optional post-GAP 1×1 head is present but not used in `forward`).
  - `forward` returns raw logits (no `log_softmax`).
- **Regularization**: Dropout=0.1 (slightly stronger than Scenario 1).
- **Training**:
  - Optimizer: SGD (momentum=0.9)
  - Scheduler: OneCycleLR with cosine anneal (epochs=16, `max_lr≈0.08`)
  - Batch size: 512
- **Outcome**: Test accuracy reaches about **99.2%**.
- **Why accuracy improved**:
  - **OneCycleLR** (warmup + cosine cooldown) exposes the model to a wider, well-shaped LR trajectory, helping it discover better minima faster than a coarse StepLR.
  - **Capacity control** via 1×1 transitions and narrower channels reduces overfitting and focuses representational power.
  - Returning **logits** pairs correctly with CrossEntropyLoss (numerically stable and conventional), avoiding any potential mismatch with pre-applied log-softmax.

### Scenario 3: Head refinement + smaller batch + tuned regularization (≈99.5%)
- **Architecture refinements**:
  - Post-GAP classification is now explicit: GAP → 1×1 conv (`conv10`) → flatten to 10 logits (used in `forward`).
  - Early layers use `padding=0` in places to increase effective receptive field growth through the stack (more spatial reduction via valid-conv).
  - Channel layout adjusted (peaks at 16, later reduced to 12 before the 10-class head) for a better bias-variance balance.
- **Regularization and optimization**:
  - Dropout reduced to **0.025** (Scenario 2 used 0.1). On MNIST, overly strong dropout can underfit; dialing it down preserves signal.
  - Batch size reduced to **64** (from 512), which typically increases gradient noise and can improve generalization.
  - Optimizer switched to **Adam** with **OneCycleLR** (`max_lr≈0.01`, epochs=16), which combines adaptive updates with the beneficial OneCycle schedule.
- **Outcome**: Test accuracy reaches about **99.5%** (peaks ~99.49%).
- **Why accuracy improved further**:
  - **Explicit 1×1 post-GAP head** improves the last-mile mapping from global features to class logits.
  - **Smaller batch** injects regularizing noise into gradient estimates, often yielding better generalization on small datasets.
  - **Lower dropout** avoids underfitting while augmentations still regularize.
  - **Adam + OneCycle** provides adaptive per-parameter step sizes while still following a well-shaped LR schedule.

## Targets and analysis per scenario

| Scenario | Model size | GAP | Post-GAP layer | Optimizer | Base LR | Max LR | Scheduler       | Batch | Epochs | Dropout | Peak test acc |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| 1 | Larger baseline | Yes | No | Adam | 0.001 | — | StepLR(15, γ=0.1) | 512 | 20 | ~0.05–0.10 | ~98.98% |
| 2 | < 8k params (target) | Yes | No (defined, unused) | SGD (mom=0.9) | 0.025 | ~0.08 | OneCycle (cos) | 512 | 16 | 0.10 | ~99.21% |
| 3 | Compact (≲8k) | Yes | Yes (1×1 conv) | Adam | 0.001 | ~0.01 | OneCycle (cos) | 64 | 16 | 0.025 | ~99.49% |

Notes:
- Parameter count is reduced from Scenario 1 to 2 via narrower channels and 1×1 transitions; Scenario 3 keeps the model compact while refining the head.
- All scenarios use similar augmentations; Scenario 3 benefits more from smaller batch and tuned dropout.

## Practical guidance and takeaways
- Prefer returning **logits** from the model and use `CrossEntropyLoss`.
- Use **OneCycleLR** for small-to-medium datasets; it tends to outperform simple step schedules.
- Control **capacity** (narrow channels + 1×1 transitions) before adding depth.
- Tune **batch size** and **dropout** jointly; smaller batches often generalize better, so you may be able to reduce dropout.
- Keep **GAP** as the default classifier connection on small images; add a **1×1 conv head** if you need a bit more expressivity.

## How to run
1. Open any notebook under `session6/notebooks/` in your Jupyter environment.
2. Install dependencies if needed (PyTorch, torchvision, matplotlib, torchsummary, tqdm).
3. Run all cells. On a modern CPU/GPU, each scenario should complete within minutes.

## Possible next steps
- Try label smoothing (e.g., 0.05) and MixUp/CutMix to test robustness.
- Experiment with **SGD + OneCycle** in Scenario 3 as a comparison to Adam + OneCycle.
- Replace ReLU with **Mish/SiLU**; test slightly deeper stacks under the same param budget.
- Add **weight decay** sweeps (1e-4…1e-2) with OneCycle to probe generalization further.


