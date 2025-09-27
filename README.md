## Session 6 Notebooks: Scenario 1 → 2 → 3

<!-- OLD: Intro without explicit final target -->
This document explains what changed across the three MNIST scenarios and why those changes improved accuracy. It also provides quick steps to run the notebooks and a compact comparison table.

**Final target**: Achieve ≥99.4% test accuracy in < 15 epochs and maintain ≥99.4% consistently through the 15th epoch, with model size under 8k parameters.

### What’s here
- `notebooks/erav4_session6_scenario_1.ipynb` with `model1.py` and `train1.py`
- `notebooks/erav4_session6_scenario_2.ipynb` with `model2.py` and `train2.py`
- `notebooks/erav4_session6_scenario_3.ipynb` with `model3.py` and `train3.py`

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

<!-- OLD: Generic snapshot values copied verbatim -->
#### Snapshot — Targets, Results, Analysis
- **Targets**: Build the skeleton with proper layers and stay under 100k parameters (baseline establishment).
- **Results**: 9,822 parameters; train max 98.35% (epoch 18); test max 99.14% (epochs 17–19), ~98.9% by epoch ~11.
- **Analysis**: Strong structure and stable convergence but below 99.4%; schedule is coarse and head minimal. Next, tighten capacity and upgrade LR scheduling.
- **File Link**: [pidooma.com](https://www.pidooma.com)

### Scenario 2: Parameter-efficient CNN + OneCycle (SGD) (≈99.2%)
- **Design goals** (notebook notes):
  <!-- OLD: "1) Add a layer after GAP, 2) Reduce parameter count under ~8k." and a duplicated "2)" -->
  1) Add a layer after GAP  
  2) Reduce parameter count under ~8k  
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

<!-- OLD: Generic snapshot values copied verbatim -->
#### Snapshot — Targets, Results, Analysis
- **Targets**: Go under 8k parameters and recover/improve accuracy using a better schedule.
- **Results**: 5,512 parameters; train max 98.59% (epoch 13); test max 99.24% (epoch 15).
- **Analysis**: OneCycle with SGD improved generalization and peak accuracy; still shy of 99.4%. Post-GAP head and milder dropout are the next levers.
- **File Link**: [pidooma.com](https://www.pidooma.com)

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

<!-- OLD: Generic snapshot values copied verbatim -->
#### Snapshot — Targets, Results, Analysis
- **Targets**: Achieve ≥99.4% in <15 epochs, maintain ≥99.4% through epoch 15, under 8k parameters.
- **Results**: 7,592 parameters; train max 99.14% (epoch 16); test 99.42–99.54% by epochs 11–15, sustained ≥99.4% to epoch 15.
- **Analysis**: Post-GAP 1×1 head + smaller batch + reduced dropout + Adam OneCycle delivered the final push and stability.
- **File Link**: [pidooma.com](https://www.pidooma.com)


## Targets and analysis per scenario

| Scenario | Model size | GAP | Post-GAP layer | Optimizer | Base LR | Max LR | Scheduler       | Batch | Epochs | Dropout | Peak test acc |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 9,822 params | Yes | No | Adam | 0.001 | — | StepLR(15, γ=0.1) | 512 | 20 | ~0.05–0.10 | ~98.98% |
| 2 | 5,512 params | Yes | No (defined, unused) | SGD (mom=0.9) | 0.025 | ~0.08 | OneCycle (cos) | 512 | 16 | 0.10 | ~99.21% |
| 3 | 7,592 params | Yes | Yes (1×1 conv) | Adam | 0.001 | ~0.01 | OneCycle (cos) | 64 | 16 | 0.025 | ~99.49% |

Notes:
- Parameter count is reduced from Scenario 1 to 2 via narrower channels and 1×1 transitions; Scenario 3 keeps the model compact while refining the head.
- All scenarios use similar augmentations; Scenario 3 benefits more from smaller batch and tuned dropout.

## Accuracy extremes by scenario


| Scenario | Test accuracy MIN (epoch) | Test accuracy MAX (epoch) | Train accuracy MIN (epoch) | Train accuracy MAX (epoch) |
|---|---|---|---|---|
| 1 | 94.82% (epoch 1) | 99.14% (epochs 17, 19) | 72.33% (epoch 1) | 98.35% (epoch 18) |
| 2 | 89.99% (epoch 1, first run) / 92.08% (epoch 1, OneCycle run) | 99.24% (epoch 15, OneCycle run) | 66.47% (epoch 1) | 98.59% (epoch 13) |
| 3 | 97.60% (epoch 3) | 99.54% (epoch 15) | 87.82% (epoch 1) | 99.14% (epoch 16) |






## The story: how I reached Scenario 3

### Prologue — aiming for 99.4% under 8k
I set a concrete target early: break ≥99.4% test accuracy in fewer than 15 epochs, keep it ≥99.4% through epoch 15, and stay under 8k parameters. This constraint guided every change.

### Chapter 1 — Scenario 1: the dependable baseline
I began with a familiar CNN: stacks of 3×3 convs with BN and small dropout, two pooling stages, GAP to 10 channels, and Adam + StepLR. It quickly reached ~98.9% and then plateaued. That stall said two things: the network had enough capacity for MNIST, but the learning-rate schedule and the last-mile mapping (GAP → flatten) likely limited further gains.

What I learned:
- Coarse StepLR didn’t explore and anneal as effectively; it converged safely, not optimally.
- The head had minimal expressivity; a post-GAP 1×1 could help without adding many params.

### Chapter 2 — Scenario 2: slimming down and pacing better
Next I reduced parameters with narrower channels and 1×1 transitions, and I switched to OneCycleLR with SGD. Training felt livelier and accuracy climbed to ~99.21%. Under 8k params was achieved without sacrificing performance.

What I learned:
- OneCycleLR’s warmup and cosine cooldown helped optimization discover better basins.
- The compact trunk generalized well, but the classifier head still felt tight, and dropout at 0.10 looked slightly heavy for MNIST.

Decision point:
- Add a light post-GAP 1×1 conv for a smarter head.
- Reduce dropout and try smaller batches to increase gradient noise (regularization) while keeping schedule benefits.

### Chapter 3 — Scenario 3: the last-mile refinement
I added the post-GAP 1×1 conv, reduced dropout to 0.025, shrank batch size to 64, and paired the setup with Adam + OneCycle at a modest max_lr (~0.01). The result crossed 99.4% around epochs 11–12 and stayed there through epoch 15, satisfying all constraints (params=7,592).

Why it worked:
- The 1×1 head improved the mapping from global features to logits with negligible params.
- Smaller batches plus reduced dropout struck a better bias–variance balance.
- OneCycle retained its convergence benefits; Adam at lower max_lr behaved smoothly with the smaller batch.

### Epilogue — what I’d keep and try next
- Keep: GAP-based trunk, parameter-efficient design, post-GAP 1×1 head, OneCycle schedule, smaller batch.
- Try next: label smoothing, mild weight decay sweeps, compare SGD+OneCycle vs Adam+OneCycle on this final architecture.

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


