# ViT Teacher to Hailo-8 Student Report

## Scope

This document summarizes the current repository context and proposes an edge deployment path:

- strong server-side `ViT` teacher training on the CSIRO biomass dataset
- a `Hailo-8` friendly single-frame student model
- a `Raspberry Pi 5` video-stream fusion layer that uses multiple frames to approach teacher accuracy

The goal is not to deploy the current dual-stream `DINOv3-L` model directly on `Hailo-8`. The goal is to preserve as much predictive quality as possible while meeting edge constraints.

## Current Context

This repository already supports the score-first training workflow for the CSIRO biomass dataset:

- dual-stream image splitting from the original wide image into left/right crops
- supervised training with `5` regression heads and `5` interval-classification heads
- multi-`fold` and multi-`seed` server sweeps
- `OOF` aggregation and model selection
- `TTA` and checkpoint averaging at inference time

Relevant local docs:

- [reproduction.md](/home/winbeau/Selab-gh/SilkRoad-EcoVision/docs/reproduction.md)
- [server-training.md](/home/winbeau/Selab-gh/SilkRoad-EcoVision/docs/server-training.md)

Current training objective:

- maximize performance on the current CSIRO dataset first
- treat distillation as a later deployment path, not as the primary score-optimization path

That decision is still correct. A student model should be derived from the best server-side teacher after the main model search is stable.

## Why Direct Deployment Is Not the Right Target

The current strongest candidates in this project are large vision backbones such as:

- `DINOv3 ViT-L`
- `DINOv2 ViT-L / ViT-G`
- `SigLIP SO400M`

These are poor direct deployment targets for `Raspberry Pi 5 + Hailo-8` because:

- `Hailo-8` is not a general-purpose GPU
- deployment is constrained by the `Hailo Dataflow Compiler` supported graph and operators
- large transformer-style models are much harder to compile and operate efficiently on this stack
- the current repository model is also dual-stream, which further increases edge inference cost

Practical implication:

- do not try to force the current teacher architecture onto `Hailo-8`
- distill teacher behavior into a smaller `Hailo-friendly` student

## Recommended Edge Architecture

Use a split architecture:

1. `Hailo-8` runs a single-frame lightweight student model.
2. `Raspberry Pi 5` maintains a short temporal window over the video stream.
3. The Pi fuses multiple frame predictions into one final biomass estimate.

This is a better hardware match than trying to compile a heavy temporal or transformer stack directly for `Hailo-8`.

### Proposed Runtime Graph

```text
video frame
  -> preprocess
  -> Hailo-8 student
       -> 5 regression outputs
       -> 5 interval-classification outputs
       -> optional low-dim embedding
  -> Pi 5 temporal fusion over last K frames
       -> final biomass prediction
       -> optional confidence / stability score
```

## Why Multi-Frame Edge Inference Can Beat Single-Frame Edge Inference

This only helps if nearby frames contain new information, not just duplicates.

When that condition holds, multi-frame fusion improves edge quality for four reasons:

- it reduces random single-frame noise from blur, compression, exposure shifts, or partial occlusion
- it accumulates slightly different viewpoints over time
- it enforces temporal consistency for a quantity that should not jump sharply frame to frame
- it lets a cheap temporal fusion head recover part of the information lost when compressing the teacher into a much smaller student

For this biomass task, that matters because plant texture, density, and coverage can be ambiguous in a single frame, while a short window often gives more stable cues.

## Can a ViT Teacher Be Distilled into a Hailo-Friendly Student?

Yes, but the correct question is not whether the student can reproduce the teacher architecture. It cannot. The real question is whether it can reproduce enough of the teacher's behavior.

Cross-architecture distillation is a standard approach:

- `ViT` teacher
- lightweight `CNN` or lightweight `hybrid` student

The student does not copy the teacher block-by-block. It learns:

- final predictions
- class/logit structure
- projected feature relations
- ranking and confidence behavior

Expected outcome:

- a well-designed student can get meaningfully closer to a strong single teacher
- it will usually still trail the best ensemble
- it can still be the right tradeoff if the edge target requires real-time inference and low power

## Student Model Candidates

The student should be chosen for deployability first, then distilled for accuracy.

Most practical candidates:

- `MobileNetV3`
- `EfficientNet-Lite`
- `EdgeNeXt`
- `EfficientFormer`
- a small `ConvNeXt` variant, if the deployment graph is acceptable

Selection criteria:

- known ONNX export stability
- good compatibility with Hailo-supported operators
- low latency at `224-320` input
- enough capacity for regression, not just coarse classification

## Input Design Options

There are three realistic student input strategies.

### Option A: Single-frame stitched image

- keep the original left/right information in one combined image
- simplest deployment path
- easiest to export and compile

Tradeoff:

- the student must learn left/right interactions internally

### Option B: Two-frame or two-crop lightweight dual branch

- keep two small branches and fuse them late
- more faithful to the current teacher

Tradeoff:

- more deployment complexity
- may be less compiler-friendly

### Option C: Single-frame student plus video window

- infer one compact frame representation at a time
- rely on temporal fusion to restore information lost from removing explicit dual-stream complexity

Tradeoff:

- best system simplicity on device
- places more burden on the Pi fusion layer

Recommendation for v1:

- start with `Option C`
- use a single-frame student and recover quality through short-window temporal fusion

## Distillation Targets

The student should not learn only from hard labels.

Recommended targets from the best teacher or teacher ensemble:

- `5` regression outputs
- `5` interval-classification logits or probabilities
- optional projected intermediate feature vectors
- optional sample confidence or uncertainty estimate

Recommended loss mix:

- hard supervised regression loss
- hard supervised interval-classification loss
- teacher regression distillation loss
- teacher logit distillation loss
- optional feature distillation loss
- temporal consistency loss for video-window training

Illustrative form:

```text
L_total =
  0.4 * hard_regression +
  0.2 * hard_interval_cls +
  0.2 * teacher_regression +
  0.1 * teacher_logits +
  0.1 * temporal_consistency
```

Exact weights should be tuned empirically.

## Training Strategy

### Stage 1: Train the best teacher first

Do not start edge work before the server teacher is stable.

Inputs:

- best single model from the current server sweep
- optionally the best small ensemble for offline pseudo-target generation

Outputs:

- teacher checkpoints
- per-sample teacher predictions
- optionally per-sample intermediate embeddings

### Stage 2: Generate student training targets

For each training image or video-derived frame:

- save teacher regression outputs
- save teacher interval logits or probabilities
- optionally save projected feature embeddings

If video data is available, sample short windows and assign:

- per-frame teacher outputs
- window-level fused teacher target

### Stage 3: Train the single-frame student

Train the compact student first without temporal fusion.

Goals:

- stable exportable architecture
- acceptable single-frame quality
- good calibration for the downstream temporal window

### Stage 4: Train or calibrate the temporal fusion module

Run the student over short windows, then fuse recent predictions using one of:

- exponential moving average
- confidence-weighted average
- small `GRU`
- small `1D TCN`
- small `MLP` over the stacked recent outputs

Recommendation for v1:

- start with confidence-weighted EMA
- then test a very small `GRU` only if the simpler rule saturates

## Temporal Fusion Design on Raspberry Pi 5

The temporal module should stay on the Pi CPU, not on the Hailo.

Reason:

- it is small and cheap enough to run on CPU
- it needs state and window logic
- it is easier to iterate than re-compiling a device graph

Suggested inputs per frame:

- `5` regression values
- `5 x bins` interval probabilities
- optional confidence
- optional `64`- or `128`-dim student embedding

Suggested window size:

- start with `K = 8`
- test `K = 4, 8, 16`

Suggested frame policy:

- do not fuse all raw frames blindly
- keep only frames with enough visual novelty
- skip near-duplicates to avoid overweighting the same view

## Evaluation Plan

Three evaluation levels matter.

### Level 1: Offline accuracy

Compare:

- best teacher single model
- best teacher ensemble
- student single-frame
- student plus temporal fusion

Metrics:

- weighted `R2`
- per-target correlation
- `MAE`
- `RMSE`

### Level 2: Edge suitability

Measure:

- ONNX export success
- Hailo compile success
- device memory and throughput
- end-to-end latency on `Pi 5 + Hailo-8`

### Level 3: Stability on stream data

Measure:

- frame-to-frame variance
- robustness to blur and exposure shifts
- sensitivity to repeated frames
- degradation when viewpoint motion is limited

## Expected Accuracy Tradeoff

Reasonable expectation:

- best teacher ensemble remains strongest
- best single teacher remains the realistic upper bound for edge imitation
- single-frame student trails the single teacher
- student plus short-window temporal fusion can recover part of the gap

Inference:

- if the video stream provides useful viewpoint diversity, the temporal system can materially outperform the same student on single frames
- if the stream is almost static, gains will be much smaller

## Risks

- the best student architecture for ImageNet-style tasks may still be poor for biomass regression
- `Hailo` compile support may eliminate some otherwise attractive student models
- video-frame redundancy may make temporal fusion less useful than expected
- aggressive quantization can harm small-value targets such as `Dry_Clover_g`
- a student may learn teacher bias, not just teacher strength

## Recommended Project Sequence

1. Finish and lock the best CSIRO teacher workflow on the server.
2. Select `2-3` export-friendly student architectures.
3. Train single-frame students with hard labels plus teacher distillation.
4. Validate ONNX export and Hailo compile feasibility early.
5. Add Pi-side temporal fusion on top of the best student.
6. Compare:
   - single-frame student
   - student plus temporal fusion
   - single teacher
7. Only then decide whether the edge path is worth productionizing.

## Practical Recommendation

For this repository, the best first edge prototype is:

- teacher: best current server-side `ViT` single model or small ensemble
- student: `MobileNetV3` or `EdgeNeXt-S`
- input: single compact frame representation, not the full current dual-stream teacher graph
- runtime: `Hailo-8` for per-frame inference, `Pi 5` for temporal fusion
- temporal fusion: confidence-weighted EMA first, tiny `GRU` second

This is the most realistic path to a deployable system that still benefits from the current strong `ViT` teacher stack.

## References

- Hailo Model Zoo: https://github.com/hailo-ai/hailo_model_zoo
- Hailo Applications: https://github.com/hailo-ai/hailo-apps
- Hailo Community on device/compiler constraints: https://community.hailo.ai/t/vision-acceleration-on-rpi-hailo-8l/12402/2
- Hailo Community on transformer support: https://community.hailo.ai/t/support-for-vision-transformers/14715
- Online Model Distillation for Efficient Video Inference: https://www.ri.cmu.edu/publications/online-model-distillation-for-efficient-video-inference/
- Cross-modality online distillation for multi-view action recognition: https://doi.org/10.1016/j.neucom.2021.05.077
- Cross-Architecture Knowledge Distillation: https://openaccess.thecvf.com/content/ACCV2022/papers/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.pdf
- Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation: https://arxiv.org/abs/2107.01378
- MiniViT: Compressing Vision Transformers with Weight Multiplexing: https://doi.org/10.1109/CVPR52688.2022.01183
- EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications: https://doi.org/10.1007/978-3-031-25082-8_1
- Knowledge Distillation: A Survey: https://doi.org/10.1007/s11263-021-01453-z
- Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks: https://doi.org/10.1109/TPAMI.2021.3055564
