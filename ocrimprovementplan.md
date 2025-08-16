## OCR Improvement Plan — Checklist and Roadmap

Goal: Raise recognition accuracy on paragraph images using the existing in‑browser stack, while keeping the app educational and approachable.

### Success metrics (track these for each iteration)
- [ ] Character accuracy on the Quick Fox sample (≥ 95%)
- [ ] Word accuracy on Quick Fox (≥ 85%)
- [ ] Inference latency for a 1–2 line image (≤ 200 ms on desktop)
- [ ] Stability across 2–3 uploaded font styles

## Training data improvements

- [ ] Render full lines/paragraphs, then segment using the SAME OCR pipeline used at inference; train on those crops (removes domain gap)
  - Files: `App.UI/src/data/generator.ts`, new `src/data/linegen.ts`, training script path (worker compile usage)
  - DoD: A preview toggle shows line rendering and segmentation boxes; generated training batches come from these segmented crops

- [ ] Margin/scale jitter per glyph (already added: `contentScale`, `contentJitter`) — broaden ranges
  - [ ] Use contentScale 0.70–1.00 with jitter ±0.15
  - [ ] Random X/Y offsets ±2 px; ensure consistent padding distribution

- [ ] Stroke/shape augmentation
  - [ ] Morphology ±1 px (dilate/erode) at random
  - [ ] Mild blur/contrast jitter (±10%)

- [ ] Font diversity
  - [ ] Add 2–3 additional bundled fonts (clear, slab, sans)
  - [ ] Random kerning/baseline offset (±1–2 px)

- [ ] Hard‑negative mining
  - [ ] Collect OCR misclassifications (I/1/L, O/0, 5/S, 2/Z, 8/B, 7/T)
  - [ ] Oversample these crops for a portion of steps

## Mode / model improvements

- [ ] Prefer the CNN mode for OCR; fine‑tune with OCR preset
  - [ ] Small LR 1e‑3 → 5e‑4; 10–20k steps; save a checkpoint
  - [ ] Track accuracy on a fixed validation line set

- [ ] Calibrate outputs
  - [ ] Temperature scaling or simple probability calibration on a validation set

- [ ] Decoding with a tiny language prior (post‑prediction)
  - [ ] Add bigram costs for A–Z, 0–9; apply in greedy or beam (k=3) decoding
  - [ ] Reduce common flips (I/1/L, O/0) without heavy dependencies

## OCR pipeline improvements

- [ ] Adaptive thresholding
  - [ ] Replace global threshold with Sauvola/local mean (block size ~15–31)
  - [ ] Optional 1‑px close/open morphology to remove speckle

- [ ] Deskew per line
  - [ ] Estimate dominant angle via projection/Hough and rotate once per image

- [ ] Segmentation upgrades
  - [ ] Merge tiny fragments (area/height thresholds)
  - [ ] Split overly wide components with vertical projection minima
  - [ ] Keep line grouping by y; sort left‑to‑right

- [ ] Centering to training distribution
  - [ ] Compute center of mass; translate crop so the ink is centered in 28×28
  - [ ] Fit content into 18–22 px box with slight jitter

- [ ] Batch predictions
  - [ ] Run glyphs in batches (e.g., 32) through the worker for speed

- [ ] Ambiguity heuristics (interim)
  - [ ] O vs 0 using roundness and hole; I/1/L via aspect; apply only when probabilities are close

## UI/UX and site organization

### Requested changes
1) Move “OCR training preset” to the Dataset page
- [ ] Add an “OCR preset” toggle in the Dataset panel and preview how margins/jitter look live
- [ ] Wire Train to read the preset from Dataset state (remove the Train toggle)

2) Make the site more educational and guided
- [ ] Expand inline guidance copy across tabs (what, why, how)
- [ ] Tooltips/glossary for terms (loss, batch, optimizer, conv filter, etc.)

3) Reorganize tabs (proposed flow)
- [ ] About Neural Networks tab (new) — “What is a neural network?”
  - Content outline:
    - Visual of a simple perceptron (input → weights → activation) [Illustration: diagram of nodes/edges]
    - Layers stack and feature extraction (Conv/ReLU/Pool) [Illustration: feature maps]
    - Training loop steps (forward → loss → backward → update) [Illustration: flow arrows]
    - Softmax vs MLP vs CNN — strengths/tradeoffs [Illustration: 3 mini diagrams]
  - Images: generate vector illustrations (SVG or canvas) and embed as assets

- [ ] Training Data tab (rename Dataset)
  - [ ] Clear explanation of synthetic glyphs, seeds, fonts, margins, and why augmentation matters (domain gap talk)
  - [ ] OCR preset toggle with live preview

- [ ] Train tab (combine current Train + Anatomy)
  - [ ] Show training controls AND live anatomy/weights side‑by‑side
  - [ ] Inline “What changed this step?” callouts; overlays toggle integrated
  - [ ] Math (mode‑specific) shown contextually in a sidebar or collapsible section

- [ ] Playground tab (combine current Playground + OCR)
  - [ ] Left: single‑glyph playground (Top‑5, heatmaps later)
  - [ ] Right: OCR demo (examples + upload); recognized text + boxes
  - [ ] Guidance copy: per‑character → real‑world paragraph scans

- [ ] Mode selector & Math location
  - [ ] Move Mode selector into Train header (and optionally into About for demos)
  - [ ] Display Math content inline on Train; remove standalone Math tab or keep it as a deep‑dive

4) Longer‑term training expansion
- [ ] Add lowercase a–z
- [ ] Add more fonts and kerning variations
- [ ] Extend to punctuation (.,:;!?), optional

## Phasing (suggested sprints)

- Sprint 1 (pipeline + preset)
  - [x] Implement adaptive threshold + deskew + better segmentation + centering
  - [x] Batch predictions in OCR panel
  - [x] Move OCR preset to Dataset; Train reads it; preview reflects margins

- Sprint 2 (training)
  - [ ] Render lines → segment → train on those crops with CNN
  - [ ] Add stroke/blur/contrast augmentations; run 10–20k steps; save checkpoint
  - [ ] Hard‑negative mining loop from OCR demo errors

- Sprint 3 (education + layout)
  - [ ] Add About tab with illustrations
  - [ ] Combine Train + Anatomy; combine Playground + OCR
  - [ ] Inline guidance + glossary

## Notes / File map
- Generator/preset: `src/data/generator.ts`, `src/ui/components/DatasetPanel.tsx`
- Train wiring: `src/ui/components/TrainPanel.tsx`, worker messages/config
- OCR pipeline: `src/ui/components/OCRPanel.tsx` (thresholding/segmentation/centering & batching)
- CNN training: `src/models/cnn.ts`, worker train loop


