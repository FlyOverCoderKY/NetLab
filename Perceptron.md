## Perceptron Plan (4×4 Grid Teaching App)

This document defines the non-code plan and build tasks for an educational Perceptron demo. It teaches how inputs × weights + bias produce an activation, demonstrates the perceptron learning rule, and includes a mini-project: classifying T vs J shapes.

### 1) Goals & Scope
- Primary: Help users see how inputs, weights, and bias combine into a decision, and how the perceptron learning rule updates weights to reduce mistakes.
- Secondary: Show limits (linear separability) and encourage experimentation.
- Audience: High-school to early undergrad; no prior ML required.

### 2) Core Concepts to Teach
- Model: \( z = \sum_{i=1}^{16} w_i x_i + b \), output \(\hat{y} = \mathbb{1}[z > 0] \)
- Targets: \( y \in \{0, 1\} \) (default: T = 1, J = 0)
- Online learning rule:
  - \( \mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x} \)
  - \( b \leftarrow b + \eta (y - \hat{y}) \)
- Intuition: If wrong, nudge weights toward the true class; if right, no change.
- Limits: Works for linearly separable tasks only.

### 3) App Structure (Sections)
- Welcome / Overview
- Playground: 4×4 input grid, weight sliders (4×4 + bias), live output
- Learning Rule Walkthrough: step-by-step with visual deltas
- T vs J Challenge: dataset, manual or auto-train, accuracy
- Reflection & Limits

### 4) UI/UX Layout (single page with sections)
- Left: Inputs
  - 4×4 input grid (toggles 0/1)
  - Presets: T, J, Clear, Random
  - Target label selector (T = 1, J = 0)
- Center: Weights & Bias
  - 4×4 weight sliders (−20…+20, step 0.5) with numeric display
  - Bias slider (−20…+20)
  - Learning rate slider (\(\eta\), e.g., 0.01–1.0)
  - Buttons: Reset Weights, Small Random Init, Zero Init
- Right: Output & Feedback
  - Calculation panel: show \( z \), threshold comparison, final \(\hat{y}\)
  - Heatmap overlay toggle: visualize \( w_i x_i \) contributions
  - Update panel: when applying learning rule, show \(\Delta w\) and \(\Delta b\)
  - Console/log: last N updates (sample, pred, error, \(\lVert\Delta\rVert\))
- Bottom: Training Controls (dataset mode)
  - Datasets: T vs J (balanced), T-noisy, Mixed orientations (optional)
  - Train mode: manual step, single pass, auto-train (epochs)
  - Metrics: accuracy, mistakes per epoch

### 5) Data & Encoding
- Input encoding: flatten 4×4 row-major → \( \mathbf{x} \in \{0,1\}^{16} \)
- Targets: T = 1, J = 0 (configurable)
- Preset shapes:
  - T: top row = 1s; center column = 1s; others 0
  - J: right column = 1s; bottom row = 1s; stylized “J” within 4×4
- Dataset generation:
  - 20–40 examples; half T, half J
  - Optional noise: flip 1–2 pixels with small probability (T-noisy)
  - Optional jittered variants (still separable)

### 6) App State (React)
- `inputs: number[16]` (0/1)
- `weights: number[16]` (−20…20)
- `bias: number` (−20…20)
- `eta: number` (learning rate)
- `target: 0|1`
- `prediction: 0|1` (derived from z)
- `z: number` (derived)
- `dataset: Array<{ x:number[16], y:0|1, id:string }>`
- `training: { mode: 'manual'|'single-pass'|'auto', epoch:number, index:number, mistakes:number }`
- `history: Array<{ sampleId:string, yTrue:0|1, yPred:0|1, deltaW:number[16], deltaB:number }>`

### 7) Interactions & Flows
- Playground
  - Click grid to toggle inputs → recompute \( z \), \(\hat{y}\)
  - Adjust sliders → recompute live
  - Apply Learning Rule → compute error \(y-\hat{y}\), update weights/bias by \(\eta\cdot\text{error}\cdot\mathbf{x}\), animate changes, log update
- Dataset training
  - Load dataset, show thumbnails
  - Manual step / Single pass / Auto-train; report mistakes per epoch

### 8) Visualizations
- Contribution heatmap: color each cell by \( w_i x_i \) (sign and magnitude), tooltip with \(w_i\), \(x_i\), product
- Update animation: flash changed cells; arrows for increase/decrease
- Training chart (optional): mistakes vs epoch

### 9) Decision Boundary Explanation (intuitive)
- Explain: In 16-D, the perceptron finds hyperplane \( \sum w_i x_i + b = 0 \)
- Show a “score bar” for \( z \) with threshold at 0; watch movement during updates

### 10) Guided Lesson Content (collapsible)
- Lesson 1: What’s a Perceptron? Inputs, weights, bias, step activation; try toggling a pixel
- Lesson 2: The Learning Rule; cases: correct, missed positive, missed negative; hands-on update
- Lesson 3: T vs J Mini-Project; converge to 0 mistakes; discuss learned weights
- Lesson 4: Limits & Extensions; linear separability; MLP/logistic extensions

### 11) Edge Cases & Settings
- Threshold tie: if \( z=0 \), define output (document choice; default 1), optional toggle
- Learning rate: default 0.1; warn on too high/low
- Clamping: keep weights/bias within [−20, 20]
- Initialization: zero vs small random (±0.5)
- Noise mode: demonstrate convergence when still separable

### 12) Accessibility & Usability
- Keyboard-operable grid (arrow keys + space)
- High-contrast mode and visible focus states
- Live region for output changes
- Numeric inputs mirroring sliders
- Tooltips with plain-language math

### 13) Content Copy (concise)
- Definition: “A perceptron adds up ‘on’ pixels times their importance (weights), adds a bias, and says ‘yes’ if the sum is above 0.”
- Learning rule tl;dr: “If it’s wrong, nudge weights toward the right answer; if it’s right, don’t touch them.”
- Why bias? “Bias lets the model say ‘yes’ even if few pixels are on—or require more evidence to say ‘yes’.”

### 14) Assessment & Prompts
- Quick checks: weight direction on missed positives/negatives; bias behavior
- Mini-challenges: vary \(\eta\); add a noisy pixel; reach 100% accuracy on clean T vs J

### 15) Technical Skeleton (implementation guidance)
- State mgmt: local React state or lightweight store; no backend needed
- Components (names indicative):
  - `InputGrid16`
  - `WeightsGrid16`
  - `BiasControl`
  - `OutputPanel`
  - `UpdateButton` and `UpdatesLog`
  - `DatasetPanel`
  - `ChartMistakes` (optional)
- Pure functions:
  - `predict(x, w, b) -> { z, yHat }`
  - `update(w, b, x, y, yHat, eta) -> { wPrime, bPrime, delta }`
  - `trainEpoch(dataset, w, b, eta)`
- Data: JSON arrays for T/J samples; generator for noise/jitter

### 16) Stretch Ideas (if time allows)
- Confusion matrix & per-sample margins \( y(2\hat{y}-1)\cdot z \) (advanced)
- Custom shapes: user-drawn 4×4 glyphs added to dataset
- Export/import: save weights/bias
- Compare modes: Perceptron vs logistic regression

---

## Task Checklist

- [ ] Page shell: `PerceptronPage` routing ready (links from Home)
- [ ] Input grid (4×4): toggles, keyboard support, presets (T, J, Clear, Random)
- [ ] Target selector (T/J)
- [ ] Weights grid (16 sliders) with numeric displays
- [ ] Bias control (slider + numeric)
- [ ] Learning rate control (slider + numeric)
- [ ] Reset/Init buttons (zero, small random)
- [ ] Live output panel: z value, threshold, final \(\hat{y}\)
- [ ] Contribution heatmap overlay toggle
- [ ] Apply Learning Rule button: compute and apply updates
- [ ] Visualize \(\Delta w\), \(\Delta b\) changes (color/animation)
- [ ] Updates log (last N)
- [ ] Dataset generation (T/J clean, T-noisy, mixed orientations optional)
- [ ] Dataset panel (thumbnails, sample selection)
- [ ] Training modes: manual step, single pass, auto-train with pause
- [ ] Metrics: mistakes per epoch and accuracy
- [ ] Score bar for z with threshold at 0
- [ ] Lessons (collapsible content)
- [ ] Edge cases settings: z=0 tie behavior, clamping, init presets
- [ ] Accessibility pass (keyboard, screen reader live regions, focus)
- [ ] Optional chart: mistakes vs epoch
- [ ] Content proofreading




