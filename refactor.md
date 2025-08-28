## NetLab Refactor Plan

This plan outlines a staged, non-destructive rewrite of the UI and OCR pipeline. We will not remove existing code during the refactor. Instead, we will build new, parallel components and routes, and migrate feature-by-feature with clear checkpoints.

### Goals
- Establish a consistent, site-wide layout: `AppHeader` → `TopNav` → page content → `Footer`.
- Replace tab-based intra-app switching with actual routes: `Home`, `Perceptron`, `OCR`.
- Keep current panels accessible as a “Legacy” area while we migrate.
- Unify the OCR training and inference pipelines so the model trains on what it will see at inference.

---

## Phase 1: Navigation and Layout (Foundation)

Outcome: Every page uses the same `AppHeader` and `Footer`, with a reusable `TopNav` placed directly beneath the header. Routing replaces global tab state for top-level navigation.

1) Introduce routing without breaking existing features
- Install router (PowerShell):
  - `npm i react-router-dom@6`
- Create `App.UI/src/app/router.tsx` with routes:
  - `/` → `HomePage`
  - `/perceptron` → `PerceptronPage`
  - `/ocr` → `OCRPage`
  - `/legacy` → hosts existing tabbed panels (wrap current `RoutesView`)
- Update `App.UI/src/App.tsx` to wrap the app in `BrowserRouter` and render `TopNav` immediately below `AppHeader`. Keep `AppToolbar` visible on `/legacy` only (or temporarily everywhere until we finish Phase 1).

2) Create `TopNav` (site-wide primary nav)
- New file: `App.UI/src/ui/components/TopNav.tsx`
- Behavior:
  - Renders links to `Home`, `Perceptron`, `OCR`.
  - Responsive, keyboard accessible, with visible focus states.
  - Highlights the active route using `NavLink`.
  - Lives directly under `AppHeader` in `App.tsx`.

3) Page wrappers
- New files:
  - `App.UI/src/ui/pages/HomePage.tsx`
  - `App.UI/src/ui/pages/PerceptronPage.tsx`
  - `App.UI/src/ui/pages/OCRPage.tsx`
- Each page renders only its content (no header/footer; those are supplied by `App.tsx`).

4) Keep the existing experience intact
- Preserve `AppToolbar` and `Tabs` for `/legacy` to avoid deleting any code.
- Add a small notice on `/legacy` explaining it’s the previous UI with tabs.

Acceptance criteria
- `TopNav` is visible below `AppHeader` on all routes, and `Footer` remains site-wide.
- Navigating via `TopNav` changes URL and content without full reload.
- `/legacy` route shows the current panels (Playground, Dataset, Train, OCR, Anatomy, Math) exactly as they work today.

---

## Phase 2: Home Page

Outcome: A friendly landing page describing neural networks and what users can try here.

Scope
- Introductory content: What is a neural network? Why visualization matters.
- Quick links to `Perceptron` and `OCR` pages.
- Optional: A small hero graphic or diagram.

Acceptance criteria
- Clear explanation plus links to try demos.
- Mobile-friendly layout.

---

## Phase 3: Perceptron Page (Educational Demo)

Outcome: A comprehensive perceptron teaching demo will be built in a later phase. See `Perceptron.md` for the full plan and the detailed task checklist.

Features (moved to `Perceptron.md`)

Implementation notes (see `Perceptron.md`)

Acceptance criteria (see `Perceptron.md`)

---

## Phase 4: OCR Page (Unified Train/Infer Pipeline)

Core idea
- Train on what you’ll see. Generate training samples by rendering text, segmenting with the same OCR used at inference, then preprocessing glyphs with one shared pipeline.

4.1) Create a single shared preprocess library
- New directory: `App.UI/src/lib/preprocess/`
  - `index.ts` (composes the pipeline, used by both training and inference)
  - `binarize.ts` (Otsu/Sauvola/fixed)
  - `geometry.ts` (deskew ±7°, center-of-mass centering)
  - `resize.ts` (keep-aspect resize, pad-to-square)
  - `normalize.ts` (polarity + scaling)
  - `artifacts.ts` (blur, dilate/erode, jpegArtifacts, paperNoise)
- One function to rule them all:
  - `preprocessChar(bitmap, opts): Float32Array` → returns 28×28 floats
- Use the same compose in both training and inference paths.

4.2) Segment via OCR to create training data
- New directory: `App.UI/src/lib/ocr/`
  - `segment.ts` (projection profiles or connected components; lines → char boxes)
  - `visualize.ts` (helpers for overlays, debug thumbnails)
- New data helpers: `App.UI/src/data/synthText.ts` (render phrases/lines with fonts)
- New: `App.UI/src/data/makeTrainSet.ts` → render lines → segment → `preprocessChar` per segment → return labeled examples.

4.3) Normalize consistently (killer details)
- Polarity: choose one and stick to it (white bg, black ink → 0 = black, 1 = white).
- Never stretch: scale longest side, then pad to 28×28.
- Center by moments, not just bbox.
- Light deskew.
- Either binarize or apply contrast normalization for grayscale.

4.4) Realistic augmentations
- Morphology: 1px dilate/erode for bold/thin.
- Blur: Gaussian σ∈[0.4, 1.0].
- Translation: ±2px; Rotation: ±5° after deskew.
- Noise: light speckle/paper texture.
- Compression: light JPEG artifacts.
- UI: expose sliders to see robustness vs. accuracy tradeoffs.

4.5) Instrumentation — Pipeline Inspector
- New components in `App.UI/src/ui/components/ocr/`:
  - `PipelineInspector.tsx` (left: raw crop; middle: step-by-step thumbnails; right: final 28×28 and activations)
  - `ActivationBars.tsx` (top-k predictions)
  - Metrics (SSIM/MSE) vs. a reference training char.

4.6) Curriculum & evaluation
- Curriculum: train on direct-rendered, then mix OCR’d crops, then all OCR-on.
- Two validation sets: VAL-A (synthetic), VAL-B (OCR’d). Show both curves.

4.7) Model and loss
- Keep Tiny CNN as primary; include MLP/Softmax as modes.
- Add label smoothing (ε≈0.05) and L2 weight decay (1e-4).

4.8) Fallbacks & sanity checks
- Template-matching baseline: cosine similarity vs. average prototypes.
- Distribution checks: per-pixel mean/std for train vs. infer; if drift > threshold, show UI warning.

4.9) Worker integration (no deletions)
- Update `App.UI/src/worker/trainer.ts` to call `makeTrainSet` for batches already passed through OCR+preprocess.
- Ensure inference path calls the same `preprocessChar`.

Acceptance criteria
- One shared pipeline used by both training and inference.
- VAL-A and VAL-B both reported; users can see domain gap reduce.
- Pipeline Inspector shows each step + final prediction.

---

## Migration Map (from current code to new structure)

Keep existing files; add new ones and gradually switch callers.

- `App.UI/src/App.tsx`: Add `BrowserRouter`, render `TopNav` under `AppHeader`. Limit `AppToolbar` to `/legacy` during migration.
- `App.UI/src/ui/routes.tsx`: Keep as-is but mount under `/legacy` route.
- `App.UI/src/ui/components/Tabs.tsx`, `AppToolbar.tsx`: Continue to operate in `/legacy`.
- `App.UI/src/ui/components/OCRPanel.tsx`: Split later into `OCRPage`, `PipelineInspector`, `TrainerPanel`, `DatasetPanel` (new OCR variants). Keep original panel available under `/legacy` until parity is confirmed.
- `App.UI/src/data/generator.ts`, `fonts.ts`, `seed.ts`: Reuse fonts and seeds; replace glyph-only generation with `synthText.ts` that renders lines/phrases.
- `App.UI/src/lib/ocr/` and `App.UI/src/lib/rendering/`: Add new OCR+preprocess modules; do not remove existing helpers.
- `App.UI/src/models/`: Continue using `cnn.ts`, `mlp.ts`, `softmax.ts`. Add label smoothing and weight decay flags.
- `App.UI/src/worker/`: Update `trainer.ts` to call the new dataset builder; keep old paths behind a flag.

---

## Step-by-step Checklist

- [x] Install routing: `npm i react-router-dom@6`
- [x] Add `TopNav` and place under `AppHeader`
- [x] Create routes: `/`, `/perceptron`, `/ocr`, `/legacy`
- [x] Implement `HomePage` content
- [ ] Build `PerceptronPage` demo (sliders, plot, presets)
- [ ] Create `OCRPage` shell (tabs: Train, Inspect, Inference)
- [ ] Implement preprocess library (`preprocessChar` end-to-end)
- [ ] Implement OCR segmentation and visualization helpers
- [ ] Implement `synthText.ts` and `makeTrainSet.ts`
- [ ] Wire worker training to `makeTrainSet` (feature flag)
- [ ] Add augmentations toggles and defaults
- [ ] Add Pipeline Inspector UI + SSIM/MSE metrics
- [ ] Add VAL-A/VAL-B tracking and charting
- [ ] Add template-matching baseline
- [ ] Add distribution drift checks and UI warnings
- [ ] Migrate OCR UI from `/legacy` to `/ocr` incrementally
- [ ] After parity, retire `/legacy` (post-migration task)
 
Progress notes
- Completed Phase 1 routing and navigation wiring:
  - Router installed and configured
  - `TopNav` added below `AppHeader`
  - Routes created: `/`, `/perceptron`, `/ocr`, `/legacy`
  - `Legacy` renders original tabbed UI with `AppToolbar` and `RoutesView`
  - Lint fixed in `PerceptronPage` select handler

---

## Dependencies and Suggested Libraries

- Routing: `react-router-dom@6`
- Metrics: SSIM/MSE (light-weight implementation or small utility; can implement SSIM directly to avoid heavy deps)
- Optional numeric helpers: small image ops can be custom; avoid heavy dependencies to keep the app light.

PowerShell commands
- Install router: `npm i react-router-dom@6`

---

## Risks and Mitigations

- Risk: Divergent pipelines between train/infer persist.
  - Mitigation: One exported `preprocessChar` used in both code paths; unit tests enforce identical outputs.
- Risk: Segmentation imperfections degrade training.
  - Mitigation: Keep curriculum that starts synthetic-only and mixes in OCR crops gradually.
- Risk: Performance on low-end devices.
  - Mitigation: Reuse WASM backend checks; keep batch sizes modest by default; expose controls.

---

## Acceptance Criteria (Project-level)

- Navigation
  - `TopNav` under `AppHeader`, visible on all pages.
  - `/`, `/perceptron`, `/ocr`, `/legacy` routes working.
- Perceptron
  - Interactive demo with decision boundary; no errors.
- OCR
  - Shared preprocess used by both train and infer.
  - Two validation curves (VAL-A synthetic, VAL-B OCR’d) displayed.
  - Pipeline Inspector present with per-step thumbnails and activations.
  - Basic accuracy improvement visible when training with OCR-on data.

---

## Appendix: Component and File Stubs (sketches)

These are high-level sketches to guide implementation. Names and props may evolve.

```tsx
// App.UI/src/ui/components/TopNav.tsx (sketch)
export const TopNav: React.FC = () => (
  <nav aria-label="Primary">
    {/* NavLink to /, /perceptron, /ocr */}
  </nav>
);
```

```tsx
// App.UI/src/app/router.tsx (sketch)
export const AppRouter = () => (
  <Routes>
    <Route path="/" element={<HomePage />} />
    <Route path="/perceptron" element={<PerceptronPage />} />
    <Route path="/ocr" element={<OCRPage />} />
    <Route path="/legacy" element={<LegacyShell />}> {/* wraps current RoutesView */}</Route>
  </Routes>
);
```

```tsx
// App.UI/src/lib/preprocess/index.ts (sketch)
export function preprocessChar(bitmap: ImageData | Float32Array, opts: PreprocessOptions): Float32Array {
  // deskew → crop → resizeKeepAspect → padToSquare → centerByMoments → normalize → artifacts (optional)
  return output28x28;
}
```

```tsx
// App.UI/src/data/makeTrainSet.ts (sketch)
export async function makeTrainSet(labels: string[]): Promise<{ x: Float32Array[]; y: number[] }>{
  // render lines → ocr.segment → preprocessChar per segment → return labeled samples
}
```

```tsx
// App.UI/src/ui/components/ocr/PipelineInspector.tsx (sketch)
export const PipelineInspector: React.FC = () => {
  // left: raw crop, middle: step thumbnails, right: final 28×28 + activations
  return null;
};
```


