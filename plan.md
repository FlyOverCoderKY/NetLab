# Teaching Neural Nets in the Browser — Build Plan (React + TF.js)

**Goal:** a single static React app that teaches three models—**Softmax**, **MLP**, **Tiny CNN**—with shared UI, in-browser training/visualization, and a math-first explanation.  
**Constraints:** small, deterministic, inspectable; runs fully offline (no server).  
**Tooling:** React + TypeScript + Vite, TensorFlow.js (WebGL + WASM fallback), Web Workers, Zustand state, Recharts (or lightweight D3), Canvas for images/heatmaps.

> 🧩 **How to use this plan:** Each phase is intentionally bite-sized (buildable within a typical Cursor agent session). Every task has **Deliverables** + **Definition of Done (DoD)**. Copy the task block you’re working on into a Cursor run.

---

## Tech Decisions (fixed upfront)

- **Runtime:** React 18, TypeScript, Vite.
- **ML:** `@tensorflow/tfjs` + `@tensorflow/tfjs-backend-webgl` + `@tensorflow/tfjs-backend-wasm`.
- **State:** Zustand.
- **Worker:** Dedicated Web Worker for training; main thread for UI. Worker uses the **WASM** backend only; WebGL remains on the main thread.
- **Charts:** Recharts (for low-frequency/static charts) or D3. For high-frequency training metrics, use a lightweight Canvas-based chart and disable animations during training.
- **Styling:** Tailwind (optional) or CSS Modules.
- **Testing:** Vitest + React Testing Library.
- **Hosting:** Azure Static Web Apps (free tier) or GitHub Pages.
- **Data:** Synthetic glyphs rendered on a hidden `<canvas>`; default 28×28 grayscale; classes = 36 (A–Z, 0–9).
- **Fonts:** Bundle redistributable fonts and wait for `document.fonts.ready` before drawing. For cross-device determinism, prefer bundled bitmap/vector glyphs rather than `fillText`.
- **Package manager:** npm (aligns with existing `package-lock.json`).

---

## Repo/Folder Sketch

Note: In this repo, place the following under `App.UI/src`.

```
/src
  /models
    softmax.ts
    mlp.ts
    cnn.ts
    index.ts          // dynamic loaders
    types.ts          // TeachModel interface
  /worker
    trainer.ts        // worker entry
    messages.ts       // message schemas
  /data
    generator.ts      // glyph synth + augmentation
    fonts.ts          // font faces & fallbacks
    seed.ts           // deterministic RNG
  /ui
    App.tsx
    routes.tsx
    components/
      ModeSwitcher.tsx
      DatasetPanel.tsx
      TrainPanel.tsx
      AnatomyPanel.tsx
      MathPanel.tsx
      Heatmap.tsx
      ConfusionMatrix.tsx
      MetricChart.tsx
      Toggle.tsx
      Slider.tsx
  /state
    store.ts
  /utils
    array.ts
    tf.ts             // backend selection, helpers
    viz.ts            // tensor->image, normalize
/assets
index.html
vite.config.ts
```

---

## Core Interfaces (copy into `src/models/types.ts`)

```ts
export type Grid = number[][];

export type Visuals = {
  weights?: { name: string; grid: Grid }[];     // e.g., class weight heatmaps
  filters?: { name: string; grid: Grid }[];     // e.g., conv kernels
  activations?: { layer: string; grid: Grid }[];// small featuremaps
};

export interface TeachModel {
  name: "softmax" | "mlp" | "cnn";
  inputShape: [number, number, number]; // [28,28,1]
  init(params?: Record<string, unknown>): Promise<void>;
  predict(x: tf.Tensor4D): Promise<{ logits: tf.Tensor2D; probs: tf.Tensor2D }>;
  trainStep(batch: { x: tf.Tensor4D; y: tf.Tensor1D }): Promise<{ loss: number; visuals?: Visuals }>;
  evaluate(ds: Iterable<{ x: tf.Tensor4D; y: tf.Tensor1D }>): Promise<{ acc: number }>;
  getVisuals(): Promise<Visuals>;
  serialize(): Promise<Record<string, unknown>>;
  load(state: Record<string, unknown>): Promise<void>;
  dispose(): void;
}
```

---

## Worker Message Schema (copy into `src/worker/messages.ts`)

```ts
export type TrainConfig = {
  modelType: "softmax" | "mlp" | "cnn";
  seed: number;
  batchSize: number;
  learningRate: number;
  optimizer: "sgd" | "adam";
  weightDecay?: number; // L2 coefficient added to the loss (not decoupled)
  steps: number;        // total steps to run
  snapshotEvery: number;// send visuals/metrics every N steps
};

export type InMsg =
  | { type: "init"; payload: { backend: "webgl" | "wasm" } }
  | { type: "compile"; payload: TrainConfig }
  | { type: "step"; payload?: {} }             // single step (for step-through)
  | { type: "run"; payload: { steps: number } } // run N steps
  | { type: "pause" }
  | { type: "set-weights"; payload: any }      // manual edits (softmax)
  | { type: "switch-model"; payload: { modelType: TrainConfig["modelType"] } }
  | { type: "dispose" };

export type OutMsg =
  | { type: "ready" }
  | { type: "compiled"; payload: { params: number } }
  | { type: "metrics"; payload: { step: number; loss: number; acc?: number } }
  | { type: "visuals"; payload: import("../models/types").Visuals }
  | { type: "weights"; payload: any } // serialized weights
  | { type: "error"; payload: { message: string } }
  | { type: "done"; payload?: {} };
```

---

## Math (for the “Math” tab)

- **Softmax Regression**  
  \( z = Wx + b \), \( p = \mathrm{softmax}(z) \)  
  Loss (CE): \( \mathcal{L} = -\log p_y \)  
  Gradients: \( \frac{\partial \mathcal{L}}{\partial W} = (p - y) x^\top \), \( \frac{\partial \mathcal{L}}{\partial b} = p - y \)

- **MLP (1 hidden, ReLU)**  
  \( h = \max(0, W_1 x + b_1) \), \( z = W_2 h + b_2 \), \( p=\mathrm{softmax}(z) \)  
  Backprop: standard dense layer derivatives; illustrate chain rule visually.

- **Tiny CNN**  
  \( h_{k} = \mathrm{ReLU}(x * K_k + b_k) \) → MaxPool(2) → Flatten → Dense → Softmax  
  Discuss parameter sharing & locality.

---

## Phases & Tasks

> ⚠️ Keep phases atomic. Each ends with a running build and visible UI change.

### Phase 0 — Bootstrap

**Tasks**
1. `npm create vite@latest` (React + TS); add `tfjs`, `tfjs-backend-webgl`, `tfjs-backend-wasm`, `zustand`, `recharts`, `vitest`.
2. Set up Tailwind (optional) or minimal CSS.
3. Add `src/utils/tf.ts` to select backend (`tf.setBackend("webgl")` with WASM fallback).
4. Configure worker bundling: create workers via `new Worker(new URL("./worker/trainer.ts", import.meta.url), { type: "module" })` and verify WASM assets resolve at runtime.

**Deliverables**
- App boots, shows chosen TF backend in footer. ✅ Completed.

**DoD**
- `npm run dev` renders “Backend: webgl|wasm”; build succeeds.

---

### Phase 1 — Deterministic Glyph Generator

**Tasks**
1. Choose determinism strategy:
   - Prefer bundled bitmap/vector glyphs for cross-device determinism; or
   - Accept same-device determinism when using `CanvasRenderingContext2D#fillText`.
2. `data/seed.ts`: xorshift/alea RNG with settable seed.
3. `data/generator.ts`: canvas synth for 28×28 grayscale glyphs (wait for `document.fonts.ready` when using system fonts):
   - Inputs: char, font family, size, thickness, jitter (±px), rotation (±deg), noise/blur toggles.
   - Outputs: `Float32Array` of length 784, values ∈ [0,1] (1=white, 0=black) or inverted if preferred.
4. Preview grid in `DatasetPanel.tsx` with sliders.

**Deliverables**
- Can render a 6×6 sample grid of all classes with tunable noise. ✅ Completed (basic version: `DatasetPanel` in `App.UI/src/ui/components/DatasetPanel.tsx`).

**DoD**
- Deterministic per seed on the same device (or cross-device if using bundled glyphs); no layout shift; 60fps when adjusting sliders. Current: deterministic per seed on same device; performance acceptable for initial version.

---

### Phase 2 — App State + Routing

**Tasks**
1. Zustand store: mode (`softmax|mlp|cnn`), dataset params, training params (LR, batch, optimizer).
2. Top-level layout with tabs: **Playground**, **Train**, **Anatomy**, **Math**.
3. Mode switcher (dropdown) that updates store.

**Deliverables**
- Switching tabs/modes updates UI without reload. ✅ Completed.

**DoD**
- State persists during navigation; TS types enforced. ✅ Types enforced via `useAppStore`; state retained during tab switches.

Current:
- Added `src/state/store.ts` with `mode`, `tab`, `dataset`, `training` and actions.
- Added `ui/components/Tabs.tsx` and `ui/routes.tsx`; wired into `App.tsx`.
- Added `ui/components/ModeSwitcher.tsx` in header.
- Refactored `DatasetPanel` to read/write global dataset params.

---

### Phase 3 — Worker Scaffolding

**Tasks**
1. Create `worker/trainer.ts` with message loop (see schema).
2. In main thread, a `TrainerClient` wrapper to `postMessage`/`onmessage`.
3. Wire **Init** and **Dispose** messages. Initialize TF.js in the worker with the **WASM** backend explicitly; do not attempt WebGL in the worker.
4. Instantiate the worker via `new Worker(new URL("./worker/trainer.ts", import.meta.url), { type: "module" })` to ensure Vite bundles it correctly.
5. Preload TF.js WASM in the worker and verify asset URL resolution.
6. Use transferable objects when posting arrays (`postMessage(arr, [arr.buffer])`).

**Deliverables**
- Worker replies `ready`; switching backend emits status. ✅ Basic init/ready implemented.
- Compile/Run/Step/Pause message handling with simulated metrics and throttling. ✅ Implemented.

**DoD**
- No console errors; worker survives hot reload. ✅ Initial smoke via `TrainerStatus`.

Current:
- Added `src/worker/messages.ts` (schema) and `src/worker/trainer.ts` with compile/run/step/pause and simulated metrics; signals `ready` immediately for scaffolding.
- Added `src/worker/client.ts` singleton `TrainerClient` to abstract messaging.
- `TrainerStatus` now uses the client; `TrainPanel` wired to compile/run/pause/step and consumes worker `metrics`/`done` events.
- Next: verify WASM asset resolution when TF.js is enabled in worker; convert to transferable typed arrays for future visuals.

---

### Phase 4 — TeachModel Interface + Softmax Skeleton

**Tasks**
1. Add `models/types.ts` (above).
2. Implement `models/softmax.ts` using either:
   - **Layers API:** `tf.sequential([Flatten, Dense(36, useBias:true)])`, logits out.
   - Or manual vars `W(784,36)`, `b(36)` for easier weight painting.
3. Implement `predict`, `trainStep` (single mini-batch, CE loss), `getVisuals` (36 heatmaps from columns of `W`).

**Deliverables**
- Softmax model compiles, returns logits for a batch.

**DoD**
- One `trainStep` reduces loss on a synthetic tiny batch (overfit sanity check).

Current:
- Added `src/models/types.ts` with `TeachModel` and `Visuals` definitions. ✅
- Added `src/models/softmax.ts` with a minimal `Flatten → Dense(36)` model and stubbed training/eval. Compilation works; real batch wiring pending. 🚧

---

### Phase 5 — Data Loader & Batch Sampler

**Tasks**
1. Add a `makeDataset(seed, params)` generator that yields labeled mini-batches `{ x: Tensor4D, y: Tensor1D }`.
2. Class list: `['A'..'Z','0'..'9']`. Map to indices 0..35.
3. Batch builder: random classes with augmentation; ensure balanced sampling over time.

**Deliverables**
- Worker can request batches; main thread isn’t blocked (prepare in worker if possible).

**DoD**
- 1,000 batch generations < 1s on mid hardware; memory stable.

---

### Phase 6 — Train Loop (Run & Step)

**Tasks**
1. Implement `compile` (optimizer, LR, decay) & `run/step` messages.
2. On every `snapshotEvery`, post `{ step, loss, acc? }` and visuals. Throttle updates to ≤ 4 Hz. Send visuals only for the currently selected class/layer when applicable. Convert tensors to typed arrays and pass as transferables.
3. Add **Start/Pause/Step** buttons in `TrainPanel`.

**Deliverables**
- Loss curve visibly trends downward.

**DoD**
- Pause works; step-through performs exactly one update; no UI jank.

Current:
- Added `TrainPanel` with Start/Pause/Step and a lightweight Canvas chart that simulates decreasing loss for UI scaffolding. Real loop will be driven by the worker in Phase 6 implementation. 🚧

---

### Phase 7 — Playground + Prediction UX

**Tasks**
1. **Playground** tab: show random sample, **Predict** button, top-k probabilities bar chart.
2. Add “Cycle Samples” (auto-advance every N sec).
3. Show current model’s confidence.

**Deliverables**
- Users can see model guessing the character live.

**DoD**
- Predict does not retrigger training; no race conditions.

---

### Phase 8 — Anatomy: Softmax Visuals & Weight Painting

**Tasks**
1. `Heatmap` component: draw 28×28 with color scale; hover to show weight value.
2. Grid of 36 heatmaps (one per class).
3. “Edit mode”: brush adds/subtracts weight at pixels; send `set-weights` to worker. Limit gradient/overlay computations and updates to the selected class only; throttle UI updates.

**Deliverables**
- Editing weights changes predictions immediately on the playground sample.

**DoD**
- Undo/Reset works; weight edits are confined to selected class; no crashes.

---

### Phase 9 — Metrics: Accuracy & Confusion Matrix

**Tasks**
1. `evaluate()` over a fixed, seeded micro validation set on a schedule (e.g., every M snapshots), throttled.
2. `ConfusionMatrix` component: 36×36 with tooltips, normalized rows.
3. Display top confusions (“O↔0”, “I↔1↔L”) as a list.
4. Ensure evaluation does not block training; reuse the worker on a time budget or evaluate on a small subset per tick.

**Deliverables**
- Accuracy trends up; confusions are readable.

**DoD**
- Matrix re-renders smoothly; evaluation runs on worker thread.

---

### Phase 10 — MLP Adapter

**Tasks**
1. `models/mlp.ts`: `Flatten → Dense(64, ReLU) → Dense(36)`.
2. Visuals:
   - First-layer weight tiles: 64 heatmaps (paginate 16 per page).
   - Activation bars for hidden layer on current sample.
3. Ensure param count stays ~50–110k.

**Deliverables**
- Mode switcher can swap to MLP without page reload; training works.

**DoD**
- Accuracy > Softmax under mild augmentations with same train budget.

---

### Phase 11 — Tiny CNN Adapter

**Tasks**
1. `models/cnn.ts`: `Conv2D(8, 5×5, relu) → MaxPool(2) → Flatten → Dense(36)`.
2. Visuals:
   - Filter tiles (8× 5×5).
   - First-layer feature maps (8 small 12×12 grids) for current sample.
3. Optional: simple Class Activation Map (CAM) via gradients or last conv features.

**Deliverables**
- CNN trains and surpasses MLP on distorted samples.

**DoD**
- Filters become edge/spot-like with training; FPS acceptable.

---

### Phase 12 — Math Tab (Dynamic)

**Tasks**
1. Render equations relevant to mode (see **Math** section).
2. Inline “What changed this step?” overlay during step-through:
   - Highlight \( \partial L/\partial W \) magnitude on heatmaps (compute for the selected class/layer only).
   - Show LR effect \( \Delta W = -\eta \nabla W \). Throttle overlays to keep UI responsive.

**Deliverables**
- Users can follow forward → loss → backward → update visually.

**DoD**
- Step-through overlays disappear when training resumes.

---

### Phase 13 — Settings, Persistence, Checkpoints

**Tasks**
1. Save/load weights JSON (per model) via `serialize()/load()`.
2. Include 1–2 **pretrained checkpoints** (static JSON) for instant demo.
3. Settings: choose backend, toggle deterministic mode, set RNG seed.

**Deliverables**
- Load pretrained → jump to >90% val acc (with moderate augmentation).

**DoD**
- JSON round-trip works across sessions; versioned schema.

---

### Phase 14 — Performance & Stability Pass

**Tasks**
1. Throttle visuals to ≤ 4Hz; decimate tensors to `Float32Array` before crossing threads and use transferables.
2. Avoid large postMessage payloads (e.g., send only a subset of heatmaps per snapshot).
3. Guard LR, batch size to safe ranges; NaN detection & graceful reset.
4. Establish explicit `tf.tidy` boundaries in train/predict paths and dispose all intermediate tensors. Add a small leak test that runs a tight loop and asserts stable memory.
5. Prefer Canvas-based charts for frequently updating metrics; keep Recharts for low-frequency visuals. Consider `OffscreenCanvas` where supported (optional).
6. Mobile/iOS: detect reduced WASM capabilities (no SIMD/threads) and lower default batch sizes accordingly.

**Deliverables**
- Smooth UI while training continuously.

**DoD**
- No memory leaks in a 10-minute run (Chrome DevTools heap stable).

---

### Phase 15 — Accessibility & UX Polish

**Tasks**
1. Keyboard controls for Start/Pause/Step.
2. Alt text for heatmaps; high-contrast mode switch.
3. Tooltips with plain-language explanations; “Why this matters” cards.

**Deliverables**
- Basic a11y check passes (tab order, ARIA on charts).

**DoD**
- Lighthouse a11y ≥ 90.

---

### Phase 16 — Tests & CI

**Tasks**
1. Unit tests: data generator determinism; softmax gradient sign sanity (loss decreases).
2. Component tests: Heatmap renders shape; Mode switch retains settings.
3. GitHub Actions: `npm ci`, `npm run build`, `npm test`. Pin TF.js backend to **WASM** in tests (`setBackend('wasm')`) for stability.

**Deliverables**
- Green CI badge; test coverage on critical paths.

**DoD**
- Failing gradient sanity test blocks PRs.

Current:
- CI configured to run lint and build, and deploy to Azure Static Web Apps on pushes/PRs. Tests to be added.

---

### Phase 17 — Docs & Deploy

**Tasks**
1. README: quickstart, architecture diagram, pedagogy notes.
2. Azure Static Web Apps deploy (or GH Pages).  
   - Ensure `wasm` assets are copied; set correct MIME for `.wasm`.
   - Verify worker context can resolve TF.js WASM URLs after build; add a CI check that fetches the built `.wasm` from the expected path.
3. Version the demo checkpoints and dataset seeds.

**Deliverables**
- Public URL; first-time load < 2MB gzipped (code-split models). ✅ Deployed at netlab.flyovercoder.com.

**DoD**
- Cold load ≤ 3s on mid hardware; offline via PWA cache (optional).
Current:
- Deployed via GitHub Actions to Azure Static Web Apps with prebuilt `dist` artifacts.

