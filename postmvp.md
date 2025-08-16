## Post‑MVP: Gaps, Issues, and Suggestions

Short list of items to address after the initial buildout. Each item links to concrete files and proposes an action.

### Top priorities

- **Weight painting/edit mode (Softmax first)**
  - Gap: No UI to paint weights; worker does not handle `set-weights`.
  - References: `App.UI/src/ui/components/AnatomyPanel.tsx`, `App.UI/src/worker/messages.ts` (has `set-weights`), `App.UI/src/worker/trainer.ts` (no handler).
  - Suggestion: Add brush UI in `AnatomyPanel` with a selected class; send `set-weights` with sparse edits. Implement handler in worker to update softmax Dense kernel column; add Undo/Reset and throttle updates.
  - Status: DONE (basic controls). Added selection/UI actions for zero/randomize class in `AnatomyPanel`; implemented `set-weights` in worker for Softmax class columns and visuals refresh.

- **Model switching message**
  - Gap: `InMsg` defines `switch-model`, but the worker ignores it; mode changes only take effect on `compile`.
  - References: `App.UI/src/worker/messages.ts`, `App.UI/src/worker/trainer.ts` (no `switch-model` case).
  - Suggestion: Handle `switch-model` by disposing the current model and creating the requested one. Optionally auto‑recompile with last hyperparameters.
  - Status: DONE. Worker now handles `switch-model` and initializes a new model; updates `currentCfg.modelType`.

- **Dataset parameters not used by training**
  - Gap: Worker `compile` uses hardcoded dataset settings and `seed: 1234`; doesn’t consume UI `dataset` params.
  - References: `App.UI/src/state/store.ts` (dataset in store), `App.UI/src/ui/components/TrainPanel.tsx` (passes `seed: 1234`), `App.UI/src/worker/trainer.ts` (hardcoded glyph params in `handleCompile`).
  - Suggestion: Extend `TrainConfig` to include dataset params; pass from `TrainPanel` using store values; remove hardcoded defaults in worker.
  - Status: DONE. Extended `TrainConfig.dataset`; `TrainPanel` now passes store dataset values; worker uses them for train/val iterators.

- **L2 weight decay not applied**
  - Gap: `weightDecay` exists but is unused in losses.
  - References: `App.UI/src/models/{softmax,mlp,cnn}.ts` (`trainStep`).
  - Suggestion: If `weightDecay>0`, add `0.5*wd*||W||^2` for trainable weights to the loss (simple sum over Dense/Conv kernels).
  - Status: DONE. Implemented basic L2 regularization in all three models and plumbed `weightDecay` via `setHyperparams` from worker.

- **MLP/CNN visuals missing activations**
  - Gap: Only softmax class tiles and CNN filters are shown; no feature maps or hidden activations.
  - References: `App.UI/src/models/mlp.ts` (`getVisuals()` → first layer weights only), `App.UI/src/models/cnn.ts` (`filters` only), `App.UI/src/ui/components/AnatomyPanel.tsx` (renders `weights` only).
  - Suggestion: For CNN, add first‑layer feature maps (e.g., 8× 12×12) for the latest sample; for MLP, add a small bar array of hidden activations. Render conditionally in `AnatomyPanel`.
  - Status: DONE (initial). `getVisuals` now optionally returns CNN feature maps and MLP hidden activation bars when an input sample is provided; `AnatomyPanel` renders `weights`, `filters`, and `activations` collectively. Worker currently calls `getVisuals()` without sample; passing a live sample is a future enhancement (see Visuals transfer performance).

- **Visuals transfer performance**
  - Gap: Visuals are posted as nested `number[][]`; not using transferables.
  - References: `App.UI/src/worker/trainer.ts` (posts visuals), `App.UI/src/ui/components/AnatomyPanel.tsx`.
  - Suggestion: Convert tiles to `Float32Array` (flat), transfer buffers across threads, and reconstruct on the main thread; continue throttling ≤ 4 Hz.
  - Status: DONE. Models emit `*Arr` fields; worker now posts transfer-optimized payloads using transfer lists; `AnatomyPanel` consumes `*Arr` first and falls back to nested grids.

### Medium priorities

- **Expose Dataset preview in UI**
  - Gap: `DatasetPanel.tsx` exists but is not reachable via tabs.
  - Suggestion: Add a “Dataset” tab or a collapsible section inside the Playground.
  - Status: DONE. Added `dataset` tab, wired route to `DatasetPanel`.

- **Playground auto‑cycle**
  - Gap: Only manual Next/Predict.
  - Suggestion: Add a toggle to auto‑advance samples every N seconds with a safe FPS budget; pause during training if needed.
  - Status: DONE. Added auto‑cycle toggle and interval control; triggers Next+Predict on a timer.

- **Math step overlays**
  - Gap: No gradient/∆W overlays during step‑through.
  - Suggestion: On `step` when paused, compute gradients for the selected class only (softmax first) via `tf.grads`, stream magnitude heatmaps, and clear when running resumes.
  - Status: DONE. Worker computes and streams a per-class gradient-magnitude heatmap (`overlaysArr`) for all model types using input gradients; toggle added in `TrainPanel`.

- **Param count reporting**
  - Gap: `compiled.params` is hardcoded for softmax.
  - References: `App.UI/src/worker/trainer.ts` (`payload: { params: 36 * 784 + 36 }`).
  - Suggestion: Compute from the instantiated model’s weights (`sum(model.layers[i].getWeights()[j].size)`).
  - Status: DONE. Worker now computes parameter count from model layers when available.

- **Docs/branding cleanup**
  - Gaps:
    - `App.UI/README.md` is template content pointing to `Sorter/Template`.
    - `App.UI/package.json` name is `template-ui`.
  - Suggestion: Update README with project quickstart and architecture; rename package to `netlab-ui` (or similar); ensure links and badges are correct.
  - Status: DONE. README updated; package renamed to `netlab-ui`.

- **CI/CD**
  - Gap: Plan mentions Azure Static Web Apps via GitHub Actions; the repo tree doesn’t show a workflow file here.
  - Suggestion: Add a workflow to run `npm ci`, `npm run build`, and `npm test`; ensure `.wasm` assets are served with the correct MIME type; deploy to target.
  - Status: DONE (checks), PARTIAL (deploy). Consolidated lint/test/build into existing Azure SWA workflow; deprecated standalone CI workflow. Deployment continues via SWA action; no changes needed.

### Nice‑to‑haves and polish

- **Recharts usage**
  - Gap: Listed in dependencies but not used.
  - Suggestion: Either remove the dependency or use Recharts for low‑frequency/static plots and keep Canvas for high‑frequency charts.
  - Status: DONE. Removed `recharts` from dependencies to simplify footprint.

- **Font determinism**
  - Gap: No bundled font assets or `fonts.ts`; generator relies on system fonts.
  - Suggestion: Bundle a redistributable font and use it by default; keep `waitForFontsReady` guard; document expected variability across devices otherwise.
  - Status: DONE (initial). Added `fonts.ts` with bundled font registration and updated generator to prefer the bundled font if available; preloads added in `index.html`. Next: add the actual font asset file in `/public/fonts` (missing in repo).

- **OffscreenCanvas**
  - Suggestion: Where supported, consider `OffscreenCanvas` for the mini chart and heatmaps to minimize main‑thread work during training.
  - Status: DONE (initial). `Heatmap` now uses `OffscreenCanvas` when available to compose source images before scaling. Data generator already supported OffscreenCanvas.

- **Mobile defaults**
  - Suggestion: Detect lack of WASM SIMD/threads and reduce default batch sizes.
  - Status: DONE. Added runtime capability detection and automatic batch size reduction (to 16) on constrained WASM devices.

### Verification checklist

- **Functional**
  - Weight brush edits update softmax predictions immediately; Undo/Reset works.
  - Switching modes without recompiling uses `switch-model` correctly or triggers auto‑compile.
  - Training uses UI dataset params and seed; reproducible across runs on the same device.
  - L2 decay changes training dynamics when enabled.

- **Performance**
  - Visuals/metrics updates ≤ 4 Hz; visuals use transferables.
  - 10‑minute training run shows stable memory (no leaks).

- **UX/Docs**
  - Dataset preview reachable; Playground auto‑cycle toggle present.
  - Math overlays appear in step mode and clear on run.
  - README and package metadata updated; CI builds, tests, and deploys.


