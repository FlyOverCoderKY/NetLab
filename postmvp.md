## Post‑MVP: Gaps, Issues, and Suggestions

Short list of items to address after the initial buildout. Each item links to concrete files and proposes an action.

### Top priorities

- **Weight painting/edit mode (Softmax first)**
  - Gap: No UI to paint weights; worker does not handle `set-weights`.
  - References: `App.UI/src/ui/components/AnatomyPanel.tsx`, `App.UI/src/worker/messages.ts` (has `set-weights`), `App.UI/src/worker/trainer.ts` (no handler).
  - Suggestion: Add brush UI in `AnatomyPanel` with a selected class; send `set-weights` with sparse edits. Implement handler in worker to update softmax Dense kernel column; add Undo/Reset and throttle updates.

- **Model switching message**
  - Gap: `InMsg` defines `switch-model`, but the worker ignores it; mode changes only take effect on `compile`.
  - References: `App.UI/src/worker/messages.ts`, `App.UI/src/worker/trainer.ts` (no `switch-model` case).
  - Suggestion: Handle `switch-model` by disposing the current model and creating the requested one. Optionally auto‑recompile with last hyperparameters.

- **Dataset parameters not used by training**
  - Gap: Worker `compile` uses hardcoded dataset settings and `seed: 1234`; doesn’t consume UI `dataset` params.
  - References: `App.UI/src/state/store.ts` (dataset in store), `App.UI/src/ui/components/TrainPanel.tsx` (passes `seed: 1234`), `App.UI/src/worker/trainer.ts` (hardcoded glyph params in `handleCompile`).
  - Suggestion: Extend `TrainConfig` to include dataset params; pass from `TrainPanel` using store values; remove hardcoded defaults in worker.

- **L2 weight decay not applied**
  - Gap: `weightDecay` exists but is unused in losses.
  - References: `App.UI/src/models/{softmax,mlp,cnn}.ts` (`trainStep`).
  - Suggestion: If `weightDecay>0`, add `0.5*wd*||W||^2` for trainable weights to the loss (simple sum over Dense/Conv kernels).

- **MLP/CNN visuals missing activations**
  - Gap: Only softmax class tiles and CNN filters are shown; no feature maps or hidden activations.
  - References: `App.UI/src/models/mlp.ts` (`getVisuals()` → first layer weights only), `App.UI/src/models/cnn.ts` (`filters` only), `App.UI/src/ui/components/AnatomyPanel.tsx` (renders `weights` only).
  - Suggestion: For CNN, add first‑layer feature maps (e.g., 8× 12×12) for the latest sample; for MLP, add a small bar array of hidden activations. Render conditionally in `AnatomyPanel`.

- **Visuals transfer performance**
  - Gap: Visuals are posted as nested `number[][]`; not using transferables.
  - References: `App.UI/src/worker/trainer.ts` (posts visuals), `App.UI/src/ui/components/AnatomyPanel.tsx`.
  - Suggestion: Convert tiles to `Float32Array` (flat), transfer buffers across threads, and reconstruct on the main thread; continue throttling ≤ 4 Hz.

### Medium priorities

- **Expose Dataset preview in UI**
  - Gap: `DatasetPanel.tsx` exists but is not reachable via tabs.
  - Suggestion: Add a “Dataset” tab or a collapsible section inside the Playground.

- **Playground auto‑cycle**
  - Gap: Only manual Next/Predict.
  - Suggestion: Add a toggle to auto‑advance samples every N seconds with a safe FPS budget; pause during training if needed.

- **Math step overlays**
  - Gap: No gradient/∆W overlays during step‑through.
  - Suggestion: On `step` when paused, compute gradients for the selected class only (softmax first) via `tf.grads`, stream magnitude heatmaps, and clear when running resumes.

- **Param count reporting**
  - Gap: `compiled.params` is hardcoded for softmax.
  - References: `App.UI/src/worker/trainer.ts` (`payload: { params: 36 * 784 + 36 }`).
  - Suggestion: Compute from the instantiated model’s weights (`sum(model.layers[i].getWeights()[j].size)`).

- **Docs/branding cleanup**
  - Gaps:
    - `App.UI/README.md` is template content pointing to `Sorter/Template`.
    - `App.UI/package.json` name is `template-ui`.
  - Suggestion: Update README with project quickstart and architecture; rename package to `netlab-ui` (or similar); ensure links and badges are correct.

- **CI/CD**
  - Gap: Plan mentions Azure Static Web Apps via GitHub Actions; the repo tree doesn’t show a workflow file here.
  - Suggestion: Add a workflow to run `npm ci`, `npm run build`, and `npm test`; ensure `.wasm` assets are served with the correct MIME type; deploy to target.

### Nice‑to‑haves and polish

- **Recharts usage**
  - Gap: Listed in dependencies but not used.
  - Suggestion: Either remove the dependency or use Recharts for low‑frequency/static plots and keep Canvas for high‑frequency charts.

- **Font determinism**
  - Gap: No bundled font assets or `fonts.ts`; generator relies on system fonts.
  - Suggestion: Bundle a redistributable font and use it by default; keep `waitForFontsReady` guard; document expected variability across devices otherwise.

- **OffscreenCanvas**
  - Suggestion: Where supported, consider `OffscreenCanvas` for the mini chart and heatmaps to minimize main‑thread work during training.

- **Mobile defaults**
  - Suggestion: Detect lack of WASM SIMD/threads and reduce default batch sizes.

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


