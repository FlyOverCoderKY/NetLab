# NetLab UI

Interactive neural network visualizer and trainer (Softmax, MLP, Tiny CNN) running fully in the browser with TensorFlow.js. Uses a Web Worker for training and streams visuals/metrics to the UI.

## Tech

- React 18 + TypeScript + Vite
- TensorFlow.js (WebGL main thread, WASM in worker)
- Zustand for app state
- Canvas-based rendering for fast visuals
- Vitest + Testing Library for tests

## Getting Started

```bash
cd App.UI
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Scripts

- `npm run dev` – start Vite dev server
- `npm run build` – typecheck and production build
- `npm run preview` – preview dist build
- `npm run test` – run unit/component tests (jsdom)
- `npm run lint` – run ESLint

## Notes

- The worker is pinned to the WASM backend; the main thread prefers WebGL with WASM fallback.
- Visuals are throttled and sent as transfer-optimized Float32Array buffers.
- Dataset preview and training parameters are configurable via the UI.

## License

MIT (see root `LICENSE`).
