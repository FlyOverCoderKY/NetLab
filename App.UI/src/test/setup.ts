import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";
// Extend expect with jest-dom matchers for JSDOM
import "@testing-library/jest-dom/vitest";

// In jsdom, canvas getContext is not implemented and throws "Not implemented".
// Stub it to return null so our code takes the graceful fallback path without noisy errors.
try {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (HTMLCanvasElement.prototype as any).getContext = () => null;
} catch {
  // ignore
}

// Use WASM backend in tests for stability
beforeAll(async () => {
  try {
    await tf.setBackend("wasm");
    await tf.ready();
  } catch {
    // Fallback to CPU if WASM is not available in the test env
    await tf.setBackend("cpu");
    await tf.ready();
  }
});
