import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-wasm";

export async function initTFBackend(): Promise<string> {
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    if (tf.getBackend() === "webgl") {
      return "webgl";
    }
  } catch (_) {
    // fall through to wasm
  }
  await tf.setBackend("wasm");
  await tf.ready();
  return tf.getBackend();
}

export function getTFBackend(): string {
  return tf.getBackend();
}
