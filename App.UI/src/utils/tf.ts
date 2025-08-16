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

export async function detectWasmCapabilities(): Promise<{
  simd: boolean;
  threads: boolean;
}> {
  const envGet = (tf as unknown as { env?: { get?: (k: string) => unknown } })
    .env?.get;
  const simd = Boolean(envGet?.("WASM_HAS_SIMD_SUPPORT"));
  const threads = Boolean(envGet?.("WASM_HAS_THREADS_SUPPORT"));
  return { simd, threads };
}
