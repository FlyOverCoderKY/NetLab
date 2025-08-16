import * as tf from "@tensorflow/tfjs";
import { CLASS_LIST, GlyphParams, renderGlyphTo28x28 } from "./generator";
import { XorShift32 } from "./seed";

export type DatasetParams = GlyphParams & { seed: number };

export type Batch = { x: tf.Tensor4D; y: tf.Tensor1D };

export function* makeDataset(
  params: DatasetParams,
  batchSize: number,
): Generator<Batch> {
  const rng = new XorShift32(params.seed);
  const bs = Math.max(1, Math.min(1024, Math.floor(batchSize) || 1));
  // Round-robin with jittered classes for mild balance
  let index = 0;
  let iter = 0;
  while (true) {
    const xs = new Float32Array(bs * 28 * 28);
    const ys = new Int32Array(bs);
    for (let i = 0; i < bs; i++) {
      // Deterministic round-robin over classes for stable supervision
      const classIdx = (index + i) % CLASS_LIST.length;
      const ch = CLASS_LIST[classIdx];
      const img = renderGlyphTo28x28(ch, params, rng);
      xs.set(img, i * 28 * 28);
      ys[i] = classIdx | 0;
    }
    index = (index + bs) % CLASS_LIST.length;
    try {
      if (iter % 200 === 0) {
        console.debug("makeDataset", { iter, bs, xsLen: xs.length });
      }
      const x = tf.tensor4d(xs, [bs, 28, 28, 1], "float32");
      const y = tf.tensor1d(ys, "int32");
      iter += 1;
      yield { x, y };
    } catch (err) {
      console.error("makeDataset tensor construction failed", {
        iter,
        bs,
        xsLen: xs.length,
        err,
      });
      throw err;
    }
  }
}
