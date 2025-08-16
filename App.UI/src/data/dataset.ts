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
  // Round-robin with jittered classes for mild balance
  let index = 0;
  while (true) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      const classIdx =
        (index + i + rng.nextInt(CLASS_LIST.length)) % CLASS_LIST.length;
      const ch = CLASS_LIST[classIdx];
      const img = renderGlyphTo28x28(ch, params, rng);
      for (let p = 0; p < 28 * 28; p++) xs.push(img[p]);
      ys.push(classIdx);
    }
    index = (index + batchSize) % CLASS_LIST.length;
    const x = tf.tensor4d(xs, [batchSize, 28, 28, 1], "float32");
    const y = tf.tensor1d(ys, "int32");
    yield { x, y };
  }
}
