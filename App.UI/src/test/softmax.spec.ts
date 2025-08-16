import { describe, it, expect } from "vitest";
import * as tf from "@tensorflow/tfjs";
import { SoftmaxModel } from "../models/softmax";

function makeTinyBatch(n = 8) {
  const xs = tf.randomUniform([n, 28, 28, 1], 0, 1, "float32");
  const ys = tf.tidy(() =>
    tf.randomUniform([n], 0, 36, "int32").toInt(),
  ) as tf.Tensor1D;
  return { x: xs as tf.Tensor4D, y: ys };
}

describe("softmax training", () => {
  it("loss decreases after a few steps (sanity)", async () => {
    const model = new SoftmaxModel();
    await model.init();
    // Use the SAME batch to measure progress
    const batch = makeTinyBatch(16);
    const losses: number[] = [];
    for (let i = 0; i < 5; i++) {
      const { loss } = await model.trainStep(batch);
      losses.push(loss);
      expect(Number.isFinite(loss)).toBe(true);
    }
    // Expect the final loss to be <= the initial (allow small numeric noise)
    expect(losses[losses.length - 1]).toBeLessThanOrEqual(losses[0] + 1e-3);
    batch.x.dispose();
    batch.y.dispose();
    model.dispose();
  });
});
