import * as tf from "@tensorflow/tfjs";
import type { TeachModel, Visuals } from "./types";

export class MLPModel implements TeachModel {
  name = "mlp" as const;
  inputShape: [number, number, number] = [28, 28, 1];
  private model: tf.LayersModel | null = null;

  async init(): Promise<void> {
    const m = tf.sequential();
    m.add(tf.layers.flatten({ inputShape: this.inputShape }));
    m.add(tf.layers.dense({ units: 64, activation: "relu", useBias: true }));
    m.add(tf.layers.dense({ units: 36, useBias: true }));
    this.model = m;
  }

  async predict(
    x: tf.Tensor4D,
  ): Promise<{ logits: tf.Tensor2D; probs: tf.Tensor2D }> {
    if (!this.model) throw new Error("Model not initialized");
    const logits = this.model.predict(x) as tf.Tensor2D;
    const probs = tf.softmax(logits);
    return { logits, probs };
  }

  async trainStep(batch: {
    x: tf.Tensor4D;
    y: tf.Tensor1D;
  }): Promise<{ loss: number; visuals?: Visuals }> {
    if (!this.model) throw new Error("Model not initialized");
    const optimizer = tf.train.sgd(0.01);
    let lossValue = 0;
    optimizer.minimize(() => {
      const logits = this.model!.predict(batch.x) as tf.Tensor2D;
      const onehot = tf.oneHot(batch.y as tf.Tensor1D, 36);
      const loss = tf.losses.softmaxCrossEntropy(onehot, logits).mean();
      lossValue = (loss.dataSync?.()[0] as number) ?? 0;
      return loss as tf.Scalar;
    });
    optimizer.dispose();
    return { loss: lossValue };
  }

  async evaluate(
    _: Iterable<{ x: tf.Tensor4D; y: tf.Tensor1D }>,
  ): Promise<{ acc: number }> {
    void _;
    return { acc: 0 };
  }

  async getVisuals(): Promise<Visuals> {
    if (!this.model) return {};
    // Extract first Dense layer weights for receptive field tiles
    const dense1 = this.model.layers.find((l) => l.getClassName() === "Dense");
    if (!dense1) return {};
    const [kernel] = dense1.getWeights(); // shape [784, 64]
    if (!kernel) return {};
    const arr = (await (kernel as tf.Tensor2D).array()) as number[][];
    const width = 28;
    const height = 28;
    const numUnits = Math.min(arr[0]?.length ?? 0, 36); // cap to 36 tiles per snapshot
    const tiles: { name: string; grid: number[][] }[] = [];
    for (let u = 0; u < numUnits; u++) {
      const grid: number[][] = Array.from({ length: height }, () =>
        Array(width).fill(0),
      );
      for (let i = 0; i < width * height; i++) {
        const y = Math.floor(i / width);
        const x = i % width;
        grid[y][x] = arr[i][u];
      }
      let min = Infinity;
      let max = -Infinity;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const v = grid[y][x];
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      const range = max - min || 1;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          grid[y][x] = (grid[y][x] - min) / range;
        }
      }
      tiles.push({ name: `Unit ${u}`, grid });
    }
    return { weights: tiles };
  }

  async serialize(): Promise<Record<string, unknown>> {
    if (!this.model) return {};
    return { name: this.name, inputShape: this.inputShape };
  }

  async load(_: Record<string, unknown>): Promise<void> {
    void _;
  }

  dispose(): void {
    if (this.model) this.model.dispose();
    this.model = null;
  }
}
