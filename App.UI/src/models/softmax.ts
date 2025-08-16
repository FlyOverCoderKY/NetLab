import * as tf from "@tensorflow/tfjs";
import type { TeachModel, Visuals } from "./types";

export class SoftmaxModel implements TeachModel {
  name = "softmax" as const;
  inputShape: [number, number, number] = [28, 28, 1];
  private model: tf.LayersModel | null = null;
  private learningRate = 0.01;
  private optimizerType: "sgd" | "adam" = "sgd";
  private weightDecay = 0;

  async init(): Promise<void> {
    const m = tf.sequential();
    m.add(tf.layers.flatten({ inputShape: this.inputShape }));
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
    const lr = this.learningRate;
    const optimizer =
      this.optimizerType === "adam" ? tf.train.adam(lr) : tf.train.sgd(lr);
    let lossValue = 0;
    optimizer.minimize(() => {
      const logits = this.model!.predict(batch.x) as tf.Tensor2D;
      const onehot = tf.oneHot(batch.y as tf.Tensor1D, 36);
      const ce = tf.losses.softmaxCrossEntropy(onehot, logits).mean();
      let total = ce as tf.Tensor;
      if (this.weightDecay > 0) {
        const dense = this.model!.layers.find(
          (l) => l.getClassName() === "Dense",
        );
        if (dense) {
          const w = dense.getWeights()[0] as tf.Tensor;
          const l2 = tf.mul(0.5 * this.weightDecay, tf.sum(tf.square(w)));
          total = tf.add(total, l2);
        }
      }
      lossValue = (total.dataSync?.()[0] as number) ?? 0;
      return total as tf.Scalar;
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
    const dense = this.model.layers.find((l) => l.getClassName() === "Dense");
    if (!dense) return {};
    const weights = dense.getWeights();
    if (!weights.length) return {};
    const kernel = weights[0] as tf.Tensor2D; // shape [784,36]
    const arr = await kernel.array();
    const width = 28;
    const height = 28;
    const numClasses = arr[0]?.length ?? 36;
    const tiles: { name: string; grid: number[][] }[] = [];
    const tilesArr: {
      name: string;
      width: number;
      height: number;
      data: Float32Array;
    }[] = [];
    for (let c = 0; c < numClasses; c++) {
      const grid: number[][] = Array.from({ length: height }, () =>
        Array(width).fill(0),
      );
      // Extract column c from kernel and map into 28x28
      for (let i = 0; i < width * height; i++) {
        const y = Math.floor(i / width);
        const x = i % width;
        grid[y][x] = arr[i][c];
      }
      // Normalize to 0..1 for visualization
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
      const flat = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const norm = (grid[y][x] - min) / range;
          grid[y][x] = norm;
          flat[y * width + x] = norm;
        }
      }
      tiles.push({ name: `Class ${c}`, grid });
      tilesArr.push({ name: `Class ${c}`, width, height, data: flat });
    }
    return { weights: tiles, weightsArr: tilesArr };
  }

  async serialize(): Promise<Record<string, unknown>> {
    if (!this.model) return {};
    const layers = [] as Array<{
      type: string;
      weights: { shape: number[]; data: number[] }[];
    }>;
    for (const layer of this.model.layers) {
      const tensors = layer.getWeights();
      if (!tensors.length) continue;
      const weights: { shape: number[]; data: number[] }[] = [];
      for (const t of tensors) {
        const shape = t.shape.slice();
        const data = Array.from((await t.data()) as Float32Array);
        weights.push({ shape, data });
      }
      layers.push({ type: layer.getClassName(), weights });
    }
    return { name: this.name, inputShape: this.inputShape, layers };
  }

  async load(_: Record<string, unknown>): Promise<void> {
    if (!this.model) await this.init();
    const state = _ as {
      layers?: Array<{
        type: string;
        weights: { shape: number[]; data: number[] }[];
      }>;
    };
    if (!state?.layers) return;
    const tensorsPerLayer: tf.Tensor[] = [];
    for (const layerState of state.layers) {
      for (const w of layerState.weights) {
        tensorsPerLayer.push(tf.tensor(w.data, w.shape as number[], "float32"));
      }
    }
    const perLayer: tf.Tensor[][] = [];
    let idx = 0;
    for (const layer of this.model!.layers) {
      const expect = layer.getWeights().length;
      if (expect === 0) continue;
      perLayer.push(tensorsPerLayer.slice(idx, idx + expect));
      idx += expect;
    }
    // Apply weights layer by layer
    let li = 0;
    for (const layer of this.model!.layers) {
      const expect = layer.getWeights().length;
      if (expect === 0) continue;
      (layer as unknown as { setWeights: (w: tf.Tensor[]) => void }).setWeights(
        perLayer[li++] ?? [],
      );
    }
  }

  dispose(): void {
    if (this.model) this.model.dispose();
    this.model = null;
  }

  setHyperparams(h: {
    learningRate: number;
    optimizer: "sgd" | "adam";
    weightDecay?: number;
  }) {
    this.learningRate = Math.max(1e-5, Math.min(1, h.learningRate));
    this.optimizerType = h.optimizer;
    this.weightDecay = Math.max(0, h.weightDecay ?? 0);
  }
}
