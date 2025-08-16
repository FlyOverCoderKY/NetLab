import * as tf from "@tensorflow/tfjs";
import type { TeachModel, Visuals } from "./types";

export class CNNModel implements TeachModel {
  name = "cnn" as const;
  inputShape: [number, number, number] = [28, 28, 1];
  private model: tf.LayersModel | null = null;
  private learningRate = 0.01;
  private optimizerType: "sgd" | "adam" = "sgd";

  async init(): Promise<void> {
    const m = tf.sequential();
    m.add(
      tf.layers.conv2d({
        inputShape: this.inputShape,
        filters: 8,
        kernelSize: 5,
        activation: "relu",
        useBias: true,
        padding: "valid",
      }),
    );
    m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    m.add(tf.layers.flatten());
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
    const conv = this.model.layers.find((l) => l.getClassName() === "Conv2D");
    if (!conv) return {};
    const weights = conv.getWeights();
    if (!weights.length) return {};
    const kernel = weights[0] as tf.Tensor4D; // [kh, kw, inC, outC]
    const arr = (await kernel.array()) as number[][][][];
    const kh = arr.length;
    const kw = arr[0]?.length ?? 0;
    const outC = arr[0]?.[0]?.[0]?.length ?? 0;
    const tiles: { name: string; grid: number[][] }[] = [];
    for (let f = 0; f < outC; f++) {
      const grid: number[][] = Array.from({ length: kh }, () =>
        Array(kw).fill(0),
      );
      for (let y = 0; y < kh; y++) {
        for (let x = 0; x < kw; x++) {
          // inC is 1 for grayscale
          grid[y][x] = arr[y][x][0][f];
        }
      }
      // normalize
      let min = Infinity;
      let max = -Infinity;
      for (let y = 0; y < kh; y++) {
        for (let x = 0; x < kw; x++) {
          const v = grid[y][x];
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      const range = max - min || 1;
      for (let y = 0; y < kh; y++) {
        for (let x = 0; x < kw; x++) {
          grid[y][x] = (grid[y][x] - min) / range;
        }
      }
      tiles.push({ name: `Filter ${f}`, grid });
    }
    return { filters: tiles };
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

  setHyperparams(h: { learningRate: number; optimizer: "sgd" | "adam" }) {
    this.learningRate = Math.max(1e-5, Math.min(1, h.learningRate));
    this.optimizerType = h.optimizer;
  }
}
