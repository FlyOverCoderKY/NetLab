import * as tf from "@tensorflow/tfjs";
import type { TeachModel, Visuals } from "./types";

export class MLPModel implements TeachModel {
  name = "mlp" as const;
  inputShape: [number, number, number] = [28, 28, 1];
  private model: tf.LayersModel | null = null;
  private learningRate = 0.01;
  private optimizerType: "sgd" | "adam" = "sgd";
  private weightDecay = 0;

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
        for (const layer of this.model!.layers) {
          const ws = layer.getWeights();
          if (ws.length > 0) {
            const w = ws[0];
            const l2 = tf.mul(0.5 * this.weightDecay, tf.sum(tf.square(w)));
            total = tf.add(total, l2);
          }
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

  async getVisuals(inputSample?: tf.Tensor4D): Promise<Visuals> {
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
    const visuals: Visuals = { weights: tiles };
    // Optional: hidden activation bars on current sample
    if (inputSample) {
      try {
        const acts = tf.tidy(() => {
          const layers = this.model!.layers;
          const flat = (
            layers[0] as { apply: (x: tf.Tensor) => tf.Tensor }
          ).apply(inputSample) as tf.Tensor2D;
          const hidden = (
            layers[1] as { apply: (x: tf.Tensor) => tf.Tensor }
          ).apply(flat) as tf.Tensor2D; // [1,64]
          return hidden;
        });
        const arr = Array.from((await acts.data()) as Float32Array);
        const hBars = Math.min(arr.length, 36);
        const barTiles: { layer: string; grid: number[][] }[] = [];
        const barArr: {
          layer: string;
          width: number;
          height: number;
          data: Float32Array;
        }[] = [];
        for (let i = 0; i < hBars; i++) {
          const v = Math.max(0, Math.min(1, arr[i]));
          // Make a 1x16 bar scaled by v
          const len = 16;
          const grid = [
            Array.from({ length: len }, (_, x) => (x / (len - 1) <= v ? v : 0)),
          ];
          const flat = new Float32Array(len);
          for (let x = 0; x < len; x++) flat[x] = grid[0][x];
          barTiles.push({ layer: `h${i}`, grid });
          barArr.push({ layer: `h${i}`, width: len, height: 1, data: flat });
        }
        visuals.activations = barTiles;
        visuals.activationsArr = barArr;
        acts.dispose();
      } catch {
        // ignore
      }
    }
    return visuals;
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
