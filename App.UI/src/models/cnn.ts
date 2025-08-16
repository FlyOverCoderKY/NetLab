import * as tf from "@tensorflow/tfjs";
import type { TeachModel, Visuals } from "./types";

export class CNNModel implements TeachModel {
  name = "cnn" as const;
  inputShape: [number, number, number] = [28, 28, 1];
  private model: tf.LayersModel | null = null;
  private learningRate = 0.01;
  private optimizerType: "sgd" | "adam" = "sgd";
  private weightDecay = 0;

  async init(): Promise<void> {
    const m = tf.sequential();
    m.add(
      tf.layers.conv2d({
        inputShape: this.inputShape,
        filters: 8,
        kernelSize: 5,
        activation: "relu",
        useBias: true,
        padding: "same",
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
    tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
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
        return total as tf.Scalar;
      });
      lossValue = (value.dataSync?.()[0] as number) ?? 0;
      // Use type assertion to bypass the type checking issue with TensorFlow.js types
      (
        optimizer as { applyGradients: (grads: tf.NamedTensorMap) => void }
      ).applyGradients(grads);
      Object.values(grads).forEach((g) => g.dispose());
      value.dispose();
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
    const tilesArr: {
      name: string;
      width: number;
      height: number;
      data: Float32Array;
    }[] = [];
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
      const flat = new Float32Array(kh * kw);
      for (let y = 0; y < kh; y++) {
        for (let x = 0; x < kw; x++) {
          const norm = (grid[y][x] - min) / range;
          grid[y][x] = norm;
          flat[y * kw + x] = norm;
        }
      }
      tiles.push({ name: `Filter ${f}`, grid });
      tilesArr.push({ name: `Filter ${f}`, width: kw, height: kh, data: flat });
    }
    const visuals: Visuals = { filters: tiles, filtersArr: tilesArr };
    // Optional: first conv layer feature maps for current sample (12x12 after conv+pool)
    if (inputSample) {
      try {
        const fm = tf.tidy(() => {
          const layers = this.model!.layers;
          const convOut = (
            layers[0] as { apply: (x: tf.Tensor) => tf.Tensor }
          ).apply(inputSample) as tf.Tensor4D;
          const pooled = (
            layers[1] as { apply: (x: tf.Tensor) => tf.Tensor }
          ).apply(convOut) as tf.Tensor4D;
          return pooled as tf.Tensor4D;
        });
        const fArr = (await fm.array()) as number[][][][]; // [1,h,w,c]
        const h = fArr[0].length;
        const w = fArr[0][0].length;
        const c = fArr[0][0][0].length;
        const actTiles: { layer: string; grid: number[][] }[] = [];
        const actArr: {
          layer: string;
          width: number;
          height: number;
          data: Float32Array;
        }[] = [];
        for (let k = 0; k < Math.min(c, 8); k++) {
          const grid: number[][] = Array.from({ length: h }, () =>
            Array(w).fill(0),
          );
          let min = Infinity,
            max = -Infinity;
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const v = fArr[0][y][x][k];
              if (v < min) min = v;
              if (v > max) max = v;
            }
          }
          const range = max - min || 1;
          const flat = new Float32Array(h * w);
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const norm = (fArr[0][y][x][k] - min) / range;
              grid[y][x] = norm;
              flat[y * w + x] = norm;
            }
          }
          actTiles.push({ layer: "conv1", grid });
          actArr.push({ layer: "conv1", width: w, height: h, data: flat });
        }
        visuals.activations = actTiles;
        visuals.activationsArr = actArr;
        fm.dispose();
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
