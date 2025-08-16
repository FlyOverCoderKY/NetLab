import type * as tf from "@tensorflow/tfjs";

export type Grid = number[][];

export type Visuals = {
  weights?: { name: string; grid: Grid }[];
  filters?: { name: string; grid: Grid }[];
  activations?: { layer: string; grid: Grid }[];
};

export interface TeachModel {
  name: "softmax" | "mlp" | "cnn";
  inputShape: [number, number, number];
  init(params?: Record<string, unknown>): Promise<void>;
  predict(x: tf.Tensor4D): Promise<{ logits: tf.Tensor2D; probs: tf.Tensor2D }>;
  trainStep(batch: {
    x: tf.Tensor4D;
    y: tf.Tensor1D;
  }): Promise<{ loss: number; visuals?: Visuals }>;
  evaluate(
    ds: Iterable<{ x: tf.Tensor4D; y: tf.Tensor1D }>,
  ): Promise<{ acc: number }>;
  getVisuals(): Promise<Visuals>;
  serialize(): Promise<Record<string, unknown>>;
  load(state: Record<string, unknown>): Promise<void>;
  dispose(): void;
}
