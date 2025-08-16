import { SoftmaxModel } from "./softmax";
import { MLPModel } from "./mlp";
import { CNNModel } from "./cnn";
export type { TeachModel } from "./types";
export { SoftmaxModel } from "./softmax";
export { MLPModel } from "./mlp";

export function createModel(name: "softmax" | "mlp" | "cnn") {
  switch (name) {
    case "softmax":
      return new SoftmaxModel();
    case "mlp":
      return new MLPModel();
    case "cnn":
      return new CNNModel();
    default:
      return new SoftmaxModel();
  }
}
