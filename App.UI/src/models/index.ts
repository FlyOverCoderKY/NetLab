import { SoftmaxModel } from "./softmax";
import { MLPModel } from "./mlp";
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
    default:
      // Placeholder: default to softmax until CNN is implemented
      return new SoftmaxModel();
  }
}
