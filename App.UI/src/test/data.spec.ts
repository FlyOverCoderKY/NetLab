import { describe, it, expect } from "vitest";
import { renderGlyphTo28x28 } from "../data/generator";
import { XorShift32 } from "../data/seed";

const defaultParams = {
  fontFamily: "Inter",
  fontSize: 20,
  thickness: 20,
  jitterPx: 1,
  rotationDeg: 0,
  invert: false,
  noise: false,
};

describe("data generator determinism", () => {
  it("produces identical outputs for same seed and params", () => {
    const rng1 = new XorShift32(1234);
    const rng2 = new XorShift32(1234);
    const a = renderGlyphTo28x28("A", defaultParams, rng1);
    const b = renderGlyphTo28x28("A", defaultParams, rng2);
    expect(a.length).toBe(28 * 28);
    expect(b.length).toBe(28 * 28);
    // exact match
    let same = true;
    for (let i = 0; i < a.length; i++)
      if (a[i] !== b[i]) {
        same = false;
        break;
      }
    expect(same).toBe(true);
  });
});
