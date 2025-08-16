export class XorShift32 {
  private state: number;

  constructor(seed: number) {
    // Ensure non-zero 32-bit state
    this.state = seed | 0 || 0x9e3779b9;
  }

  /** Returns a float in [0, 1) */
  next(): number {
    let x = this.state | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x | 0;
    // Convert to [0,1)
    return (x >>> 0) / 0x100000000;
  }

  /** Integer in [0, n) */
  nextInt(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Gaussian via Box–Muller */
  nextGaussian(mu = 0, sigma = 1): number {
    let u = 0,
      v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mu + z * sigma;
  }
}
