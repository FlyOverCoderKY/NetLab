import { XorShift32 } from "./seed";

export type GlyphParams = {
  fontFamily: string;
  fontSize: number;
  thickness: number;
  jitterPx: number;
  rotationDeg: number;
  invert: boolean;
  noise: boolean;
};

export async function waitForFontsReady(): Promise<void> {
  const hasDocument = typeof document !== "undefined";
  if (!hasDocument) return;
  const doc = document as Document & { fonts?: FontFaceSet };
  const fonts = doc.fonts;
  if (fonts && typeof fonts.ready?.then === "function") {
    await fonts.ready;
  }
}

export function renderGlyphTo28x28(
  char: string,
  params: GlyphParams,
  rng: XorShift32,
): Float32Array {
  const hasDocument = typeof document !== "undefined";
  const Offscreen = (globalThis as unknown as { OffscreenCanvas?: unknown })
    .OffscreenCanvas as unknown as
    | (new (w: number, h: number) => OffscreenCanvas)
    | undefined;

  let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null =
    null;
  if (hasDocument) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = 28;
      canvas.height = 28;
      ctx = canvas.getContext("2d");
    } catch {
      ctx = null;
    }
  } else if (typeof Offscreen === "function") {
    try {
      const canvas = new Offscreen(28, 28);
      ctx = canvas.getContext("2d");
    } catch {
      ctx = null;
    }
  }

  if (!ctx) {
    // Fallback: generate a simple synthetic pattern in absence of any canvas
    const out = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const dx = x - 14;
        const dy = y - 14;
        const r = Math.sqrt(dx * dx + dy * dy);
        const v = Math.max(0, 1 - r / 14);
        out[y * 28 + x] = params.invert ? 1 - v : v;
      }
    }
    return out;
  }

  try {
    ctx.clearRect(0, 0, 28, 28);
    (ctx as CanvasRenderingContext2D).fillStyle = params.invert
      ? "#000"
      : "#fff";
    ctx.fillRect(0, 0, 28, 28);

    ctx.save();
    const jitterX = (rng.next() * 2 - 1) * params.jitterPx;
    const jitterY = (rng.next() * 2 - 1) * params.jitterPx;
    const theta = (params.rotationDeg / 180) * Math.PI * (rng.next() * 2 - 1);
    ctx.translate(14 + jitterX, 14 + jitterY);
    ctx.rotate(theta);
    ctx.textAlign = "center" as CanvasTextAlign;
    ctx.textBaseline = "middle" as CanvasTextBaseline;
    (ctx as CanvasRenderingContext2D).fillStyle = params.invert
      ? "#fff"
      : "#000";
    (ctx as CanvasRenderingContext2D).font =
      `${params.fontSize}px ${params.fontFamily}, sans-serif`;
    // Drawing text – in workers, fonts may differ; acceptable for current use.
    ctx.fillText(char, 0, 0);
    ctx.restore();

    if (params.noise) {
      const img = ctx.getImageData(0, 0, 28, 28);
      for (let i = 0; i < img.data.length; i += 4) {
        const n = (rng.next() - 0.5) * 12;
        img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
        img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
        img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n));
      }
      ctx.putImageData(img, 0, 0);
    }

    const imgData = ctx.getImageData(0, 0, 28, 28);
    const out = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const idx = (y * 28 + x) * 4;
        const rVal = imgData.data[idx];
        const gVal = imgData.data[idx + 1];
        const bVal = imgData.data[idx + 2];
        const gray = (rVal + gVal + bVal) / (3 * 255);
        out[y * 28 + x] = params.invert ? 1 - gray : gray;
      }
    }
    return out;
  } catch {
    // If any 2D context operation is not available, fall back to synthetic pattern
    const out = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const dx = x - 14;
        const dy = y - 14;
        const r = Math.sqrt(dx * dx + dy * dy);
        const v = Math.max(0, 1 - r / 14);
        out[y * 28 + x] = params.invert ? 1 - v : v;
      }
    }
    return out;
  }
}

export const CLASS_LIST: string[] = [
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."0123456789",
];
