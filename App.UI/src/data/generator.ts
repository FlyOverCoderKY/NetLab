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
  const fonts = (document as Document & { fonts?: FontFaceSet }).fonts;
  if (fonts && typeof fonts.ready?.then === "function") {
    await fonts.ready;
  }
}

export function renderGlyphTo28x28(
  char: string,
  params: GlyphParams,
  rng: XorShift32,
): Float32Array {
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext("2d")!;

  ctx.clearRect(0, 0, 28, 28);
  ctx.fillStyle = params.invert ? "#000" : "#fff";
  ctx.fillRect(0, 0, 28, 28);

  ctx.save();
  const jitterX = (rng.next() * 2 - 1) * params.jitterPx;
  const jitterY = (rng.next() * 2 - 1) * params.jitterPx;
  const theta = (params.rotationDeg / 180) * Math.PI * (rng.next() * 2 - 1);
  ctx.translate(14 + jitterX, 14 + jitterY);
  ctx.rotate(theta);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = params.invert ? "#fff" : "#000";
  ctx.font = `${params.thickness}px ${params.fontFamily}, sans-serif`;
  ctx.font = `${params.fontSize}px ${params.fontFamily}, sans-serif`;
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
      const r = imgData.data[idx];
      const g = imgData.data[idx + 1];
      const b = imgData.data[idx + 2];
      const gray = (r + g + b) / (3 * 255);
      out[y * 28 + x] = params.invert ? 1 - gray : gray;
    }
  }
  return out;
}

export const CLASS_LIST: string[] = [
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."0123456789",
];
