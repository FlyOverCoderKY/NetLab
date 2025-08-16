// Default to a widely available system font to avoid loading a missing asset
export const BUNDLED_FONT_FAMILY = "Inter";

export function registerBundledFonts(): void {
  // No-op until we bundle actual font assets
  return;
}

export async function waitForBundledFontReady(): Promise<void> {
  try {
    const fonts: FontFaceSet | undefined = (
      document as Document & { fonts?: FontFaceSet }
    ).fonts;
    if (fonts && typeof fonts.load === "function") {
      await fonts.load(`16px ${BUNDLED_FONT_FAMILY}`);
      const ready = (fonts as unknown as { ready?: Promise<unknown> }).ready;
      await ready?.catch?.(() => void 0);
    }
  } catch {
    // ignore
  }
}
