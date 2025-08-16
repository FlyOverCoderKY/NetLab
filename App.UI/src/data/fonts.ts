export const BUNDLED_FONT_FAMILY = "NetLabDemo";

export function registerBundledFonts(): void {
  try {
    const id = "netlab-font-style";
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = `@font-face{font-family:${BUNDLED_FONT_FAMILY};src:url('/fonts/NetLabDemo.woff2') format('woff2');font-weight:400;font-style:normal;font-display:swap}`;
    document.head.appendChild(style);
  } catch {
    // ignore
  }
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
