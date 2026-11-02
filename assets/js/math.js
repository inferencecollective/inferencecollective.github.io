import renderMathInElement from "../vendor/auto-render.js";
import katex from "../vendor/katex.js";

/**
 * Initializes and renders all LaTeX math expressions across the site using KaTeX.
 * Supports:
 * - Block math: $$ ... $$ and \[ ... \]
 * - Inline math: $ ... $ and \( ... \)
 */
export function initMath(root = document.body) {
  if (!root) return;

  try {
    renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false
    });
  } catch (err) {
    console.warn("KaTeX math rendering warning:", err);
  }
}

export { katex, renderMathInElement };
