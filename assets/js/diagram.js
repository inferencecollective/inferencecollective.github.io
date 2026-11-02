import { mount, render } from "../vendor/pele.js";

/**
 * Initializes and mounts all Mermaid diagrams using Obsidian Pele.
 * Pele is a synchronous, zero-dependency, ultra-fast SVG diagram renderer.
 *
 * Supports:
 * - Markdown code blocks: <pre><code class="language-mermaid">...</code></pre>
 * - Dedicated containers: <div class="mermaid">...</div> or <pre class="mermaid">...
 * - Data attributes: <div data-mermaid="..."></div>
 */
export function initDiagrams() {
  const codeBlocks = document.querySelectorAll("pre > code.language-mermaid, pre > code.mermaid");
  codeBlocks.forEach((codeEl) => {
    if (codeEl.dataset.peleRendered === "true") return;

    const preEl = codeEl.closest("pre");
    if (!preEl || preEl.dataset.peleRendered === "true") return;

    const source = (codeEl.textContent || "").trim();
    if (!source) return;

    try {
      const wrapper = document.createElement("div");
      wrapper.className = "pele-wrapper relative my-6 p-4 sm:p-6 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 shadow-xs overflow-x-auto";
      wrapper.dataset.peleMounted = "true";

      // Insert wrapper right before pre
      preEl.parentNode.insertBefore(wrapper, preEl);
      preEl.style.display = "none";

      try {
        mount(wrapper, source, { zoom: "auto" });
      } catch (mountErr) {
        // Fallback directly to synchronous render if mount encounters layout race
        const result = render(source);
        if (result && result.svg) {
          wrapper.innerHTML = result.svg;
        } else {
          throw mountErr;
        }
      }

      codeEl.dataset.peleRendered = "true";
      preEl.dataset.peleRendered = "true";
    } catch (err) {
      console.warn("Pele diagram render failed:", err);
      if (preEl) preEl.style.display = "";
    }
  });

  const containers = document.querySelectorAll("div.mermaid, pre.mermaid, [data-mermaid]");
  containers.forEach((el) => {
    if (el.dataset.peleRendered === "true") return;

    const source = (el.getAttribute("data-mermaid") || el.textContent || "").trim();
    if (!source) return;

    try {
      if (el.tagName.toLowerCase() === "pre") {
        const wrapper = document.createElement("div");
        wrapper.className = "pele-wrapper relative my-6 p-4 sm:p-6 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 shadow-xs overflow-x-auto";
        el.parentNode.insertBefore(wrapper, el);
        el.style.display = "none";
        mount(wrapper, source, { zoom: "auto" });
        el.dataset.peleRendered = "true";
        wrapper.dataset.peleMounted = "true";
      } else {
        el.classList.add("pele-wrapper", "overflow-x-auto");
        mount(el, source, { zoom: "auto" });
        el.dataset.peleRendered = "true";
      }
    } catch (err) {
      console.warn("Pele diagram render failed:", err);
      try {
        const result = render(source);
        if (result && result.svg) {
          el.innerHTML = result.svg;
          el.dataset.peleRendered = "true";
        }
      } catch (renderErr) {
        console.warn("Pele fallback render also failed:", renderErr);
      }
    }
  });
}

export { render, mount };
