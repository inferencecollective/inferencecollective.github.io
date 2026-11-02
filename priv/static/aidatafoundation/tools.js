import { entries } from "/aidatafoundation/entries.js";

const root = document.querySelector("#kube-tools") || document.querySelector("#adf-tools");
const tags = [...new Set(entries.map(({ tag }) => tag).filter(Boolean))].sort();
const pageSize = 24;
let page = 1;
let currentTier = "";

function escape(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[character]);
}

function projectUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function formatStars(stars) {
  if (!stars || stars <= 0) return "";
  if (stars >= 1000) {
    return (stars / 1000).toFixed(1) + "k";
  }
  return stars.toLocaleString();
}

function maturityBadge(maturity) {
  const m = (maturity || "").toLowerCase();
  if (m === "graduated") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/30"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Graduated</span>`;
  }
  if (m === "incubating") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-sky-500/15 text-sky-700 dark:text-sky-400 border border-sky-500/30"><span class="w-1.5 h-1.5 rounded-full bg-sky-500"></span> Incubating</span>`;
  }
  if (m === "sandbox") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span> Sandbox</span>`;
  }
  return "";
}

function card(entry) {
  const link = projectUrl(entry.link);
  const starsFormatted = formatStars(entry.stars);
  const matBadge = maturityBadge(entry.maturity);
  
  return `
    <article class="p-6 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 shadow-xs hover:border-emerald-500/40 hover:shadow-md transition-all flex flex-col justify-between group">
      <div>
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div class="flex items-center gap-1.5 flex-wrap">
            ${matBadge || `<span class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">${escape(entry.tag || "AI Tool")}</span>`}
            ${entry.subcategory ? `<span class="text-[11px] text-slate-500 dark:text-slate-400 font-medium">${escape(entry.subcategory)}</span>` : ""}
          </div>
          ${
            starsFormatted
              ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-xs font-bold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200/60 dark:border-amber-800/40">
                  <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
                  ${starsFormatted}
                </span>`
              : ""
          }
        </div>
        <h3 class="text-lg font-bold text-slate-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
          ${escape(entry.title)}
        </h3>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-2 line-clamp-3 leading-relaxed">
          ${escape(entry.description || "Explore this open-source AI project, documentation, and installation instructions.")}
        </p>
      </div>
      <div class="mt-5 pt-4 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between">
        ${
          link
            ? `<a class="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-400 hover:underline" href="${escape(link)}" target="_blank" rel="noopener noreferrer">Visit Project &nearr;</a>`
            : '<span class="text-slate-400 text-xs">Link unavailable</span>'
        }
        ${
          entry.github
            ? `<a href="https://github.com/${escape(entry.github)}" target="_blank" rel="noopener noreferrer" class="text-slate-400 hover:text-slate-900 dark:hover:text-white text-xs inline-flex items-center gap-1" title="GitHub Repository">
                <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
                <span>GitHub</span>
              </a>`
            : ""
        }
      </div>
    </article>
  `;
}

if (root) {
  root.innerHTML = `
    <!-- CNCF / LF AI Landscape Tier Quick Filters -->
    <div class="mb-6 flex flex-wrap items-center gap-2" id="tier-filters">
      <button type="button" data-tier="" class="tier-pill active px-4 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer bg-slate-900 text-white dark:bg-white dark:text-slate-900 border-slate-900 dark:border-white shadow-xs">
        All Projects
      </button>
      <button type="button" data-tier="graduated" class="tier-pill px-4 py-2 rounded-xl text-xs font-bold border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-emerald-500/50 hover:text-emerald-600 transition-all cursor-pointer">
        ⭐ Graduated (13)
      </button>
      <button type="button" data-tier="incubating" class="tier-pill px-4 py-2 rounded-xl text-xs font-bold border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-sky-500/50 hover:text-sky-600 transition-all cursor-pointer">
        🚀 Incubating (22)
      </button>
      <button type="button" data-tier="sandbox" class="tier-pill px-4 py-2 rounded-xl text-xs font-bold border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-amber-500/50 hover:text-amber-600 transition-all cursor-pointer">
        🌱 Sandbox (22)
      </button>
      <button type="button" data-tier="mcp" class="tier-pill px-4 py-2 rounded-xl text-xs font-bold border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-emerald-500/50 hover:text-emerald-600 transition-all cursor-pointer">
        🔌 MCP Servers
      </button>
    </div>

    <form class="kube-tool-controls adf-tool-controls flex flex-col md:flex-row gap-3 p-4 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 mb-6" role="search" aria-label="Find AI and LLM tools">
      <div class="flex-1 relative">
        <label for="tool-query" class="sr-only">Search projects</label>
        <input id="tool-query" name="q" type="search" placeholder="Search Milvus, ONNX, Docling, Flyte, Feast, Ludwig, MCP servers…" autocomplete="off" class="w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500">
      </div>
      <div class="flex flex-wrap items-center gap-3">
        <label for="tool-category" class="sr-only">Category</label>
        <select id="tool-category" name="category" class="px-3.5 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500">
          <option value="">All Categories (${tags.length})</option>
          ${tags.map(tag => `<option value="${escape(tag)}">${escape(tag)}</option>`).join("")}
        </select>
        <label for="tool-sort" class="sr-only">Sort by</label>
        <select id="tool-sort" name="sort" class="px-3.5 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500">
          <option value="stars">Most GitHub Stars</option>
          <option value="az">Name: A–Z</option>
          <option value="za">Name: Z–A</option>
        </select>
        <button type="button" class="kube-button kube-button-outline kube-filter-reset adf-filter-reset text-xs py-2 px-3">Reset Filter</button>
      </div>
    </form>

    <div class="my-4 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
      <p class="kube-tool-count adf-tool-count" role="status" aria-live="polite" aria-atomic="true"></p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="tool-results"></div>
    <div class="mt-12 text-center">
      <button type="button" class="kube-button" id="tool-more">Load More Projects</button>
    </div>
  `;

  const form = root.querySelector("form");
  const input = root.querySelector("#tool-query");
  const category = root.querySelector("#tool-category");
  const sort = root.querySelector("#tool-sort");
  const results = root.querySelector("#tool-results");
  const count = root.querySelector('[role="status"]');
  const more = root.querySelector("#tool-more");
  const clear = root.querySelector(".kube-filter-reset");
  const tierPills = root.querySelectorAll(".tier-pill");

  function restore() {
    const params = new URLSearchParams(location.search);
    input.value = params.get("q") || "";
    category.value = tags.includes(params.get("category")) ? params.get("category") : "";
    sort.value = ["stars", "za", "az"].includes(params.get("sort")) ? params.get("sort") : "stars";
    currentTier = params.get("tier") || "";
    updateTierPills();
    page = 1;
  }

  function updateTierPills() {
    tierPills.forEach(pill => {
      const tier = pill.getAttribute("data-tier");
      if (tier === currentTier) {
        pill.className = "tier-pill active px-4 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer bg-slate-900 text-white dark:bg-white dark:text-slate-900 border-slate-900 dark:border-white shadow-xs";
      } else {
        pill.className = "tier-pill px-4 py-2 rounded-xl text-xs font-bold border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-emerald-500/50 hover:text-emerald-600 transition-all cursor-pointer";
      }
    });
  }

  function updateUrl() {
    const url = new URL(location.href);
    for (const [key, value] of [
      ["q", input.value],
      ["category", category.value],
      ["sort", sort.value === "stars" ? "" : sort.value],
      ["tier", currentTier]
    ]) {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    history.replaceState(null, "", url);
  }

  function render({ append = false } = {}) {
    const words = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const visible = entries
      .filter((entry) => {
        const haystack = `${entry.title} ${entry.description || ""} ${entry.tag || ""} ${entry.category || ""} ${entry.subcategory || ""}`.toLowerCase();
        
        // Tier filter
        if (currentTier === "graduated" && entry.maturity !== "graduated") return false;
        if (currentTier === "incubating" && entry.maturity !== "incubating") return false;
        if (currentTier === "sandbox" && entry.maturity !== "sandbox") return false;
        if (currentTier === "mcp" && !String(entry.tag || "").includes("MCP")) return false;

        return (
          (!category.value || entry.tag === category.value) &&
          words.every((word) => haystack.includes(word))
        );
      })
      .sort((a, b) => {
        if (sort.value === "stars") {
          return (b.stars || 0) - (a.stars || 0);
        }
        if (sort.value === "za") {
          return -1 * a.title.localeCompare(b.title);
        }
        return a.title.localeCompare(b.title);
      });

    const limit = Math.min(page * pageSize, visible.length);
    const cards = visible
      .slice(append ? (page - 1) * pageSize : 0, limit)
      .map(card)
      .join("");

    if (append) {
      results.insertAdjacentHTML("beforeend", cards);
    } else {
      results.innerHTML =
        cards ||
        `
        <div class="col-span-full py-16 text-center">
          <div class="inline-flex items-center justify-center w-12 h-12 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 mb-4 font-bold text-xl">
            ?
          </div>
          <h3 class="text-xl font-bold text-slate-900 dark:text-white mb-2">No matching projects found</h3>
          <p class="text-slate-500 max-w-md mx-auto text-sm">Try broader keywords or reset your filters to browse the complete landscape.</p>
        </div>
      `;
    }

    count.textContent = `Showing ${limit} of ${visible.length} projects (${entries.length} total indexed in LF AI & Data Landscape)`;
    more.hidden = limit >= visible.length;
    clear.disabled = !input.value && !category.value && sort.value === "stars" && !currentTier;
  }

  tierPills.forEach(pill => {
    pill.addEventListener("click", () => {
      currentTier = pill.getAttribute("data-tier") || "";
      updateTierPills();
      page = 1;
      updateUrl();
      render();
    });
  });

  form.addEventListener("submit", (event) => event.preventDefault());
  input.addEventListener("input", () => {
    page = 1;
    updateUrl();
    render();
  });
  for (const select of [category, sort]) {
    select.addEventListener("change", () => {
      page = 1;
      updateUrl();
      render();
    });
  }
  clear.addEventListener("click", () => {
    input.value = "";
    category.value = "";
    sort.value = "stars";
    currentTier = "";
    updateTierPills();
    page = 1;
    updateUrl();
    render();
    input.focus();
  });
  more.addEventListener("click", () => {
    page += 1;
    render({ append: true });
  });
  window.addEventListener("popstate", () => {
    restore();
    render();
  });

  restore();
  render();
}

const starCounters = document.querySelectorAll("#kube-github-stars, #adf-github-stars");
if (starCounters.length > 0) {
  fetch("https://api.github.com/repos/aidatafoundation/aidatafoundation.github.io", {
    signal: AbortSignal.timeout(5000)
  })
    .then((res) => (res.ok ? res.json() : null))
    .then((repo) => {
      if (Number.isFinite(repo?.stargazers_count)) {
        const text = `${repo.stargazers_count.toLocaleString()} GitHub stars`;
        starCounters.forEach(el => el.textContent = text);
      }
    })
    .catch(() => {});
}