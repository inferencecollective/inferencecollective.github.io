import { entries } from "/aidatafoundation/entries.js";

const root = document.querySelector("#kube-tools") || document.querySelector("#adf-tools");
const tags = [...new Set(entries.map(({ tag }) => tag).filter(Boolean))].sort();
const pageSize = 24;
let page = 1;
let currentTier = "";

const projectLogos = {
  "docling": "https://lfai.landscape2.io/logos/9d44b3470b9e33063c570ef5202c336bfecbd21a8fe9f5df49aba63a5e327de7.svg",
  "milvus": "https://lfai.landscape2.io/logos/fe575e6ec12d7cc38be20f35320abfeae45c23c7222851a5223046fc061e64dc.svg",
  "onnx": "https://lfai.landscape2.io/logos/00e81757af5c54743ecebb70a603871898e87ca543188caa0623acef441a831e.svg",
  "kedro": "https://lfai.landscape2.io/logos/67316a9bd73b4499de748b71b61ff1f3aa692590f25958bafee3a1d104aaefc6.svg",
  "pyro": "https://lfai.landscape2.io/logos/b5b08b9853f4c85567da899c277269716c008817fa6b6035ae54d402eb828519.svg",
  "flyte": "https://lfai.landscape2.io/logos/b6bb210370181e7508656146b5a126c426ed0943da463e95f51edd30b4bee9a5.svg",
  "angel-ml": "https://lfai.landscape2.io/logos/a3731bc9233c00ec0d37eeae53afdbcbcbd3f10b6a6214641a6fc541d982f385.svg",
  "adversarial robustness toolkit": "https://lfai.landscape2.io/logos/ef9e520f54489cebf8f794fc75d2d225179d65ddd108cbd503e0a8515efe0767.svg",
  "docarray": "https://lfai.landscape2.io/logos/61ad13af9d0f874996bf23ff188affa619ad3857a35fcac89fa85dcae05ef4d9.svg",
  "openlineage": "https://lfai.landscape2.io/logos/6085536cce4892ef5dbdb5aa0d6d3d7ae610adbc1a8dee13880cc7f91bb6f925.svg",
  "marquez": "https://lfai.landscape2.io/logos/a47d7ea58c501a4e4a4f3699cb14fd68c2b078447b8133482fa86e0063368674.svg",
  "bitol": "https://lfai.landscape2.io/logos/2398bc2f37c3106a2cb3dfd4f94cb6a65e23dd9987aff09b1a0389724823970c.svg",
  "egeria": "https://lfai.landscape2.io/logos/7258c5df9d3a9b9834ab527adc6ab5b65fcfab2647ec9bcd6571ebed1e01750a.svg",
  "ludwig": "https://lfai.landscape2.io/logos/dce2da56c68f55f968ac980ceed8b2cafca62610f648f38d4ddd5d0b42f96afc.svg",
  "delta lake": "https://lfai.landscape2.io/logos/b52a55b1ce3bec1177f86b4878b39120a2ce476dd0c5d6d591dcd8b89da2ed2c.svg",
  "feast": "https://lfai.landscape2.io/logos/cb053c87a423e875eeec8bd2adbf11c013be0d7583cc206393aea545f06f0f1c.svg",
  "fate": "https://lfai.landscape2.io/logos/95e570a1597eb1d4a5f4c3d167463024864562c4cc6c144e4564913ffd070956.svg",
  "janusgraph": "https://lfai.landscape2.io/logos/b4fcdacd992712b81c4ed4340f918ee457fd8b15844f507fe4890fd721a245d4.svg",
  "beeai": "https://lfai.landscape2.io/logos/32d5927b73c8c67d4c631f4f08e56a57138b3892ba53dfcc8cef41b77fa008f8.svg",
  "ai fairness 360 toolkit": "https://lfai.landscape2.io/logos/b9ee060a23ede8270ba8ad094359e07e2bb22f7329ca85bf5062b23023a8c6f7.svg",
  "kompute": "https://lfai.landscape2.io/logos/5502d046cbe8595182f34bdacd4a8abded8f206bdb240068ffc04e7106c163d7.svg",
  "elyra": "https://lfai.landscape2.io/logos/9c6d206c8ac62aec0016067986e434a3483be0e90e3bb04fb541933a2b561618.svg",
  "ai explainability 360 toolkit": "https://lfai.landscape2.io/logos/30d98d4b12175f9d19a9ab5cb1185fbacf526991d0e6c450a11024ac004dc393.svg",
  "sparklyr": "https://lfai.landscape2.io/logos/05ef8e8ef2770c6b16a4d5222f09aaf5f413f7874782c1f3f73b0d0d2add5101.svg",
  "data-prep-kit": "https://lfai.landscape2.io/logos/18b12107ee709d7d22b0151639fd9aef5a21c509ed2fe4a7df1805b338a93bfb.svg",
  "adlik": "https://lfai.landscape2.io/logos/2429f1d87c5ac72ce81d1fe5c234792592c2b69d0bf1a10f4f59ba7d3d90bdc2.svg",
  "nnstreamer": "https://lfai.landscape2.io/logos/8b6e23e6740542e3d9c997aa20caf2e423e79e8387820987dd290fe1ae36dd4e.svg",
  "datashim": "https://lfai.landscape2.io/logos/8677bc8ec63a5556ac5159d7074c11a626285e4a3630c09c8addd7ac2a4995a5.svg",
  "monocole": "https://lfai.landscape2.io/logos/cd12ab207bd43dbab095895e1c2a64c16d293ebd320782fef44deeb66193d75d.svg",
  "substra": "https://lfai.landscape2.io/logos/be4eb38ae2967874613d3dbf1f9b56a6868534a2285c88eb5fb1bff2d8f1aed5.svg",
  "elastic deep learning (edl)": "https://lfai.landscape2.io/logos/1037d8a87ab736d0aecd02310d07807f085f05d63e898e17c5c6aa76a29639f5.svg",
  "rwkv": "https://lfai.landscape2.io/logos/94ee04d0704a7f82a767fa0f924f2008d540a9181eefe791e92a032440567cd2.svg",
  "soajs": "https://lfai.landscape2.io/logos/7f6137acbc1f597480e1f5dc460d2d855bdf6085d2ec3c4e8ebd294c9a5cc38e.svg",
  "open voice interoperability": "https://lfai.landscape2.io/logos/a1a47cc28291c728280a466feded0b76e6a05eda44d0918f616097da382144e7.svg",
  "open voice trustmark": "https://lfai.landscape2.io/logos/cd62b8987afe6e9129b26860e9f65d698c2d73dee71f535a4ca86b8a43f54514.svg",
  "recommenders": "https://lfai.landscape2.io/logos/a6c6d5ef4c460ba1d15f0e1a48f9a0e5a0a792483c34849d7d46afef7b8f1330.svg",
  "iree": "https://lfai.landscape2.io/logos/aac9593d73082b0f4c0a9b0774f72f221cd2c0d77a2a15474ca993426aa30731.svg",
  "flagai": "https://lfai.landscape2.io/logos/9767edb6cdb8ea390d6bf54d4840c3997c52aab803d039201f8b96ff3cb20f26.svg",
  "unity catalog": "https://lfai.landscape2.io/logos/1abcb4f52794f08a105060d30ccdd46707dcf5777c4b831787c9f8c927de9c64.svg",
  "lakesoul": "https://lfai.landscape2.io/logos/01f1f13f3e6501e261301ef2b8d0e2230333c405e3d6ba3d1bb57c68301f43d1.svg",
  "claimed": "https://lfai.landscape2.io/logos/44a369978e187c3393714aa1c6cabaddbf4688beeaa365bfb56f83ec347a8dd4.svg",
  "feathr": "https://lfai.landscape2.io/logos/87df0b3cdfc990ef23863ae0c95707349f1e3c2a3b125d6aa2f66c96be87cd94.svg",
  "dlrover": "https://lfai.landscape2.io/logos/86f4b0b2d75678e7bb26566d92d855f37b997b3fc102a04af3e10804d875331e.svg",
  "deeprec": "https://lfai.landscape2.io/logos/b31e69c85a0877784421d4b8551abeef556781568f3f5e6e3d15e7c7bb4c7758.svg",
  "openfl": "https://lfai.landscape2.io/logos/1169e18113f8d1ece90b01c2f37afa9f5bd1e461a08701a0fd4fb0d2288477ac.svg",
  "open platform for enterprise ai (opea)": "https://lfai.landscape2.io/logos/f704eb2902039d2ea1f6a008bd1b8e99cf6f395d87e3376341a8f37199a807b9.svg",
  "sapientml": "https://lfai.landscape2.io/logos/cc4002fce274cbdc0b98dda914a2892003df5691c0eb7375c282ae0f0048b1c9.svg",
  "ryoma": "https://lfai.landscape2.io/logos/ba226880e71da313f89c0ec99d9c795c22b23d1d88e87e49a04874110bf536fe.svg",
  "deepcausality": "https://lfai.landscape2.io/logos/6d48b7c47289adaf23a950f7054e7803b0c62a1fdee2f5c16c27c741a8dab099.svg",
  "opensharing": "https://lfai.landscape2.io/logos/659be7ceeafd818b9931e84aaa893077104cf31728e3c438a28b130fe8409c89.svg",
  "artigraph": "https://lfai.landscape2.io/logos/b702cc2533987d1cc97d880491e8540bd544fad5c95b6dadde90243b9ceb53a7.svg",
  "beyondml": "https://lfai.landscape2.io/logos/5c8c3704e1e0f2705cfc84a4c456a03be58b1fde4ace088cc9db6173c7f862d6.svg",
  "opendataology": "https://lfai.landscape2.io/logos/bd7b37b3955fac8019999a853fc2011eae970393d43760806da92acb39b157a8.svg",
  "open model initiative": "https://lfai.landscape2.io/logos/f741d1f797a687e6c816a1d4712d5e60f5e17d7ef638f2e5003772190f29f465.svg",
  "airseai": "https://lfai.landscape2.io/logos/1079b50aa8aaf94d8ee63fc05ec707d18392cf5f21896088ba577d9a3e708680.svg",
  "oaax": "https://lfai.landscape2.io/logos/71c9409b490ab18285d05d8e665347fdda05074f9ad878cb066a4ce5b829981a.svg",
  "vortex": "https://lfai.landscape2.io/logos/ba2cb4288164a38e372b754e1d1a326ad6ff823ef7d07fb86de40d11a0535a54.svg",
  "delta": "https://lfai.landscape2.io/logos/04c30a3795f62d279e107012b12a563f11e2cfb592f585c7233803f9f77d2fa0.svg",
  "shadernn": "https://lfai.landscape2.io/logos/687113f973a2d2d8fa6ce08861541d0e516e189125e271c79025ae4185330309.svg",
  "1chipml": "https://lfai.landscape2.io/logos/8dc206f2847909ee01c652e209f46bec725b73098ffb9475748387c6022e3425.svg",
  "acumos": "https://lfai.landscape2.io/logos/69045b81cfaab368258233604e03e765ccee9677926c1971841d781b59008b73.svg",
  "openbytes": "https://lfai.landscape2.io/logos/df68b2ff303a3af2629db863ab519e12494e52f35fa84884c2420b1f7f1cd5a2.svg"
};

function escape(value) {
  return String(value || "").replace(/[&<>"']/g, (character) => ({
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
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> Graduated</span>`;
  }
  if (m === "incubating") {
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-sky-500/15 text-sky-400 border border-sky-500/30"><span class="w-1.5 h-1.5 rounded-full bg-sky-500"></span> Incubating</span>`;
  }
  if (m === "sandbox") {
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span> Sandbox</span>`;
  }
  return "";
}

function projectAvatar(entry) {
  const titleKey = (entry.title || "").toLowerCase().trim();
  const logoUrl = projectLogos[titleKey];

  if (logoUrl) {
    return `
      <div class="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700/80 p-1.5 flex items-center justify-center shrink-0 shadow-xs">
        <img src="${escape(logoUrl)}" alt="${escape(entry.title)}" class="w-full h-full object-contain" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
        <span class="hidden font-bold text-emerald-400 text-xs">&gt;_</span>
      </div>
    `;
  }

  const tagLower = (entry.tag || "").toLowerCase();
  const catLower = (entry.category || "").toLowerCase();

  if (tagLower.includes("mcp") || catLower.includes("mcp")) {
    return `<div class="w-10 h-10 rounded-xl bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 font-mono text-[11px] font-bold flex items-center justify-center shrink-0 shadow-xs">⚡MCP</div>`;
  }

  if (tagLower.includes("sec") || catLower.includes("sec")) {
    return `<div class="w-10 h-10 rounded-xl bg-red-950/60 border border-red-800/40 text-red-400 flex items-center justify-center shrink-0 shadow-xs text-base">🛡️</div>`;
  }

  if (tagLower.includes("vector") || catLower.includes("vector")) {
    return `<div class="w-10 h-10 rounded-xl bg-purple-950/60 border border-purple-800/40 text-purple-400 flex items-center justify-center shrink-0 shadow-xs text-base">🗄️</div>`;
  }

  if (tagLower.includes("llm") || tagLower.includes("model") || catLower.includes("deep learning")) {
    return `<div class="w-10 h-10 rounded-xl bg-sky-950/60 border border-sky-800/40 text-sky-400 flex items-center justify-center shrink-0 shadow-xs text-base">🤖</div>`;
  }

  return `<div class="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700/80 text-emerald-400 font-mono text-xs font-bold flex items-center justify-center shrink-0 shadow-xs">&gt;_</div>`;
}

function card(entry) {
  const link = projectUrl(entry.link);
  const starsFormatted = formatStars(entry.stars);
  const matBadge = maturityBadge(entry.maturity);
  const avatar = projectAvatar(entry);
  
  return `
    <article class="p-6 rounded-2xl border border-slate-800 bg-slate-900/90 shadow-xs hover:border-emerald-500/50 hover:shadow-lg hover:shadow-emerald-950/20 transition-all flex flex-col justify-between group">
      <div>
        <div class="flex items-center justify-between gap-2 mb-3">
          <div class="flex items-center gap-1.5 flex-wrap">
            ${matBadge || `<span class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">${escape(entry.tag || "AI Tool")}</span>`}
            ${entry.subcategory ? `<span class="text-[11px] text-slate-400 font-mono">${escape(entry.subcategory)}</span>` : ""}
          </div>
          ${
            starsFormatted
              ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-xs font-bold text-amber-400 bg-amber-950/40 border border-amber-800/40 font-mono">
                  <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
                  ${starsFormatted}
                </span>`
              : ""
          }
        </div>

        <div class="flex items-center gap-3.5 mb-2.5">
          ${avatar}
          <div class="min-w-0 flex-1">
            <h3 class="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors leading-snug">
              <a href="${escape(link || '#')}" target="_blank" rel="noopener noreferrer">${escape(entry.title)}</a>
            </h3>
            ${entry.category ? `<span class="text-xs text-slate-400 font-medium block truncate mt-0.5">${escape(entry.category)}</span>` : ""}
          </div>
        </div>

        <p class="text-sm text-slate-300 mt-2 line-clamp-3 leading-relaxed">
          ${escape(entry.description || "Explore this open-source AI project, documentation, and installation instructions.")}
        </p>
      </div>

      <div class="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between text-xs">
        ${
          link
            ? `<a class="inline-flex items-center gap-1 font-bold text-emerald-400 hover:underline" href="${escape(link)}" target="_blank" rel="noopener noreferrer">Visit Website &nearr;</a>`
            : '<span class="text-slate-500">Website unavailable</span>'
        }
        ${
          entry.github
            ? `<a href="https://github.com/${escape(entry.github)}" target="_blank" rel="noopener noreferrer" class="text-slate-400 hover:text-white font-medium inline-flex items-center gap-1.5 transition-colors" title="GitHub Repository">
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
      <button type="button" data-tier="" class="tier-pill active px-3.5 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer bg-emerald-500 text-slate-950 border-emerald-400 shadow-sm shadow-emerald-500/20">
        All Projects (3,780+)
      </button>
      <button type="button" data-tier="graduated" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-400 transition-all cursor-pointer">
        ⭐ Graduated (13)
      </button>
      <button type="button" data-tier="incubating" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-sky-500/50 hover:text-sky-400 transition-all cursor-pointer">
        🚀 Incubating (22)
      </button>
      <button type="button" data-tier="sandbox" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-amber-500/50 hover:text-amber-400 transition-all cursor-pointer">
        🌱 Sandbox (22)
      </button>
      <button type="button" data-tier="security" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-red-500/50 hover:text-red-400 transition-all cursor-pointer">
        🛡️ AI Security (13)
      </button>
      <button type="button" data-tier="mcp" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-400 transition-all cursor-pointer">
        🔌 MCP Servers (3,300+)
      </button>
      <button type="button" data-tier="vectordb" class="tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-purple-500/50 hover:text-purple-400 transition-all cursor-pointer">
        🗄️ Vector DBs
      </button>
    </div>

    <!-- Search & Multi-Filter Control Strip -->
    <form class="kube-tool-controls adf-tool-controls flex flex-col md:flex-row gap-3 p-4 rounded-2xl border border-slate-800 bg-slate-900/90 mb-6 shadow-xs" role="search" aria-label="Find AI and LLM tools">
      <div class="flex-1 relative">
        <div class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        </div>
        <input id="tool-query" name="q" type="search" placeholder="Search 3,780+ tools: Milvus, ONNX, Docling, Flyte, Feast, Ludwig, MCP servers, security…" autocomplete="off" class="w-full pl-10 pr-10 py-2.5 rounded-xl border border-slate-700 bg-slate-800/90 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent">
        <button type="button" id="tool-query-clear" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-white hidden cursor-pointer" aria-label="Clear search">✕</button>
      </div>
      <div class="flex flex-wrap items-center gap-2.5">
        <label for="tool-category" class="sr-only">Category</label>
        <select id="tool-category" name="category" class="px-3.5 py-2.5 rounded-xl border border-slate-700 bg-slate-800 text-xs font-semibold text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500">
          <option value="">All Categories (${tags.length})</option>
          ${tags.map(tag => `<option value="${escape(tag)}">${escape(tag)}</option>`).join("")}
        </select>
        <label for="tool-sort" class="sr-only">Sort by</label>
        <select id="tool-sort" name="sort" class="px-3.5 py-2.5 rounded-xl border border-slate-700 bg-slate-800 text-xs font-semibold text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500">
          <option value="stars">⭐ Most GitHub Stars</option>
          <option value="az">A &rarr; Z Alphabetical</option>
          <option value="za">Z &rarr; A Reverse</option>
        </select>
        <button type="button" class="kube-button kube-button-outline kube-filter-reset adf-filter-reset text-xs py-2 px-3.5 border-slate-700 text-slate-300 hover:text-white hover:border-slate-600">Reset</button>
      </div>
    </form>

    <div class="my-4 flex items-center justify-between text-xs text-slate-400">
      <p class="kube-tool-count adf-tool-count font-mono font-medium" role="status" aria-live="polite" aria-atomic="true"></p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="tool-results"></div>
    
    <div class="mt-12 text-center">
      <button type="button" class="kube-button inline-flex items-center gap-2 px-6 py-3" id="tool-more">
        <span>Load More Projects</span>
        <span aria-hidden="true">&darr;</span>
      </button>
    </div>
  `;

  const form = root.querySelector("form");
  const input = root.querySelector("#tool-query");
  const clearQueryBtn = root.querySelector("#tool-query-clear");
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
    const paramCat = params.get("category");
    
    // Support category alias matching like ?category=Security or ?category=MCP
    if (paramCat) {
      if (tags.includes(paramCat)) {
        category.value = paramCat;
      } else if (paramCat.toLowerCase() === "security") {
        currentTier = "security";
      } else if (paramCat.toLowerCase() === "mcp") {
        currentTier = "mcp";
      }
    } else {
      category.value = "";
    }

    sort.value = ["stars", "za", "az"].includes(params.get("sort")) ? params.get("sort") : "stars";
    if (params.get("tier")) {
      currentTier = params.get("tier");
    }
    updateTierPills();
    updateClearQueryBtn();
    page = 1;
  }

  function updateClearQueryBtn() {
    if (clearQueryBtn) {
      if (input.value.trim().length > 0) {
        clearQueryBtn.classList.remove("hidden");
      } else {
        clearQueryBtn.classList.add("hidden");
      }
    }
  }

  function updateTierPills() {
    tierPills.forEach(pill => {
      const tier = pill.getAttribute("data-tier");
      if (tier === currentTier) {
        pill.className = "tier-pill active px-3.5 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer bg-emerald-500 text-slate-950 border-emerald-400 shadow-sm shadow-emerald-500/20";
      } else {
        pill.className = "tier-pill px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-800 bg-slate-900/90 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-400 transition-all cursor-pointer";
      }
    });
  }

  function updateUrl() {
    const url = new URL(location.href);
    for (const [key, value] of [
      ["q", input.value.trim()],
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
        if (currentTier === "mcp" && !String(entry.tag || "").toLowerCase().includes("mcp") && !String(entry.category || "").toLowerCase().includes("mcp")) return false;
        if (currentTier === "security" && !String(entry.tag || "").toLowerCase().includes("sec") && !String(entry.category || "").toLowerCase().includes("sec") && !haystack.includes("security")) return false;
        if (currentTier === "vectordb" && !String(entry.tag || "").toLowerCase().includes("vector") && !haystack.includes("vector database")) return false;

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
        <div class="col-span-full py-16 px-6 rounded-2xl border border-slate-800 bg-slate-900/60 text-center">
          <div class="inline-flex items-center justify-center w-12 h-12 rounded-full bg-slate-800 text-slate-400 mb-4 font-bold text-xl">
            🔍
          </div>
          <h3 class="text-xl font-bold text-white mb-2">No matching projects found</h3>
          <p class="text-slate-400 max-w-md mx-auto text-sm mb-6">No technologies matched your query. Try searching for Milvus, ONNX, Docling, Flyte, or reset your filters.</p>
          <button type="button" class="kube-button text-xs py-2 px-4" onclick="document.querySelector('.kube-filter-reset').click()">
            Reset All Filters
          </button>
        </div>
      `;
    }

    count.textContent = `Showing ${limit} of ${visible.length} projects (${entries.length.toLocaleString()} total indexed in LF AI & Data Landscape)`;
    more.hidden = limit >= visible.length;
    if (!more.hidden) {
      more.querySelector("span:first-child").textContent = `Load More Projects (${visible.length - limit} remaining)`;
    }
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

  // Global trigger for hero stat cards to filter tier
  window.filterToolsTier = function(tier) {
    currentTier = tier || "";
    updateTierPills();
    page = 1;
    updateUrl();
    render();
    const elem = document.querySelector("#kube-tools");
    if (elem) {
      elem.scrollIntoView({ behavior: "smooth" });
    }
  };

  form.addEventListener("submit", (event) => event.preventDefault());
  input.addEventListener("input", () => {
    updateClearQueryBtn();
    page = 1;
    updateUrl();
    render();
  });

  if (clearQueryBtn) {
    clearQueryBtn.addEventListener("click", () => {
      input.value = "";
      updateClearQueryBtn();
      page = 1;
      updateUrl();
      render();
      input.focus();
    });
  }

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
    updateClearQueryBtn();
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
