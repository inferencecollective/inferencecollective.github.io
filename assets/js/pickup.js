// Reading progress pickup bar matching kubedaily.com
export function initPickup() {
  const msg = document.querySelector("#pick-up");

  if (!msg) return;

  const btn = msg?.querySelector("a");
  const closeBtn = msg?.querySelector("button");

  let pickupFrom = null;
  try {
    pickupFrom = JSON.parse(localStorage.getItem("pick-up"));
  } catch {}

  const urlParams = new URLSearchParams(window.location.search);

  function close() {
    if (msg) msg.classList.add("hidden");
  }

  if (closeBtn) closeBtn.onclick = close;

  function updateMessage() {
    if (pickupFrom && !window.location.pathname.includes("/labs/") && !window.location.pathname.includes("/blog/")) {
      if (msg) msg.classList.remove("hidden");
      if (btn) btn.href = pickupFrom.from + "?pickup=true";
    } else if (pickupFrom && window.location.pathname === pickupFrom.from && urlParams.get("pickup") === "true") {
      window.scrollTo(0, pickupFrom.scroll);
    } else if (window.location.pathname.includes("/labs/") || window.location.pathname.includes("/blog/")) {
      try {
        localStorage.setItem("pick-up", JSON.stringify({ from: window.location.pathname, scroll: window.scrollY }));
      } catch {}
    }
  }

  updateMessage();

  if (window.location.pathname.includes("/labs/") || window.location.pathname.includes("/blog/")) {
    window.addEventListener("scroll", () => {
      try {
        localStorage.setItem("pick-up", JSON.stringify({ from: window.location.pathname, scroll: window.scrollY }));
      } catch {}
    }, { passive: true });
  }

}
