export function initWorkshops() {
  for (const root of document.querySelectorAll('[data-workshop]')) {
    if (root.dataset.workshopReady) continue;
    root.dataset.workshopReady = 'true';
    const key = `ic-workshop-v1:${root.dataset.workshop}`;
    const inputs = [...root.querySelectorAll('[data-workshop-step]')];
    let saved = [];
    try { const value = JSON.parse(localStorage.getItem(key)); if (Array.isArray(value)) saved = value; } catch {}
    const update = () => {
      const done = inputs.filter(input => input.checked).map(input => input.dataset.workshopStep);
      root.querySelector('progress').value = done.length;
      root.querySelector('[data-workshop-count]').textContent = `${done.length} of ${inputs.length} labs completed`;
      try { localStorage.setItem(key, JSON.stringify(done)); } catch {}
    };
    for (const input of inputs) {
      input.checked = saved.includes(input.dataset.workshopStep);
      input.addEventListener('change', update);
    }
    root.querySelectorAll('[data-workshop-control], [data-workshop-progress]').forEach(el => el.hidden = false);
    update();
  }
}
