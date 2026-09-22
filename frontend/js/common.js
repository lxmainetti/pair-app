/* PAIR — shared helpers for the three analysis screens (no framework, no build step). */

export const $ = (sel, root = document) => root.querySelector(sel);

/** Tiny element builder. Strings become text nodes, so user input is never parsed as HTML. */
export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.style.cssText = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
  }
  return el;
}

/* ── Icons (Lucide) ────────────────────────────────────────────────────── */
const ICONS = {
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  'arrow-right': '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
  'rotate-ccw': '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
};
export function icon(name, size = 14, stroke = 2) {
  const t = document.createElement('template');
  t.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
  return t.content.firstChild;
}

/* ── Number formatting ─────────────────────────────────────────────────── */
const MINUS = '−';
function signed(r, digits, stripZero) {
  let s = Math.abs(r).toFixed(digits);
  const neg = r < 0 && Number(s) !== 0;
  if (stripZero) s = s.replace(/^0/, '');
  return (neg ? MINUS : '') + s;
}
/** −.42 */
export const fmt2 = r => signed(r, 2, true);
/** −.421 */
export const fmt3 = r => signed(r, 3, true);
/** −0.421 */
export const fmtFull = r => signed(r, 3, false);
/** +.012 / −.008 */
export const fmtDelta = d => (d < 0 && Number(Math.abs(d).toFixed(3)) !== 0 ? MINUS : '+') + Math.abs(d).toFixed(3).replace(/^0/, '');

export const pad2 = n => String(n).padStart(2, '0');

export function heatBg(r) {
  const p = Math.round(Math.min(1, Math.abs(r)) * 100);
  return `color-mix(in srgb, var(${r >= 0 ? '--heat-pos' : '--heat-neg'}) ${p}%, var(--color-bg))`;
}

/* ── Storage (per-viewer convenience only) ─────────────────────────────── */
export function load(key) {
  try { return JSON.parse(sessionStorage.getItem(key)); } catch { return null; }
}
export function save(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

/* ── API ───────────────────────────────────────────────────────────────── */
export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export async function postJSON(url, body) {
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, 'Could not reach the PAIR server.');
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    const detail = data?.detail;
    throw new ApiError(res.status, typeof detail === 'string' ? detail : (res.statusText || 'Request failed.'));
  }
  if (data?.mock) showMockBadge(true);
  return data;
}

export function showMockBadge(on) {
  const badge = $('.mock-badge');
  if (badge) badge.hidden = !on;
}

export async function initMockBadge() {
  try {
    const s = await fetch('/api/status').then(r => r.json());
    showMockBadge(!!s.mock);
  } catch { /* leave hidden */ }
}

/* ── CSV export ────────────────────────────────────────────────────────── */
export function downloadCSV(filename, rows) {
  const cell = v => (typeof v === 'number' ? String(v) : `"${String(v).replace(/"/g, '""')}"`);
  const csv = '﻿' + rows.map(r => r.map(cell).join(',')).join('\r\n') + '\r\n';
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ── Item list editor ──────────────────────────────────────────────────────
   Structured list: one input per item. Enter moves to / adds the next row;
   pasting several lines splits them into rows. */
export function itemList(root, { prefix = '', storageKey, placeholders = [], fallbackPlaceholder = 'Add an item…', rows = 2, onChange }) {
  const stored = storageKey && load(storageKey);
  let texts = Array.isArray(stored) && stored.length ? stored.map(String) : Array(rows).fill('');
  const numOf = i => (prefix ? prefix + (i + 1) : pad2(i + 1));
  const listEl = h('div', { class: 'item-list' });

  function changed() {
    if (storageKey) save(storageKey, texts);
    onChange?.();
  }

  function render(focusIndex) {
    listEl.replaceChildren(...texts.map(row));
    if (focusIndex != null) listEl.querySelectorAll('input')[focusIndex]?.focus();
  }

  function row(text, i) {
    const input = h('input', {
      class: 'item-input', type: 'text', autocomplete: 'off',
      placeholder: placeholders[i] ?? fallbackPlaceholder,
      'aria-label': `Item ${numOf(i)}`,
    });
    input.value = text;
    input.addEventListener('input', () => { texts[i] = input.value; changed(); });
    input.addEventListener('keydown', e => {
      if (e.key !== 'Enter' || e.isComposing) return;
      e.preventDefault();
      if (i === texts.length - 1) { texts.push(''); changed(); }
      render(i + 1);
    });
    input.addEventListener('paste', e => {
      const lines = (e.clipboardData?.getData('text') ?? '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
      if (lines.length < 2) return;
      e.preventDefault();
      const before = input.value.slice(0, input.selectionStart);
      const after = input.value.slice(input.selectionEnd);
      lines[0] = before + lines[0];
      lines[lines.length - 1] += after;
      texts.splice(i, 1, ...lines);
      changed();
      render(i + lines.length - 1);
    });
    const remove = h('button', {
      class: 'btn btn-icon item-remove', type: 'button',
      title: 'Remove item', 'aria-label': `Remove item ${numOf(i)}`,
      onclick: () => {
        if (texts.length === 1) texts = [''];
        else texts.splice(i, 1);
        changed();
        render();
      },
    }, icon('x', 14));
    return h('div', { class: 'item-row' }, h('span', { class: 'item-num' }, numOf(i)), input, remove);
  }

  const add = h('button', {
    class: 'btn btn-ghost', type: 'button',
    onclick: () => { texts.push(''); changed(); render(texts.length - 1); },
  }, icon('plus', 14, 2.2), 'Add item');

  root.append(h('div', { class: 'item-editor' }, listEl, add));
  render();

  return {
    /** Non-empty items with their row numbers, in order (matches the backend's filtering). */
    values: () => texts.map((t, i) => ({ num: numOf(i), text: t.trim() })).filter(x => x.text),
  };
}

export const countLabel = n => `${n} item${n === 1 ? '' : 's'}`;

/* ── Screen controller ─────────────────────────────────────────────────────
   Owns phase (empty | loading | results | error | invalid), the view switch,
   the run button, status line and export button. Each page supplies the
   request and the results renderer. */
export function screen(cfg) {
  const st = { phase: 'empty', view: cfg.views[0][0], data: null, error: null, showAll: false };
  const els = {
    run: $('#run'), status: $('#status'), exportBtn: $('#export'),
    views: $('#view-switch'), body: $('#results-body'),
  };
  let timer = null;

  els.run.addEventListener('click', run);
  els.exportBtn.addEventListener('click', () => st.phase === 'results' && cfg.exportCSV(st.data, st));

  function setStatus(text, kind = '') {
    els.status.textContent = text;
    els.status.className = 'status' + (kind ? ' is-' + kind : '');
  }

  function renderViews() {
    els.views.hidden = st.phase !== 'results';
    els.views.replaceChildren(...cfg.views.map(([key, label]) => h('button', {
      type: 'button', 'aria-pressed': String(st.view === key),
      onclick: () => { st.view = key; st.showAll = false; render(); },
    }, label)));
  }

  function render() {
    const loading = st.phase === 'loading';
    els.run.disabled = loading;
    els.run.firstElementChild.textContent = loading ? 'Running…' : cfg.runLabel;
    els.exportBtn.disabled = st.phase !== 'results';
    renderViews();
    if (st.phase !== 'loading') clearInterval(timer);

    if (st.phase === 'results') {
      els.body.replaceChildren(...[].concat(cfg.renderResults(st.data, st, render)));
    } else if (loading) {
      const elapsed = h('span', {}, `Scoring ${st.pairCount} pairs`);
      els.body.replaceChildren(h('div', { class: 'state-loading', role: 'status' },
        h('h3', {}, `Predicting ${st.pairCount} pairs`),
        h('div', { class: 'progress' }, h('span')),
        elapsed));
      const t0 = Date.now();
      clearInterval(timer);
      timer = setInterval(() => {
        elapsed.textContent = `Scoring ${st.pairCount} pairs · ${Math.round((Date.now() - t0) / 1000)} s elapsed`;
      }, 1000);
    } else if (st.phase === 'error') {
      const e = st.error, unavailable = e.status === 503;
      els.body.replaceChildren(h('div', { class: 'state-error', role: 'alert' },
        h('span', { class: 'error-code' }, e.status ? `Error ${e.status}` : 'Network error'),
        h('h4', {}, unavailable ? 'Inference service unavailable' : 'Request failed'),
        h('p', {}, unavailable
          ? 'The PAIR model server did not respond. Your items are kept; try again in a moment.'
          : `${e.message} Your items are kept; try again in a moment.`),
        h('button', { class: 'btn btn-secondary', type: 'button', onclick: run }, icon('rotate-ccw', 14), 'Retry')));
    } else {
      els.body.replaceChildren(h('div', { class: 'state-empty' },
        h('h3', {}, cfg.empty.title), h('p', {}, cfg.empty.text)));
    }
  }

  async function run() {
    if (st.phase === 'loading') return;
    const invalid = cfg.validate();
    if (invalid) {
      st.phase = 'invalid'; st.data = null;
      setStatus(invalid, 'error');
      return render();
    }
    const { body, snapshot, pairCount } = cfg.request();
    st.phase = 'loading'; st.pairCount = pairCount;
    setStatus('Running…', 'muted');
    render();
    try {
      const data = await postJSON(cfg.endpoint, body);
      st.data = cfg.prepare(data, snapshot);
      st.phase = 'results'; st.showAll = false;
      setStatus(`${pairCount} pairs predicted.`);
    } catch (e) {
      if (e.status === 400 || e.status === 422) {
        st.phase = 'invalid'; st.data = null;
        setStatus(e.message, 'error');
      } else {
        st.phase = 'error'; st.error = e;
        setStatus(e.status === 503 ? 'Error: inference service unavailable (503).' : `Error: ${e.message}`, 'error');
      }
    }
    render();
  }

  initMockBadge();
  render();
  return { render, state: st };
}

/* ── Heatmap ───────────────────────────────────────────────────────────────
   rows/cols: [{num, text}]; value(i, j) → signed r, or null for the diagonal.
   abs: shade and label |r| (one-directional). symmetric: hovering (i,j) also
   outlines (j,i). */
export function heatmap({ rows, cols, value, cell = 52, abs = false, symmetric = false, corner = '' }) {
  const hoverLine = h('span', { class: 'hover-line' }, 'Hover a cell to read the pair.');
  const grid = h('div', {
    class: 'hm-grid',
    style: `--cell:${cell}px;grid-template-columns:var(--label-w) repeat(${cols.length}, ${cell}px)`,
  });
  grid.append(h('span', { class: 'hm-corner' }, corner));
  cols.forEach(c => grid.append(h('span', { class: 'hm-col' }, c.num)));

  const cells = rows.map((row, i) => {
    grid.append(h('span', { class: 'hm-label', title: row.text }, h('b', {}, row.num), h('span', {}, row.text)));
    return cols.map((_, j) => {
      const r = value(i, j);
      if (r == null) return grid.appendChild(h('span', { class: 'hm-cell is-diag' }));
      const shown = abs ? Math.abs(r) : r;
      const el = h('span', {
        class: 'hm-cell' + (Math.abs(r) > 0.75 ? ' is-dark' : ''),
        style: `background:${heatBg(shown)}`, 'data-i': i, 'data-j': j,
      }, fmt2(shown));
      return grid.appendChild(el);
    });
  });

  let lit = [];
  grid.addEventListener('mouseover', e => {
    const el = e.target.closest('.hm-cell[data-i]');
    if (!el) return;
    const i = +el.dataset.i, j = +el.dataset.j;
    lit.forEach(c => c.classList.remove('is-on'));
    lit = [el];
    if (symmetric && cells[j]?.[i]) lit.push(cells[j][i]);
    lit.forEach(c => c.classList.add('is-on'));
    hoverLine.textContent = `${rows[i].num} “${rows[i].text}” × ${cols[j].num} “${cols[j].text}” — r = ${fmtFull(value(i, j))}`;
  });

  const legend = abs
    ? h('div', { class: 'legend' }, h('span', {}, '0'), h('div', { class: 'legend-bar is-abs' }), h('span', {}, '1'))
    : h('div', { class: 'legend' }, h('span', {}, '−1'), h('div', { class: 'legend-bar is-signed' }), h('span', {}, '+1'));

  return h('div', { class: 'heatmap' }, hoverLine, h('div', { class: 'hm-scroll' }, grid), legend);
}

/* ── Ranked pairs ──────────────────────────────────────────────────────────
   pairs: [{a:{num,text}, b:{num,text}, r}]. abs: one-directional |r| bars. */
export function ranked({ pairs, abs = false, st, rerender, limit = 15 }) {
  const sorted = [...pairs].sort((p, q) => Math.abs(q.r) - Math.abs(p.r));
  const shown = st.showAll ? sorted : sorted.slice(0, limit);
  const pct = r => Math.round(Math.min(1, Math.abs(r)) * 100) + '%';

  const head = h('div', { class: 'ranked-head' },
    h('span', {}, 'Pair'),
    abs
      ? h('span', { class: 'axis-labels' }, h('span', {}, '0'), h('span', {}, '1'))
      : h('span', { class: 'axis-labels' }, h('span', {}, '−'), h('span', {}, '0'), h('span', {}, '+')),
    h('span', { class: 'r-head' }, 'r'));

  const rows = shown.map(p => {
    const bar = abs
      ? h('div', { class: 'bar-abs' }, h('div', { class: 'bar-pos', style: `width:${pct(p.r)}` }))
      : h('div', { class: 'bar-signed' },
          h('div', { class: 'neg-side' }, h('div', { class: 'bar-neg', style: `width:${p.r < 0 ? pct(p.r) : '0%'}` })),
          h('div', { class: 'pos-side' }, h('div', { class: 'bar-pos', style: `width:${p.r > 0 ? pct(p.r) : '0%'}` })));
    const r = abs ? Math.abs(p.r) : p.r;
    return h('div', { class: 'ranked-row' },
      h('div', { class: 'pair-text' },
        h('span', { title: p.a.text }, h('b', {}, p.a.num), p.a.text),
        h('span', { title: p.b.text }, h('b', {}, p.b.num), p.b.text)),
      bar,
      h('span', { class: 'r-val ' + (r >= 0 ? 'pos' : 'neg') }, fmt2(r)));
  });

  const wrap = h('div', { class: 'ranked' }, head, ...rows);
  if (!sorted.length) wrap.append(h('div', { class: 'ranked-none' }, 'No pairs to show.'));
  if (sorted.length > limit) {
    wrap.append(h('button', {
      class: 'btn btn-ghost', type: 'button',
      onclick: () => { st.showAll = !st.showAll; rerender(); },
    }, st.showAll ? `Show top ${limit}` : `Show all ${sorted.length} pairs`));
  }
  return wrap;
}

/** Unique pairs (i<j) of a square matrix over `items`. */
export function squarePairs(items, matrix) {
  const out = [];
  for (let i = 0; i < items.length; i++)
    for (let j = i + 1; j < items.length; j++)
      out.push({ a: items[i], b: items[j], r: matrix[i][j] });
  return out;
}
