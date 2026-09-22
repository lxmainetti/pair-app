import { $, h, itemList, countLabel, screen, heatmap, ranked, squarePairs, downloadCSV, fmt3, fmtDelta } from './common.js';

const count = $('#count');
const list = itemList($('#items'), {
  storageKey: 'pair:consistency',
  placeholders: ['I get stressed out easily.', 'I worry about things.'],
  onChange: () => { count.textContent = countLabel(list.values().length); },
});
count.textContent = countLabel(list.values().length);

// α if deleted comes from the backend (stats.alpha_if_deleted), one entry per item in order;
// alpha/delta are null when fewer than 2 items would remain.
function deletedView(d) {
  const rows = d.deleted.map(x => {
    const na = x.alpha == null;
    return h('div', { class: 'deleted-row' },
      h('b', {}, x.item.num),
      h('span', { class: 'text', title: x.item.text }, x.item.text),
      h('span', { class: 'num alpha' }, na ? '—' : fmt3(x.alpha)),
      h('span', { class: 'num delta' + (!na && x.delta > 0 ? ' is-up' : '') }, na ? '' : fmtDelta(x.delta)));
  });
  return h('div', { class: 'deleted' },
    h('div', { class: 'deleted-head' },
      h('span', {}, '#'), h('span', {}, 'Item'),
      h('span', {}, h('span', { class: 'nowrap-case' }, 'α'), ' if deleted'), h('span', {}, 'Δ')),
    ...rows,
    h('span', { class: 'footnote' }, 'Positive Δ means α rises when the item is removed.'));
}

screen({
  endpoint: '/api/consistency',
  runLabel: 'Compute α',
  views: [['matrix', 'Heatmap'], ['ranked', 'Ranked pairs'], ['deleted', 'α if deleted']],
  empty: {
    title: "Run a prediction to see Cronbach's α.",
    text: 'PAIR predicts every inter-item correlation and computes standardised α from them. Enter at least two items on the left.',
  },
  validate: () => (list.values().length < 2 ? 'Need at least 2 items.' : null),
  request() {
    const items = list.values();
    return { body: { items: items.map(x => x.text) }, snapshot: items, pairCount: items.length * (items.length - 1) / 2 };
  },
  prepare: (data, items) => ({
    items, matrix: data.matrix, alpha: data.alpha, interpretation: data.interpretation,
    pairs: squarePairs(items, data.matrix),
    deleted: data.alpha_if_deleted.map((x, k) => ({ item: items[k], alpha: x.alpha, delta: x.delta })),
  }),
  renderResults(d, st, rerender) {
    const hero = h('div', { class: 'hero-block' },
      h('div', { class: 'hero' },
        h('span', { class: 'hero-label' }, "Cronbach's α"),
        h('span', { class: 'hero-value' }, typeof d.alpha === 'number' ? fmt3(d.alpha) : '—'),
        h('span', { class: 'hero-interp' }, d.interpretation)),
      h('span', { class: 'hero-note' }, 'Correlations are reported as absolute values, as used in the α calculation to account for reverse-keyed items.'));
    let view;
    if (st.view === 'ranked') view = ranked({ pairs: d.pairs, abs: true, st, rerender });
    else if (st.view === 'deleted') view = deletedView(d);
    else view = heatmap({ rows: d.items, cols: d.items, abs: true, symmetric: true, value: (i, j) => (i === j ? null : d.matrix[i][j]) });
    return [hero, view];
  },
  exportCSV(d) {
    const pairs = [...d.pairs].sort((p, q) => Math.abs(q.r) - Math.abs(p.r)).map(p => [p.a.text, p.b.text, p.r]);
    const deleted = d.deleted.map(x => [x.item.text, x.alpha ?? '', x.delta ?? '']);
    downloadCSV('pair_consistency.csv', [
      ['item1', 'item2', 'r'], ...pairs,
      [],
      ['item', 'alpha_if_deleted', 'delta'], ...deleted,
    ]);
  },
});
