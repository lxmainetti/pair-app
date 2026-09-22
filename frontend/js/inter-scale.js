import { $, h, itemList, countLabel, screen, heatmap, ranked, squarePairs, downloadCSV, fmt3 } from './common.js';

const lists = {};
for (const key of ['a', 'b']) {
  const K = key.toUpperCase();
  const count = $(`#count-${key}`);
  const update = () => { count.textContent = countLabel(lists[key].values().length); };
  lists[key] = itemList($(`#items-${key}`), {
    prefix: K, rows: 1, storageKey: `pair:inter-scale:${key}`,
    placeholders: Array.from({ length: 100 }, (_, i) => `Item ${K}${i + 1}`),
    onChange: update,
  });
  update();
}

const SCOPES = [['cross', 'A × B'], ['a', 'Within A'], ['b', 'Within B']];
let scope = 'cross';

function scopeData(d) {
  if (scope === 'cross') {
    return {
      rows: d.a, cols: d.b, corner: 'A \\ B', symmetric: false,
      value: (i, j) => d.cross[i][j],
      pairs: d.a.flatMap((x, i) => d.b.map((y, j) => ({ a: x, b: y, r: d.cross[i][j] }))),
    };
  }
  const items = d[scope], m = scope === 'a' ? d.matrixA : d.matrixB;
  return {
    rows: items, cols: items, corner: '', symmetric: true,
    value: (i, j) => (i === j ? null : m[i][j]),
    pairs: squarePairs(items, m),
  };
}

screen({
  endpoint: '/api/inter-scale',
  runLabel: 'Compute inter-scale r',
  views: [['matrix', 'Heatmap'], ['ranked', 'Ranked pairs']],
  empty: {
    title: 'Enter two scales and run.',
    text: 'PAIR predicts every item correlation, then derives the scale-score correlation r(A,B) from them. Each scale needs at least one item.',
  },
  validate: () => (!lists.a.values().length || !lists.b.values().length ? 'Each scale needs at least 1 item.' : null),
  request() {
    const a = lists.a.values(), b = lists.b.values();
    const n = new Set([...a, ...b].map(x => x.text)).size;   // the backend predicts all unique pairs
    return {
      body: { items_a: a.map(x => x.text), items_b: b.map(x => x.text) },
      snapshot: { a, b }, pairCount: n * (n - 1) / 2,
    };
  },
  prepare: (data, { a, b }) => {
    scope = 'cross';
    return { a, b, rab: data.r_ab, cross: data.cross_matrix, matrixA: data.matrix_a, matrixB: data.matrix_b };
  },
  renderResults(d, st, rerender) {
    const sd = scopeData(d);
    const hero = h('div', { class: 'hero' },
      h('span', { class: 'hero-label' }, 'r(A,B)'),
      h('span', { class: 'hero-value ' + (d.rab >= 0 ? 'pos' : 'neg') }, fmt3(d.rab)),
      h('div', { class: 'scope-tabs', role: 'group', 'aria-label': 'Scope' }, SCOPES.map(([k, label]) => h('button', {
        type: 'button', 'aria-pressed': String(scope === k),
        onclick: () => { scope = k; st.showAll = false; rerender(); },
      }, label))));
    const view = st.view === 'ranked'
      ? ranked({ pairs: sd.pairs, st, rerender })
      : heatmap({ ...sd, cell: 64 });
    return [hero, view];
  },
  exportCSV(d) {
    const rows = scopeData(d).pairs.sort((p, q) => Math.abs(q.r) - Math.abs(p.r)).map(p => [p.a.text, p.b.text, p.r]);
    const name = { cross: 'AxB', a: 'within-A', b: 'within-B' }[scope];
    downloadCSV(`pair_inter-scale_${name}.csv`, [['item1', 'item2', 'r'], ...rows]);
  },
});
