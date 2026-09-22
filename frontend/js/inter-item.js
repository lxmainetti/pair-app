import { $, itemList, countLabel, screen, heatmap, ranked, squarePairs, downloadCSV } from './common.js';

const count = $('#count');
const list = itemList($('#items'), {
  storageKey: 'pair:inter-item',
  placeholders: ['I see myself as someone who is talkative.', 'I see myself as someone who is reserved.'],
  onChange: () => { count.textContent = countLabel(list.values().length); },
});
count.textContent = countLabel(list.values().length);

screen({
  endpoint: '/api/inter-item',
  runLabel: 'Predict correlations',
  views: [['matrix', 'Heatmap'], ['ranked', 'Ranked pairs']],
  empty: {
    title: 'Run a prediction to see results.',
    text: 'PAIR predicts the correlation for every pair of items from their wording alone. Enter at least two items on the left.',
  },
  validate: () => (list.values().length < 2 ? 'Need at least 2 items.' : null),
  request() {
    const items = list.values();
    return { body: { items: items.map(x => x.text) }, snapshot: items, pairCount: items.length * (items.length - 1) / 2 };
  },
  prepare: (data, items) => ({ items, matrix: data.matrix, pairs: squarePairs(items, data.matrix) }),
  renderResults(d, st, rerender) {
    if (st.view === 'ranked') return ranked({ pairs: d.pairs, st, rerender });
    return heatmap({
      rows: d.items, cols: d.items, symmetric: true,
      value: (i, j) => (i === j ? null : d.matrix[i][j]),
    });
  },
  exportCSV(d) {
    const rows = [...d.pairs].sort((p, q) => Math.abs(q.r) - Math.abs(p.r)).map(p => [p.a.text, p.b.text, p.r]);
    downloadCSV('pair_inter-item.csv', [['item1', 'item2', 'r'], ...rows]);
  },
});
