'use strict';

const { figure, color } = require('../../../_shared/figkit.js');

const left = [
  ['identity', 'digests equal', 'unchanged'],
  ['pseudocode', '272 to 272', 'unchanged'],
  ['assembly', '161 to 161', 'unchanged'],
  ['comments', '0 to 0', 'unchanged'],
  ['calls', '1 to 1', 'unchanged'],
  ['references', '3 to 3', 'unchanged'],
  ['strings_names', '1 to 1', 'unchanged'],
  ['cfg', '3 to 3', 'unchanged'],
];
const right = [
  ['identity', 'digests equal', 'unchanged'],
  ['pseudocode', '300 to 2904', 'changed'],
  ['assembly', '336 to 1866', 'changed'],
  ['comments', '0 to 0', 'unchanged'],
  ['calls', '1 to 1', 'unchanged'],
  ['references', '4 to 13', 'changed'],
  ['strings_names', '2 to 11', 'changed'],
  ['cfg', '5 to 16', 'changed'],
];

function column(f, x, kicker, head, hue, rows) {
  f.kicker(x, 18, kicker);
  f.box({ x, y: 26, w: 290, h: 26, hue, title: head, align: 'left', titleSize: 11.5, pad: 10 });
  rows.forEach(([dimension, counts, status], index) => {
    const y = 52 + index * 24;
    const changed = status === 'changed';
    f.rect(x, y, 290, 24, { fill: changed ? color('amber-fill') : color('paper'), stroke: changed ? color('amber-ink') : color('panel-edge') });
    f.text(x + 10, y + 16, dimension, { size: 11 });
    f.text(x + 120, y + 16, counts, { size: 11, hue: 'ink-soft' });
    f.text(x + 215, y + 16, status, { size: 11, weight: changed ? 700 : undefined, hue: changed ? 'amber' : 'ink-soft' });
  });
}

module.exports = figure('fig-5-6', {
  height: 300,
  title: 'compare_functions on two functions across the -O1 and -O2 builds',
  desc: 'Two tables of the eight comparison dimensions. Left, _percent_change: identity, pseudocode 272 to 272 code points, assembly 161 to 161, comments 0 to 0, calls 1 to 1, references 3 to 3, strings_names 1 to 1 and cfg 3 to 3 are all unchanged, so the status is unchanged. Right, _total: identity, comments and calls are unchanged, while pseudocode grows from 300 to 2904 code points, assembly from 336 to 1866, references from 4 to 13, strings_names from 2 to 11 and cfg from 5 to 16 basic blocks, so five dimensions are changed and the status is changed. A note repeats the text limitations: both text dimensions compare exact provider text without address or generated-name normalization, and assembly also says that relocation normalization is unavailable.',
}, f => {
  column(f, 20, 'A · _percent_change, -O1 against -O2', 'status unchanged, 8 of 8 dimensions', 'grey', left);
  column(f, 330, 'B · _total, -O1 against -O2', 'status changed, 5 of 8 dimensions', 'amber', right);
  f.text(20, 268, 'Counts are code points for the two text dimensions and items for the others.', { size: 11, serif: true, hue: 'ink-soft' });
  f.text(20, 284, 'Text rows carry an exact-text limitation, and assembly adds: relocation normalization is unavailable.', { size: 11, serif: true, hue: 'ink-soft' });
});
