'use strict';

const { figure, color } = require('../../../_shared/figkit.js');

const columns = [
  { title: 'ghidra', sub: ['29 operations', 'available'], hue: 'violet' },
  { title: 'hopper', sub: ['37 operations', 'unavailable'], hue: 'grey' },
  { title: 'ida', sub: ['13 operations', 'unavailable'], hue: 'grey' },
  { title: 'target-free', sub: ['19 operations', 'no binding'], hue: 'blue' },
];
const rows = [
  ['launches_process', ['29 of 29', '37 of 37', '0', '12 of 19']],
  ['may_write_filesystem', ['29 of 29', '6 of 37', '0', '6 of 19']],
  ['may_show_ui', ['0', '37 of 37', '4 of 13', '2 of 19']],
  ['may_access_network', ['0', '0', '13 of 13', '2 of 19']],
  ['mutates_artifact', ['3 of 29', '6 of 37', '0', '0']],
  ['changes_permissions', ['0', '0', '0', '0']],
  ['requires_root', ['0', '0', '0', '0']],
];
const colX = index => 172 + index * 112;

module.exports = figure('fig-2-4', {
  height: 330,
  title: 'Declared effects by provider in the capture',
  desc: 'A table with four columns, ghidra with 29 operations available, hopper with 37 operations unavailable, ida with 13 operations unavailable, and 19 target-free operations with no provider binding, and seven rows, one per effect flag. launches_process is true for 29 of 29 ghidra operations, 37 of 37 hopper operations, no ida operation, and 12 of 19 target-free operations. may_write_filesystem is true for 29 of 29, 6 of 37, 0, and 6 of 19. may_show_ui is true for 0, 37 of 37, 4 of 13, and 2 of 19. may_access_network is true for 0, 0, 13 of 13, and 2 of 19. mutates_artifact is true for 3 of 29, 6 of 37, 0, and 0. changes_permissions and requires_root are false everywhere.',
}, f => {
  f.kicker(20, 18, 'effect flag');
  columns.forEach((column, index) => f.box({ x: colX(index), y: 26, w: 106, h: 56, titleSize: 12, pad: 6, ...column }));
  rows.forEach(([flag, values], row) => {
    const y = 92 + row * 32;
    f.rect(20, y, 146, 32, { fill: color('paper'), stroke: color('panel-edge') });
    f.text(26, y + 20, flag, { size: 11 });
    values.forEach((value, index) => {
      const zero = value === '0';
      f.rect(colX(index), y, 106, 32, { fill: zero ? color('paper') : color(`${columns[index].hue}-fill`), stroke: color('panel-edge') });
      f.text(colX(index) + 53, y + 20, value, { size: 12, anchor: 'middle', hue: zero ? 'ink-mute' : undefined });
    });
  });
});
