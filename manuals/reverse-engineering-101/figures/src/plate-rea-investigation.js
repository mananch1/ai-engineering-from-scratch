'use strict';

const { figure, color } = require('../../../_shared/figkit.js');

const source = [
  'you wrote · ledger.c',
  'int percent_change(',
  '  int before, int after) {',
  '  if (before == 0)',
  '    return 0;',
  '  return (after - before)',
  '    * 100 / before;',
  '}',
];
const listing = [
  'clang -O1 made · 24 bytes',
  '500  cbz w0, 0x100000514',
  '504  sub w8, w1, w0',
  '508  mov w9, #0x64',
  '50c  mul w8, w8, w9',
  '510  sdiv w0, w8, w0',
  '514  ret',
];
const pseudocode = [
  'Ghidra decompiled',
  'ulong _percent_change(',
  '  ulong param_1,int param_2)',
  'iVar2 = (int)param_1;',
  'if (iVar2 != 0) {',
  '  uVar1 = 0;',
  '  if (iVar2 != 0) {',
  '  uVar1 = ((param_2 - iVar2)',
  '    * 100) / iVar2;',
  '  param_1 = (ulong)uVar1;',
  'return param_1;',
];

const column = (f, x, w, hue, lines) => {
  const h = lines.length * 15 + 14;
  f.rect(x, 166, w, h, { fill: color('paper'), stroke: color(`${hue}-ink`) });
  lines.forEach((line, index) => f.text(x + 6, 184 + index * 15, line, { size: 11, hue: index === 0 ? 'ink-soft' : undefined }));
  return 166 + h;
};

module.exports = figure('plate-rea-investigation', {
  height: 472,
  title: 'One function, four views',
  desc: 'The recorded run of rea function on ledger-arm64-O1 for _percent_change with the Ghidra provider. At the top, you call rea 6.5.0, which hands the question to the BinarySession bound to ghidra, which calls the GhidraClient over bridge.sock, which launches Ghidra 12.1.4 as a headless JVM to import and analyze the artifact ledger-arm64-O1, a mach-o arm64 executable of 33872 bytes. Three columns show the same function: the C you wrote in ledger.c, the six instructions clang -O1 made from cbz w0 at 0x100000500 to ret at 0x100000514, and the pseudocode Ghidra decompiled with a second zero check the source never had. The Evidence record ev_172d39ff at the bottom names the operation analyze_function, the provider ghidra 12.1.4, the profile digest, confidence observed, the authority of a static read, ten limitations, and the subject sha256.',
}, f => {
  f.kicker(20, 18, 'four views');
  f.text(620, 18, 'rea function ledger-arm64-O1 _percent_change --provider ghidra', { size: 11, anchor: 'end', hue: 'ink-soft' });

  const you = f.box({ x: 20, y: 30, w: 70, h: 40, hue: 'grey', title: 'you', sub: 'ledger.c', titleSize: 11.5, pad: 6 });
  const rea = f.box({ x: 150, y: 30, w: 100, h: 40, hue: 'blue', title: 'rea 6.5.0', sub: 'the CLI', titleSize: 11.5 });
  const session = f.box({ x: 310, y: 30, w: 115, h: 40, hue: 'blue', title: 'BinarySession', sub: 'ghidra bound', titleSize: 11.5, pad: 8 });
  const client = f.box({ x: 500, y: 30, w: 120, h: 40, hue: 'violet', title: 'GhidraClient', sub: 'bridge.sock', titleSize: 11.5, pad: 8 });
  f.arrow(you.right(), [rea.x - 1, you.cy], { style: 'call', label: 'function' });
  f.arrow(rea.right(), [session.x - 1, rea.cy], { style: 'call', label: 'execute' });
  f.arrow(session.right(), [client.x - 1, session.cy], { style: 'call', label: 'call' });

  const jvm = f.box({ x: 500, y: 92, w: 120, h: 44, hue: 'violet', title: 'Ghidra 12.1.4', sub: 'import, analyze', titleSize: 11.5, pad: 8 });
  f.arrow(client.bottom(), [client.cx, jvm.y - 1], { style: 'effect', label: 'launch' });
  const art = f.box({ x: 220, y: 92, w: 185, h: 44, hue: 'green', title: 'ledger-arm64-O1', sub: 'mach-o arm64, 33872 bytes', titleSize: 11.5, pad: 8 });
  f.arrow(session.bottom(), [session.cx, art.y - 1], { style: 'call', label: 'read, hash' });

  f.arrow(you.bottom(), [you.cx, 165], { style: 'call', label: 'wrote' });
  f.arrow(jvm.bottom(), [jvm.cx, 165], { style: 'reply', label: 'decompile' });
  column(f, 20, 190, 'green', source);
  column(f, 220, 185, 'green', listing);
  const bottom = column(f, 415, 205, 'violet', pseudocode);

  f.arrow([517, bottom], [517, 365], { style: 'write', label: 'recorded' });
  f.box({
    x: 20, y: 366, w: 600, h: 92, hue: 'teal', titleSize: 12,
    title: 'Evidence ev_172d39ff for _percent_change',
    sub: [
      'operation analyze_function · provider ghidra 12.1.4',
      'confidence observed · 10 limitations',
      'subject ledger-arm64-O1 · mach-o arm64',
    ],
  });
});
