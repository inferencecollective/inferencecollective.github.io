// Check that audited npm inputs match the browser files actually shipped.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const root = new URL('../', import.meta.url);
const read = path => fs.readFileSync(new URL(path, root), 'utf8');
const upstream = read('assets/node_modules/katex/dist/katex.mjs');
assert.equal(read('assets/vendor/katex.js'), upstream);
const auto = read('assets/node_modules/katex/dist/contrib/auto-render.mjs')
  .replace("from '../katex.mjs'", "from './katex.js'");
assert.equal(read('assets/vendor/auto-render.js'), auto);
const {version} = JSON.parse(read('assets/node_modules/katex/package.json'));
assert.ok(read('assets/css/app.css').includes(`katex@${version}/dist/katex.min.css`));
const {default: katex} = await import('data:text/javascript;base64,' + Buffer.from(upstream).toString('base64'));
assert.ok(katex.renderToString(String.raw`\frac{1}{2}`).includes('katex'));
// GHSA-238p-pmpm-9mq7: inherited options must not enable trusted links.
const inheritedTrust = Object.create({trust: true});
const output = katex.renderToString(String.raw`\href{javascript:alert(1)}{click}`, inheritedTrust);
assert.ok(!/href\s*=\s*["']javascript:/i.test(output));
// GHSA-cg87-wmx4-v546: malformed data-attribute names must be rejected.
assert.throws(() => katex.renderToString(String.raw`\htmlData{foo" onclick="alert(1)}{x}`, {trust: true, strict: "ignore"}));
console.log(`Verified KaTeX ${version}: vendor files match the audited package; security regression checks passed.`);
