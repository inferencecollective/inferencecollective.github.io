# Browser vendor sources

KaTeX is tracked in `assets/package.json` and `assets/package-lock.json` so dependency auditing covers the renderer shipped to the browser. The vendored files are from KaTeX **0.18.2**, downloaded from the npm release tarball and checked against its SHA-512 integrity in the lockfile:

- `katex.js`: unchanged `dist/katex.mjs`.
- `auto-render.js`: `dist/contrib/auto-render.mjs`, with only its `../katex.mjs` import rewritten to `./katex.js`.
- `KATEX-LICENSE.txt`: upstream license.
- `assets/css/app.css`: references the matching versioned KaTeX stylesheet and fonts.

After `npm ci --ignore-scripts --prefix assets`, run `npm audit --prefix assets` and `node scripts/check_vendor_katex.mjs`. The latter compares vendor files with the locked package and exercises security regressions for inherited trust options and malformed HTML data attributes. Repeat these checks when upgrading, and update the stylesheet version to match.

Other existing vendor files are not represented by this npm manifest. Pele's upstream security page reported no published advisories at the time of this review; this is not a blanket security assessment of that renderer or the application.
