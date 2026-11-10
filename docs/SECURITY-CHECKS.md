# Dependency security checks

The Pages deployment and pull-request security workflow run Hex and npm audits. Workshop CI audits the installed Python environment after executing its notebooks. Findings are not suppressed or filtered by severity.

Run locally:

```bash
mix hex.audit
npm ci --ignore-scripts --prefix assets
npm audit --prefix assets
node scripts/check_vendor_katex.mjs
python -m pip_audit
```

Install `pip-audit` into the same isolated Python environment as the track requirements. To check a fresh resolution of all declared workshop dependencies, use:

```bash
python -m pip_audit \
  -r requirements/workshops/ml-foundations.txt \
  -r requirements/workshops/pytorch.txt \
  -r requirements/workshops/vision.txt \
  -r requirements/workshops/mlx-notebook.txt
```

The 2026-11-10 remediation updates Bandit to 1.14.1, Phoenix.HTML to 4.4.0, Plug.Crypto to 2.2.1, and its Thousand Island dependency to 1.6.0. Pillow is constrained to `>=13.0,<14` in image-processing and plotting environments, and notebook installation cells are regenerated from those requirements. KaTeX's vendored browser files and matching stylesheet use 0.18.2; npm locking, file comparisons, and security regression tests keep the audited package aligned with the shipped renderer. See `assets/vendor/README.md` for provenance.

After changing notebook requirements, regenerate the notebook assets and publish that commit. Update the notebook revision in runtime configuration so Colab does not keep importing an older installation cell. Existing downloaded notebooks and already-created environments need the same package updates or a fresh installation.

GitHub Dependabot alerts describe the default branch. They will remain open until the patched manifests are merged there and GitHub rescans them; dismissing an alert is not a dependency fix.
