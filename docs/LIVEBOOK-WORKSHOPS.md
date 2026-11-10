# Livebook workshop integration

The repository supports two deployment modes. The static GitHub Pages site provides downloadable `.livemd` notebooks and the official Livebook launcher. The running Phoenix application additionally provides live rooms, participant presence, progress, help requests, and shared notebook embeds. GitHub Pages cannot run the Phoenix websocket service or notebook runtimes.

## Run the complete integration locally

Install Livebook using its [official instructions](https://livebook.dev/#install), then start it with authentication enabled (the default):

```bash
LIVEBOOK_PORT=8080 LIVEBOOK_IP=127.0.0.1 LIVEBOOK_WITHIN_IFRAME=true livebook server
```

Open the token-authenticated URL printed by Livebook. In another terminal, start this repository's Phoenix app:

```bash
WORKSHOP_LIVE_URL=http://localhost:4000 \
LIVEBOOK_URL=http://localhost:8080 \
WORKSHOP_NOTEBOOK_BASE_URL=http://localhost:4000/aidatafoundation/notebooks \
mix phx.server
```

Use `http://localhost:4000/workshops`. Use the same hostname consistently: `localhost` and `127.0.0.1` have different cookies and are different trusted origins. Each track has an Open in Livebook button, a download, and a Start a live workshop room action. The notebook launcher imports source; it does not automatically execute it.

## Collaborate with a group

1. Start a room and share the generated room link. These are guest rooms: anyone holding the link can join with a display name. They do not implement verified identities or private enrollment.
2. Each participant joins and can mark a lab complete or raise a hand. Progress is self reported, held in Presence, and removed when that connection leaves. It is not durable account progress.
3. The instructor imports a notebook once into the shared Livebook instance, then copies its `/sessions/<id>` URL without authentication tokens.
4. Paste that URL into Share notebook with room. Other participants can open the same notebook inside the page or in a new tab. Each participant must authenticate independently to Livebook. Importing the source separately creates independent notebooks.
5. Share only with trusted participants. Collaborators in one Livebook runtime can edit code and access that runtime's files and credentials. Room membership does not provision or isolate runtimes.

The Phoenix app accepts notebook session URLs only from `LIVEBOOK_URL`'s exact scheme, hostname, and port, within its session path. It rejects token/query strings, arbitrary URLs, and non-session paths. No user-supplied URL is fetched by Phoenix. If embedding is disabled by your hosting proxy, use Open in a new tab. `LIVEBOOK_WITHIN_IFRAME=true` enables Livebook's iframe support; on hosted instances restrict allowed framing origins at the reverse proxy to your workshop origin.

## Host the services

Deploy the existing Phoenix application on a host that supports long-lived Elixir processes and WebSockets. Configure `PHX_SERVER=true`, `PHX_HOST`, `SECRET_KEY_BASE`, and its usual TLS/reverse-proxy setup. Keep the normal LiveView origin checks. Run Livebook as a separate authenticated service. For unrelated learner groups, provision separate Livebook runtimes/instances rather than giving everyone access to one shared editor.

Set these integration variables:

| Variable | Purpose |
| --- | --- |
| `WORKSHOP_LIVE_URL` | Public Phoenix service origin, e.g. `https://learn.example.org` |
| `LIVEBOOK_URL` | Trusted authenticated Livebook base URL, e.g. `https://notebooks.example.org` |
| `WORKSHOP_NOTEBOOK_BASE_URL` | Public URL containing the `.livemd` assets; defaults to the GitHub Pages notebook directory |

When building the static Pages export, set `WORKSHOP_LIVE_URL` if rooms have been deployed; otherwise the site displays a clear static-mode message instead of a broken room link. A static export captures these configured URLs at build time. Use HTTPS for production. Never put an admin token into these variables or into shared session URLs.

The current implementation provides guest collaboration and trusted notebook linking; it does not add account authentication, durable progress storage, automatic runtime provisioning, per-user quotas, or deployment infrastructure.

## Notebook files and verification

The four notebooks under `priv/static/aidatafoundation/notebooks/` include Livebook Pythonx setup and Python 3.12 package requirements. They download only their listed modules/data from the published commit pinned in `scripts/generate_livebooks.py`. Each file is SHA-256 verified before executing. They use a temporary workspace and write outputs there; copy files elsewhere before stopping a runtime.

MLX notebooks must run on an Apple silicon Mac with macOS 14+ and native arm64 Python. A browser on a Mac cannot lend its GPU to a Linux server. Run MLX Livebook locally, or connect a separate Apple silicon runtime.

After changing workshop source, publish a source commit, update `SOURCE_REVISION` in the generator, and regenerate the notebooks:

```bash
python3 scripts/generate_livebooks.py
python -m pytest tests/test_livebooks.py
mix precommit
mix compile --warnings-as-errors
mix assets.deploy
mix aidatafoundation.export --output _site
python3 scripts/verify_static_export.py _site
```
