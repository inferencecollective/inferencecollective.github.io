"""Reject missing or unfingerprinted bundles before uploading a Pages artifact."""
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '_site')
manifest = json.loads((root / 'cache_manifest.json').read_text())['latest']
expected = {'/' + manifest[name] for name in ('assets/css/app.css', 'assets/js/app.js')}
errors = []

class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        for attr, url in attrs:
            if attr in ('src', 'href') and url and url.startswith('/assets/'):
                self.urls.add(url.split('?')[0])

pages = list(root.rglob('*.html'))
assert pages, 'No exported pages found'
for page in pages:
    parser = Assets()
    parser.feed(page.read_text())
    for url in parser.urls:
        if not (root / url.lstrip('/')).is_file():
            errors.append(f'{page}: missing {url}')
        if url.endswith(('/app.css', '/app.js')):
            errors.append(f'{page}: bundle has no content fingerprint: {url}')
    if not any(url in expected and url.endswith('.css') for url in parser.urls):
        errors.append(f'{page}: missing current stylesheet')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Validated fingerprinted assets across {len(pages)} exported HTML pages.')
