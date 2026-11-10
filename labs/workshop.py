"""Run curated workshops from the repository root, without requiring web dependencies."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / 'priv/static/aidatafoundation/data/workshops.json'
IMPORTS = {'scikit-learn': 'sklearn', 'Pillow': 'PIL'}


def tracks():
    return json.loads(CATALOG.read_text())['tracks']


def main(argv=None):
    catalog = {track['id']: track for track in tracks()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=catalog)
    parser.add_argument('--step', type=int, help='Run one lab, numbered from 1')
    parser.add_argument('--check', action='store_true', help='Check dependencies without running code')
    args = parser.parse_args(argv)
    track = catalog[args.track]
    if args.step is not None and not 1 <= args.step <= len(track['steps']):
        parser.error(f"--step must be between 1 and {len(track['steps'])}")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        print('Use Python 3.11–3.13 for these workshops.', file=sys.stderr)
        return 1
    missing = [name for name in track['packages']
               if importlib.util.find_spec(IMPORTS.get(name, name)) is None]
    if missing:
        print('Missing packages: ' + ', '.join(missing), file=sys.stderr)
        print(f'Install with: "{sys.executable}" -m pip install -r requirements/workshops/{args.track}.txt', file=sys.stderr)
        return 1
    print(f"Ready: {track['title']} (Python {sys.version.split()[0]})", flush=True)
    if args.check:
        return 0
    steps = track['steps'] if args.step is None else [track['steps'][args.step - 1]]
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    for step in steps:
        print(f"\nRunning: {step['title']}", flush=True)
        result = subprocess.run([sys.executable, '-m', step['module']], cwd=ROOT, env=env)
        if result.returncode:
            print('Lab stopped. Fix the error above, then rerun this step.', file=sys.stderr)
            return result.returncode
        print(f"Inspect: {step['outcome']}", flush=True)
    print('\nRun complete. Try the extension exercise before marking the lab complete.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
