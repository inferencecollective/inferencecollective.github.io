"""Validate the workshop contract shared by the website and CLI."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from labs import workshop


def test_catalog_points_to_existing_source_data_and_guides():
    import json
    guides = {lab['id'] for lab in json.loads((workshop.ROOT / 'priv/static/aidatafoundation/data/labs.json').read_text())['labs']}
    for track in workshop.tracks():
        assert (workshop.ROOT / f"requirements/workshops/{track['id']}.txt").is_file()
        assert len(track['steps']) > 0
        for step in track['steps']:
            assert (workshop.ROOT / (step['module'].replace('.', '/') + '.py')).is_file()
            assert step['guide'] in guides


def test_missing_dependencies_print_recovery_command(capsys):
    with patch.object(workshop.sys, 'version_info', (3, 12)), patch.object(workshop.importlib.util, 'find_spec', return_value=None):
        assert workshop.main(['vision', '--check']) == 1
    assert 'requirements/workshops/vision.txt' in capsys.readouterr().err


def test_check_never_executes_a_lab():
    with patch.object(workshop.sys, 'version_info', (3, 12)), patch.object(workshop.importlib.util, 'find_spec', return_value=object()), patch.object(workshop.subprocess, 'run') as run:
        assert workshop.main(['vision', '--check']) == 0
        run.assert_not_called()


def test_runner_stops_after_first_failure():
    with patch.object(workshop.sys, 'version_info', (3, 12)), patch.object(workshop.importlib.util, 'find_spec', return_value=object()), patch.object(workshop.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as run:
        assert workshop.main(['vision']) == 7
        assert run.call_count == 1
        assert run.call_args.kwargs['cwd'] == workshop.ROOT


def test_invalid_step_is_rejected():
    import pytest
    with pytest.raises(SystemExit) as error:
        workshop.main(['vision', '--step', '0'])
    assert error.value.code == 2
