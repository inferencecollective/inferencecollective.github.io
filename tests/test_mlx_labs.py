"""Numerical contracts for the Apple silicon MLX track."""
import platform
import pytest

if platform.system() != 'Darwin' or platform.machine() != 'arm64':
    pytest.skip('MLX workshop tests require Apple silicon macOS', allow_module_level=True)
pytest.importorskip('mlx.core')

from labs import lab01_mlx_autograd, lab02_mlx_compile


def test_regression_converges_and_saves_report(tmp_path, monkeypatch):
    monkeypatch.setenv('MLX_WORKSHOP_DEVICE', 'cpu')
    result = lab01_mlx_autograd.run(tmp_path)
    assert result['final_loss'] < 1e-5
    assert abs(result['slope'] - 2.5) < 0.01
    assert abs(result['intercept'] + 0.8) < 0.01
    assert (tmp_path / 'mlx-autograd.json').is_file()


def test_compilation_preserves_outputs_and_records_timings(tmp_path, monkeypatch):
    monkeypatch.setenv('MLX_WORKSHOP_DEVICE', 'cpu')
    result = lab02_mlx_compile.run(tmp_path, repeats=3)
    assert result['outputs_match'] is True
    assert result['compiled_median_ms'] > 0
    assert result['uncompiled_median_ms'] > 0
    assert (tmp_path / 'mlx-compile.json').is_file()
