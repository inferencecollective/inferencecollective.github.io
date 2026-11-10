"""Device selection shared by the MLX workshop and its preflight check."""
import os


def configure_device():
    import mlx.core as mx

    requested = os.environ.get('MLX_WORKSHOP_DEVICE', 'auto')
    available = mx.metal.is_available()
    if requested == 'gpu' and not available:
        raise RuntimeError('Metal GPU is unavailable. Retry with --device cpu.')
    device = 'gpu' if requested == 'gpu' or (requested == 'auto' and available) else 'cpu'
    mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
    # Force a real operation: import alone cannot prove the backend works.
    mx.eval(mx.array([1.0, 2.0]) + 1)
    return device
