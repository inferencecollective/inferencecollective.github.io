"""Lab 4: find a dominant tone and remove high-frequency noise."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt


def analyze_signal(samples, sampling_rate=200.0, cutoff=10.0):
    """Return dominant non-DC frequency and low-pass filtered samples."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < 32 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be a finite 1D signal with >= 32 values")
    if not 0 < cutoff < sampling_rate / 2:
        raise ValueError("cutoff must be between zero and Nyquist frequency")
    # Remove the mean so the DC bin does not dominate.
    centered = x - x.mean()
    magnitudes = np.abs(rfft(centered))
    freqs = rfftfreq(x.size, d=1.0 / sampling_rate)
    strongest_bin = int(np.argmax(magnitudes[1:]) + 1)
    sos = butter(4, cutoff, btype="lowpass", fs=sampling_rate, output="sos")
    filtered = sosfiltfilt(sos, x)
    return {"dominant_hz": float(freqs[strongest_bin]), "filtered": filtered}


if __name__ == "__main__":
    fs = 200.0
    t = np.arange(400) / fs
    signal = np.sin(2 * np.pi * 5 * t) + 0.25 * np.sin(2 * np.pi * 40 * t)
    report = analyze_signal(signal, sampling_rate=fs)
    print(f"Dominant frequency: {report['dominant_hz']:.1f} Hz")
    print("Challenge: move the noise frequency below the low-pass cutoff.")
