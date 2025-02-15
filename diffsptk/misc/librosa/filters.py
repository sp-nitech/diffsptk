import numpy as np
import scipy

from .util import normalize
from .util import pad_center
from .util import phasor
from .util import tiny

WINDOW_BANDWIDTHS = {
    "bart": 1.3334961334912805,
    "barthann": 1.4560255965133932,
    "bartlett": 1.3334961334912805,
    "bkh": 2.0045975283585014,
    "black": 1.7269681554262326,
    "blackharr": 2.0045975283585014,
    "blackman": 1.7269681554262326,
    "blackmanharris": 2.0045975283585014,
    "blk": 1.7269681554262326,
    "bman": 1.7859588613860062,
    "bmn": 1.7859588613860062,
    "bohman": 1.7859588613860062,
    "box": 1.0,
    "boxcar": 1.0,
    "brt": 1.3334961334912805,
    "brthan": 1.4560255965133932,
    "bth": 1.4560255965133932,
    "cosine": 1.2337005350199792,
    "flat": 2.7762255046484143,
    "flattop": 2.7762255046484143,
    "flt": 2.7762255046484143,
    "halfcosine": 1.2337005350199792,
    "ham": 1.3629455320350348,
    "hamm": 1.3629455320350348,
    "hamming": 1.3629455320350348,
    "han": 1.50018310546875,
    "hann": 1.50018310546875,
    "nut": 1.9763500280946082,
    "nutl": 1.9763500280946082,
    "nuttall": 1.9763500280946082,
    "ones": 1.0,
    "par": 1.9174603174603191,
    "parz": 1.9174603174603191,
    "parzen": 1.9174603174603191,
    "rect": 1.0,
    "rectangular": 1.0,
    "tri": 1.3331706523555851,
    "triang": 1.3331706523555851,
    "triangle": 1.3331706523555851,
}


def chroma(
    *,
    sr,
    n_fft,
    n_chroma=12,
    tuning=0.0,
    ctroct=5.0,
    octwidth=2,
    norm=2,
    base_c=True,
    dtype=np.float32,
):
    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    def hz_to_octs(
        frequencies,
        *,
        tuning=0.0,
        bins_per_octave=12,
    ):
        A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

        octs = np.log2(np.asanyarray(frequencies) / (float(A440) / 16))
        return octs

    frqbins = n_chroma * hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # normalize each column
    wts = normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2)),
            (n_chroma, 1),
        )

    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)], dtype=dtype)


def float_window(window_spec):
    def _wrap(n, *args, **kwargs):
        """Wrap the window"""
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode="constant")

        window[n_min:] = 0.0

        return window

    return _wrap


def get_window(window, Nx, *, fftbins=True):
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)
        return win

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ValueError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ValueError(f"Invalid window specification: {window!r}")


def relative_bandwidth(*, freqs):
    if len(freqs) <= 1:
        raise ValueError(
            f"2 or more frequencies are required to compute bandwidths. Given freqs={freqs}"
        )

    # Approximate the local octave resolution around each frequency
    bpo = np.empty_like(freqs)
    logf = np.log2(freqs)
    # Reflect at the lowest and highest frequencies
    bpo[0] = 1 / (logf[1] - logf[0])
    bpo[-1] = 1 / (logf[-1] - logf[-2])

    # For everything else, do a centered difference
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])

    # Compute relative bandwidths
    alpha = (2.0 ** (2 / bpo) - 1) / (2.0 ** (2 / bpo) + 1)
    return alpha


def wavelet(
    *,
    freqs,
    sr=22050,
    window="hann",
    filter_scale=1,
    pad_fft=True,
    norm=1,
    dtype=np.complex64,
    gamma=0,
    alpha=None,
    **kwargs,
):
    # Pass-through parameters to get the filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
    )

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = phasor(
            np.arange(-ilen // 2, ilen // 2, dtype=float) * 2 * np.pi * freq / sr
        )

        # Apply the windowing function
        sig *= float_window(window)(len(sig))

        # Normalize
        sig = normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [pad_center(filt, size=max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, lengths


def wavelet_lengths(
    *,
    freqs,
    sr=22050,
    window="hann",
    filter_scale=1,
    gamma=0,
    alpha=None,
):
    freqs = np.asarray(freqs)
    if filter_scale <= 0:
        raise ValueError(f"filter_scale={filter_scale} must be positive")

    if gamma is not None and gamma < 0:
        raise ValueError(f"gamma={gamma} must be non-negative")

    if np.any(freqs <= 0):
        raise ValueError("frequencies must be strictly positive")

    if len(freqs) > 1 and np.any(freqs[:-1] > freqs[1:]):
        raise ValueError(f"Frequency array={freqs} must be in strictly ascending order")

    if alpha is None:
        alpha = relative_bandwidth(freqs=freqs)
    else:
        alpha = np.asarray(alpha)

    if gamma is None:
        gamma_ = alpha * 24.7 / 0.108
    else:
        gamma_ = gamma
    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / alpha

    # How far up does our highest frequency reach?
    f_cutoff = max(freqs * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma_)

    # Convert frequencies to filter lengths
    lengths = Q * sr / (freqs + gamma_ / alpha)

    return lengths, f_cutoff


def window_bandwidth(window, n=1000):
    if hasattr(window, "__name__"):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = n * np.sum(win**2) / (np.sum(win) ** 2 + tiny(win))

    return WINDOW_BANDWIDTHS[key]
