# diffsptk

*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/3.3.1/)
[![Downloads](https://static.pepy.tech/badge/diffsptk)](https://pepy.tech/project/diffsptk)
[![ClickPy](https://img.shields.io/badge/downloads-clickpy-yellow.svg)](https://clickpy.clickhouse.com/dashboard/diffsptk)
[![Advisor](https://snyk.io/advisor/python/diffsptk/badge.svg)](https://snyk.io/advisor/python/diffsptk)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.3.1%20%7C%202.8.0-orange.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Codecov](https://codecov.io/gh/sp-nitech/diffsptk/branch/master/graph/badge.svg)](https://app.codecov.io/gh/sp-nitech/diffsptk)
[![License](https://img.shields.io/github/license/sp-nitech/diffsptk.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)
[![GitHub Actions](https://github.com/sp-nitech/diffsptk/workflows/package/badge.svg)](https://github.com/sp-nitech/diffsptk/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python 3.10+
- PyTorch 2.3.1+

## Documentation

- Online [documentation](https://sp-nitech.github.io/diffsptk/3.3.1/) for the reference manual
- Conference [paper](https://www.isca-archive.org/ssw_2023/yoshimura23_ssw.html) on the ISCA Archive
- Hands-on [tutorial](https://colab.research.google.com/drive/1xAoUKqXadvJXJ7RzN0OceB6y7q5i7Sn6?usp=drive_link) on Google Colab

## Installation

The latest stable release can be installed through PyPI by running

```sh
pip install diffsptk
```

The development release can be installed from the master branch:

```sh
pip install git+https://github.com/sp-nitech/diffsptk.git@master
```

## Examples

### Running on a GPU

```python
import diffsptk

stft_params = {"frame_length": 400, "frame_period": 80, "fft_length": 512}

# Read waveform.
x, sr = diffsptk.read("assets/data.wav", device="cuda")

# Compute spectrogram using a nn.Module class.
X1 = diffsptk.STFT(**stft_params, device="cuda")(x)

# Compute spectrogram using a functional method.
X2 = diffsptk.functional.stft(x, **stft_params)

print(X1.allclose(X2))
```

### Mel-cepstral analysis and synthesis

```python
import diffsptk

fl = 400     # Frame length.
fp = 80      # Frame period.
n_fft = 512  # FFT length.
M = 24       # Mel-cepstrum dimensions.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Compute STFT amplitude of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Estimate mel-cepstrum of x.
alpha = diffsptk.get_alpha(sr)
mcep = diffsptk.MelCepstralAnalysis(
    fft_length=n_fft,
    cep_order=M,
    alpha=alpha,
    n_iter=10,
)
mc = mcep(X)

# Reconstruct x.
mlsa = diffsptk.MLSA(filter_order=M, frame_period=fp, alpha=alpha, taylor_order=20)
x_hat = mlsa(mlsa(x, -mc), mc)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)

# Extract pitch of x.
pitch = diffsptk.Pitch(
    frame_period=fp,
    sample_rate=sr,
    f_min=80,
    f_max=180,
    voicing_threshold=0.4,
    out_format="pitch",
)
p = pitch(x)

# Generate excitation signal.
excite = diffsptk.ExcitationGeneration(frame_period=fp)
e = excite(p)
n = diffsptk.nrand(x.size(0) - 1)

# Synthesize waveform.
x_voiced = mlsa(e, mc)
x_unvoiced = mlsa(n, mc)

# Output analysis-synthesis result.
diffsptk.write("voiced.wav", x_voiced, sr)
diffsptk.write("unvoiced.wav", x_unvoiced, sr)
```

### WORLD analysis and synthesis

```python
import diffsptk

fp = 80       # Frame period.
n_fft = 1024  # FFT length.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Extract F0 of x, or prepare well-estimated F0.
pitch = diffsptk.Pitch(
    frame_period=fp,
    sample_rate=sr,
    f_min=80,
    f_max=180,
    voicing_threshold=0.4,
    out_format="f0",
)
f0 = pitch(x)

# Extract aperiodicity of x by D4C.
ap = diffsptk.Aperiodicity(
    frame_period=fp,
    sample_rate=sr,
    fft_length=n_fft,
    algorithm="d4c",
    out_format="a",
)
A = ap(x, f0)

# Extract spectral envelope of x by CheapTrick.
pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
    frame_period=fp,
    sample_rate=sr,
    fft_length=n_fft,
    algorithm="cheap-trick",
    out_format="power",
)
S = pitch_spec(x, f0)

# Reconstruct x.
world_synth = diffsptk.WorldSynthesis(
    frame_period=fp,
    sample_rate=sr,
    fft_length=n_fft,
)
x_hat = world_synth(f0, A, S)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### LPC analysis and synthesis

```python
import diffsptk

fl = 400  # Frame length.
fp = 80   # Frame period.
M = 24    # LPC dimensions.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Estimate LPC of x.
frame = diffsptk.Frame(frame_length=fl, frame_period=fp)
window = diffsptk.Window(in_length=fl)
lpc = diffsptk.LPC(frame_length=fl, lpc_order=M, eps=1e-5)
a = lpc(window(frame(x)))

# Convert to inverse filter coefficients.
norm0 = diffsptk.AllPoleToAllZeroDigitalFilterCoefficients(filter_order=M)
b = norm0(a)

# Reconstruct x.
zerodf = diffsptk.AllZeroDigitalFilter(filter_order=M, frame_period=fp)
poledf = diffsptk.AllPoleDigitalFilter(filter_order=M, frame_period=fp)
x_hat = poledf(zerodf(x, b), a)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Mel spectrogram analysis and synthesis

```python
import diffsptk

fl = 400         # Frame length.
fp = 80          # Frame period.
n_fft = 512      # FFT length.
n_channel = 128  # Number of channels.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Compute STFT amplitude of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Extract log-mel spectrogram.
fbank = diffsptk.FBANK(
    fft_length=n_fft,
    n_channel=n_channel,
    sample_rate=sr,
)
Y = fbank(X)

# Reconstruct linear spectrogram.
ifbank = diffsptk.IFBANK(
    n_channel=n_channel,
    fft_length=n_fft,
    sample_rate=sr,
)
X_hat = ifbank(Y)

# Reconstruct x.
griffin = diffsptk.GriffinLim(
    frame_length=fl,
    frame_period=fp,
    fft_length=n_fft,
)
x_hat = griffin(X_hat, out_length=x.size(0))

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Subband decomposition

```python
import diffsptk

K = 4   # Number of subbands.
M = 40  # Order of filter.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Decompose x.
pqmf = diffsptk.PQMF(K, M)
decimate = diffsptk.Decimation(K)
y = decimate(pqmf(x))

# Reconstruct x.
interpolate = diffsptk.Interpolation(K)
ipqmf = diffsptk.IPQMF(K, M)
x_hat = ipqmf(interpolate(K * y)).reshape(-1)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Gammatone filter bank analysis and synthesis

```python
import diffsptk

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Decompose x.
gammatone = diffsptk.GammatoneFilterBankAnalysis(sr)
y = gammatone(x)

# Reconstruct x.
igammatone = diffsptk.GammatoneFilterBankSynthesis(sr)
x_hat = igammatone(y).reshape(-1)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Fractional octave band analysis and synthesis

```python
import diffsptk

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Decompose x.
oband = diffsptk.FractionalOctaveBandAnalysis(sr)
y = oband(x)

# Reconstruct x.
x_hat = y.sum(1).reshape(-1)

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Constant-Q transform

```python
import diffsptk
import librosa  # This is to get sample audio.

fp = 128  # Frame period.
K = 252   # Number of CQ-bins.
B = 36    # Number of bins per octave.

# Read waveform.
x, sr = diffsptk.read(librosa.ex("trumpet"))

# Transform x.
cqt = diffsptk.CQT(fp, sr, n_bin=K, n_bin_per_octave=B)
c = cqt(x)

# Reconstruct x.
icqt = diffsptk.ICQT(fp, sr, n_bin=K, n_bin_per_octave=B)
x_hat = icqt(c, out_length=x.size(0))

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Modified discrete cosine transform

```python
import diffsptk

fl = 512  # Frame length.

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Transform x.
mdct = diffsptk.MDCT(fl)
c = mdct(x)

# Reconstruct x.
imdct = diffsptk.IMDCT(fl)
x_hat = imdct(c, out_length=x.size(0))

# Write reconstructed waveform.
diffsptk.write("reconst.wav", x_hat, sr)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

### Vector quantization

```python
import diffsptk

K = 2  # Codebook size.
M = 4  # Order of vector.

# Prepare input.
x = diffsptk.nrand(M)

# Quantize x.
vq = diffsptk.VectorQuantization(M, K)
x_hat, indices, commitment_loss = vq(x)

# Compute error.
error = (x_hat - x).abs().sum()
print(error)
```

## License

This software is released under the Apache License 2.0.

## Citation

```bibtex
@InProceedings{sp-nitech2023sptk,
  author = {Takenori Yoshimura and Takato Fujimoto and Keiichiro Oura and Keiichi Tokuda},
  title = {{SPTK4}: An open-source software toolkit for speech signal processing},
  booktitle = {12th ISCA Speech Synthesis Workshop (SSW 2023)},
  pages = {211--217},
  year = {2023},
}
```
