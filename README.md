# diffsptk

*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/2.4.0/)
[![Downloads](https://static.pepy.tech/badge/diffsptk)](https://pepy.tech/project/diffsptk)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0.0%20%7C%202.6.0-orange.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Codecov](https://codecov.io/gh/sp-nitech/diffsptk/branch/master/graph/badge.svg)](https://app.codecov.io/gh/sp-nitech/diffsptk)
[![License](https://img.shields.io/github/license/sp-nitech/diffsptk.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)
[![GitHub Actions](https://github.com/sp-nitech/diffsptk/workflows/package/badge.svg)](https://github.com/sp-nitech/diffsptk/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python 3.9+
- PyTorch 2.0.0+

## Documentation

- See [this page](https://sp-nitech.github.io/diffsptk/latest/) for the reference manual.
- Our [paper](https://www.isca-speech.org/archive/ssw_2023/yoshimura23_ssw.html) is available on the ISCA Archive.

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
mcep = diffsptk.MelCepstralAnalysis(cep_order=M, fft_length=n_fft, alpha=alpha, n_iter=10)
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
pitch = diffsptk.Pitch(frame_period=fp, sample_rate=sr, f_min=80, f_max=180)
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
lpc = diffsptk.LPC(frame_length=fl, lpc_order=M, eps=1e-6)
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

### Mel-spectrogram, MFCC, and PLP extraction

```python
import diffsptk

fl = 400        # Frame length
fp = 80         # Frame period
n_fft = 512     # FFT length
n_channel = 80  # Number of channels
M = 12          # MFCC/PLP dimensions

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Compute STFT amplitude of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Extract log mel-spectrogram.
fbank = diffsptk.MelFilterBankAnalysis(
    n_channel=n_channel,
    fft_length=n_fft,
    sample_rate=sr,
)
Y = fbank(X)
print(Y.shape)

# Extract MFCC.
mfcc = diffsptk.MFCC(
    mfcc_order=M,
    n_channel=n_channel,
    fft_length=n_fft,
    sample_rate=sr,
)
Y = mfcc(X)
print(Y.shape)

# Extract PLP.
plp = diffsptk.PLP(
    plp_order=M,
    n_channel=n_channel,
    fft_length=n_fft,
    sample_rate=sr,
)
Y = plp(X)
print(Y.shape)
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
imdct = diffpstk.IMDCT(fl)
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
