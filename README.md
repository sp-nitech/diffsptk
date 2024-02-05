diffsptk
========
*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/1.2.1/)
[![Downloads](https://static.pepy.tech/badge/diffsptk)](https://pepy.tech/project/diffsptk)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.11.0%20%7C%202.2.0-orange.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Codecov](https://codecov.io/gh/sp-nitech/diffsptk/branch/master/graph/badge.svg)](https://app.codecov.io/gh/sp-nitech/diffsptk)
[![License](https://img.shields.io/github/license/sp-nitech/diffsptk.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)
[![GitHub Actions](https://github.com/sp-nitech/diffsptk/workflows/package/badge.svg)](https://github.com/sp-nitech/diffsptk/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Requirements
------------
- Python 3.8+
- PyTorch 1.11.0+


Documentation
-------------
- See [this page](https://sp-nitech.github.io/diffsptk/latest/) for a reference manual.
- Our [paper](https://www.isca-speech.org/archive/ssw_2023/yoshimura23_ssw.html) is available on the ISCA Archive.


Installation
------------
The latest stable release can be installed through PyPI by running
```sh
pip install diffsptk
```
The development release can be installed from the master branch:
```sh
pip install git+https://github.com/sp-nitech/diffsptk.git@master
```


Examples
--------
### Mel-cepstral analysis and synthesis
```python
import diffsptk

# Set analysis condition.
fl = 400
fp = 80
n_fft = 512
M = 24

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
mlsa = diffsptk.MLSA(filter_order=M, frame_period=fp, alpha=alpha, taylor_order=30)
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

### Mel-spectrogram, MFCC, and PLP extraction
```python
import diffsptk

# Set analysis condition.
fl = 400
fp = 80
n_fft = 512
n_channel = 80
M = 12

# Read waveform.
x, sr = diffsptk.read("assets/data.wav")

# Compute STFT amplitude of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Extract mel-spectrogram.
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
y = decimate(pqmf(x), dim=-1)

# Reconstruct x.
interpolate = diffsptk.Interpolation(K)
ipqmf = diffsptk.IPQMF(K, M)
x_hat = ipqmf(interpolate(K * y, dim=-1)).reshape(-1)

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


License
-------
This software is released under the Apache License 2.0.


Reference
---------
```bibtex
@InProceedings{sp-nitech2023sptk,
  author = {Takenori Yoshimura and Takato Fujimoto and Keiichiro Oura and Keiichi Tokuda},
  title = {{SPTK4}: An open-source software toolkit for speech signal processing},
  booktitle = {12th ISCASpeech Synthesis Workshop (SSW 2023)},
  pages = {211--217},
  year = {2023},
}
```
