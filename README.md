diffsptk
========
*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/0.5.0/)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10.0%20%7C%201.12.0-orange.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Codecov](https://codecov.io/gh/sp-nitech/diffsptk/branch/master/graph/badge.svg)](https://app.codecov.io/gh/sp-nitech/diffsptk)
[![License](https://img.shields.io/github/license/sp-nitech/diffsptk.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)
[![GitHub Actions](https://github.com/sp-nitech/diffsptk/workflows/package/badge.svg)](https://github.com/sp-nitech/diffsptk/actions)


Requirements
------------
- Python 3.8+
- PyTorch 1.10.0+


Documentation
-------------
See [this page](https://sp-nitech.github.io/diffsptk/latest/) for a reference manual.


Installation
------------
The latest stable release can be installed through PyPI by running
```sh
pip install diffsptk
```
Alternatively,
```sh
git clone https://github.com/sp-nitech/diffsptk.git
pip install -e diffsptk
```


Examples
--------
### Mel-cepstral analysis and synthesis
```python
import diffsptk
import soundfile as sf
import torch

# Set analysis condition.
fl = 400
fp = 80
n_fft = 512
M = 24
alpha = 0.42

# Read waveform.
x, sr = sf.read("assets/data.wav")
x = torch.FloatTensor(x)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Estimate mel-cepstrum of x.
mcep = diffsptk.MelCepstralAnalysis(cep_order=M, fft_length=n_fft, alpha=alpha)
mc = mcep(X)

# Generate excitation.
e = torch.randn(x.size(0))

# Apply MLSA filter to the excitation.
mlsa = diffsptk.MLSA(filter_order=M, alpha=alpha, frame_period=fp)
y = mlsa(e, mc)

# Write waveform.
sf.write("unvoice.wav", y.cpu().numpy(), sr)
```

### Mel-spectrogram extraction
```python
import diffsptk
import soundfile as sf
import torch

# Set analysis condition.
fl = 400
fp = 80
n_fft = 512
n_channel = 80

# Read waveform.
x, sr = sf.read("assets/data.wav")
x = torch.FloatTensor(x)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
X = stft(x)

# Apply mel-filter banks to the STFT.
fbank = diffsptk.MelFilterBankAnalysis(
    n_channel=n_channel,
    fft_length=n_fft,
    sample_rate=sr,
)
Y = fbank(X)
```

### Subband decomposition
```python
import diffsptk
import soundfile as sf
import torch

K = 4   # Number of subbands.
M = 40  # Order of filter.

# Read waveform.
x, sr = sf.read("assets/data.wav")
x = torch.FloatTensor(x)

# Decompose x.
pqmf = diffsptk.PQMF(K, M)
decimate = diffsptk.Decimation(K)
y = decimate(pqmf(x), dim=-1)

# Reconstruct x.
interpolate = diffsptk.Interpolation(K)
ipqmf = diffsptk.IPQMF(K, M)
x_hat = ipqmf(interpolate(K * y, dim=-1)).reshape(-1)

# Compute error between two signals.
error = torch.abs(x_hat - x).sum()

# Write reconstructed waveform.
sf.write("reconst.wav", x_hat.cpu().numpy(), sr)
```


License
-------
This software is released under the Apache License 2.0.
