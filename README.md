diffsptk
========
*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/0.3.0/)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.9.0%20%7C%201.11.0-orange.svg)](https://pypi.python.org/pypi/diffsptk)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Codecov](https://codecov.io/gh/sp-nitech/diffsptk/branch/master/graph/badge.svg)](https://app.codecov.io/gh/sp-nitech/diffsptk)
[![License](https://img.shields.io/github/license/sp-nitech/diffsptk.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)
[![GitHub Actions](https://github.com/sp-nitech/diffsptk/workflows/package/badge.svg)](https://github.com/sp-nitech/diffsptk/actions)


Requirements
------------
- Python 3.8+
- PyTorch 1.9.0+


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
### Mel-cepstral analysis
```python
import diffsptk
import torch

# Generate waveform.
x = torch.randn(100)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=12, frame_period=10, fft_length=16)
X = stft(x)

# Estimate 4-th order mel-cepstrum of x.
mcep = diffsptk.MelCepstralAnalysis(cep_order=4, fft_length=16, alpha=0.1, n_iter=1)
mc = mcep(X)
```

### Mel-spectrogram extraction
```python
import diffsptk
import torch

# Generate waveform.
x = torch.randn(100)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=12, frame_period=10, fft_length=32)
X = stft(x)

# Apply 4 mel-filter banks to the STFT.
fbank = diffsptk.MelFilterBankAnalysis(n_channel=4, fft_length=32, sample_rate=8000, floor=1e-1)
Y = fbank(X)
```

### Subband decomposition
```python
import diffsptk
import torch

K = 4   # Number of subbands.
M = 40  # Order of filter.

# Generate waveform.
x = torch.randn(100)

# Decompose x.
pqmf = diffsptk.PQMF(K, M)
decimate = diffsptk.Decimation(K)
y = decimate(pqmf(x), dim=-1)

# Reconstruct x.
interpolate = diffsptk.Interpolation(K)
ipqmf = diffsptk.IPQMF(K, M)
x_hat = ipqmf(interpolate(K * y, dim=-1))

# Compute error between two signals.
error = torch.abs(x_hat - x).sum()
```


Status
------
~~module~~ will not be implemented in this repository.
- [x] acorr
- [ ] ~~acr2csm~~
- [ ] ~~aeq~~ (*torch.allclose*)
- [ ] ~~amgcep~~
- [ ] ~~average~~ (*torch.mean*)
- [x] b2mc
- [ ] ~~bcp~~ (*torch.split*)
- [ ] ~~bcut~~
- [x] c2acr
- [x] c2mpir
- [x] c2ndps
- [x] cdist
- [ ] ~~clip~~ (*torch.clip*)
- [ ] ~~csm2acr~~
- [x] dct
- [x] decimate
- [x] delay
- [x] delta
- [x] dequantize
- [x] df2
- [x] dfs
- [ ] ~~dmp~~
- [ ] dtw
- [ ] ~~dtw_merge~~
- [ ] ~~entropy~~ (*torch.special.entr*)
- [ ] ~~excite~~
- [ ] ~~extract~~
- [x] fbank
- [ ] ~~fd~~
- [ ] ~~fdrw~~
- [ ] ~~fft~~ (*torch.fft.fft*)
- [ ] ~~fft2~~ (*torch.fft.fft2*)
- [x] fftcep
- [ ] ~~fftr~~ (*torch.fft.rfft*)
- [ ] ~~fftr2~~ (*torch.fft.rfft2*)
- [x] frame
- [x] freqt
- [ ] ~~glogsp~~
- [ ] ~~gmm~~
- [ ] ~~gmmp~~
- [x] gnorm
- [ ] ~~gpolezero~~
- [ ] ~~grlogsp~~
- [x] grpdelay
- [ ] ~~gseries~~
- [ ] ~~gspecgram~~
- [ ] ~~gwave~~
- [ ] ~~histogram~~ (*torch.histogram*)
- [ ] ~~huffman~~
- [ ] ~~huffman_decode~~
- [ ] ~~huffman_encode~~
- [x] idct
- [ ] ~~ifft~~ (*torch.fft.ifft*)
- [ ] ~~ifft2~~ (*torch.fft.ifft2*)
- [x] ignorm
- [ ] imglsadf
- [x] impulse
- [ ] imsvq
- [x] interpolate
- [x] ipqmf
- [x] iulaw
- [x] lar2par
- [ ] ~~lbg~~
- [x] levdur
- [x] linear_intpl
- [x] lpc
- [ ] ~~lpc2c~~
- [ ] ~~lpc2lsp~~
- [x] lpc2par
- [x] lpccheck
- [ ] ~~lsp2lpc~~
- [ ] ~~lspcheck~~
- [ ] ~~lspdf~~
- [ ] ltcdf
- [x] mc2b
- [x] mcpf
- [ ] ~~median~~ (*torch.median*)
- [ ] ~~merge~~ (*torch.cat*)
- [x] mfcc
- [x] mgc2mgc
- [x] mgc2sp
- [x] mgcep
- [ ] mglsadf
- [ ] ~~mglsp2sp~~
- [ ] ~~minmax~~
- [x] mlpg (*support only unit variance*)
- [ ] mlsacheck
- [x] mpir2c
- [ ] ~~mseq~~
- [ ] msvq
- [ ] ~~nan~~ (*torch.isnan*)
- [x] ndps2c
- [x] norm0
- [ ] ~~nrand~~ (*torch.randn*)
- [x] par2lar
- [x] par2lpc
- [ ] pca
- [ ] pcas
- [x] phase
- [ ] ~~pitch~~
- [ ] ~~pitch_mark~~
- [ ] poledf
- [x] pqmf
- [x] quantize
- [x] ramp
- [ ] ~~reverse~~ (*torch.flip*)
- [ ] ~~rlevdur~~
- [x] rmse
- [ ] ~~root_pol~~
- [x] sin
- [ ] ~~smcep~~
- [x] snr
- [ ] ~~sopr~~
- [x] spec
- [x] step
- [ ] ~~swab~~
- [ ] ~~symmetrize~~
- [ ] ~~train~~
- [ ] ~~transpose~~ (*torch.transpose*)
- [x] ulaw
- [ ] ~~vc~~
- [ ] ~~vopr~~
- [ ] ~~vstat~~ (*torch.var_mean*)
- [ ] ~~vsum~~ (*torch.sum*)
- [x] window
- [ ] ~~x2x~~
- [x] zcross
- [x] zerodf


License
-------
This software is released under the Apache License 2.0.
