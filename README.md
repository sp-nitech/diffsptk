diffsptk
========
*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://sp-nitech.github.io/diffsptk/0.1.0/)
[![PyPI Version](https://img.shields.io/pypi/v/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![Python Version](https://img.shields.io/pypi/pyversions/diffsptk.svg)](https://pypi.python.org/pypi/diffsptk)
[![License](http://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)


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

x = torch.randn(100)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=12, frame_period=10, fft_length=32)
X = stft(x)

# Apply 4 mel-filter banks to the STFT.
fbank = diffsptk.MelFilterBankAnalysis(n_channel=4, fft_length=32, sample_rate=8000)
Y = fbank(X)
```

### Subband decomposition
```python
import diffsptk
import torch

K = 4   # Number of subbands.
M = 40  # Order of filter.

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
- [x] acorr
- [ ] acr2csm
- [ ] ~~aeq~~ (*torch.allclose*)
- [ ] ~~amgcep~~
- [ ] ~~average~~ (*torch.mean*)
- [x] b2mc
- [ ] ~~bcp~~
- [ ] ~~bcut~~
- [x] c2acr
- [x] c2mpir
- [x] c2ndps
- [x] cdist
- [ ] ~~clip~~ (*torch.clip*)
- [ ] csm2acr
- [x] dct
- [x] decimate
- [ ] ~~delay~~
- [x] delta
- [x] dequantize
- [ ] df2
- [ ] dfs
- [ ] ~~dmp~~
- [ ] dtw
- [ ] ~~dtw_merge~~
- [ ] ~~entropy~~ (*torch.special.entr*)
- [ ] excite
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
- [ ] grpdelay
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
- [ ] lar2par
- [ ] ~~lbg~~
- [x] levdur
- [x] linear_intpl
- [x] lpc
- [ ] lpc2c
- [ ] lpc2lsp
- [ ] lpc2par
- [ ] lpccheck
- [ ] lsp2lpc
- [ ] lspcheck
- [ ] lspdf
- [ ] ltcdf
- [x] mc2b
- [ ] mcpf
- [ ] ~~median~~ (*torch.median*)
- [ ] ~~merge~~
- [x] mfcc
- [ ] mgc2mgc
- [ ] mgc2sp
- [ ] mgcep (*mcep is now available*)
- [ ] mglsadf
- [ ] mglsp2sp
- [ ] ~~minmax~~
- [ ] mlpg
- [ ] mlsacheck
- [ ] mpir2c
- [ ] mseq
- [ ] msvq
- [ ] ~~nan~~ (*torch.isnan*)
- [x] ndps2c
- [ ] norm0
- [ ] ~~nrand~~ (*torch.randn*)
- [ ] par2lar
- [ ] par2lpc
- [ ] pca
- [ ] pcas
- [ ] phase
- [ ] pitch
- [ ] ~~pitch_mark~~
- [ ] poledf
- [x] pqmf
- [x] quantize
- [x] ramp
- [ ] ~~reverse~~
- [ ] rlevdur
- [ ] ~~rmse~~
- [ ] root_pol
- [x] sin
- [ ] smcep
- [ ] snr
- [ ] ~~sopr~~
- [x] spec
- [x] step
- [ ] ~~swab~~
- [ ] ~~symmetrize~~
- [ ] train
- [ ] ~~transpose~~
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
