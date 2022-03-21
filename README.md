diffsptk (UNDER CONSTRUCTION)
=============================
*diffsptk* is a differentiable version of [SPTK](https://github.com/sp-nitech/SPTK) based on the PyTorch framework.

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://sp-nitech.github.io/diffsptk/latest/)
[![](http://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/sp-nitech/diffsptk/blob/master/LICENSE)


Requirements
------------
- PyTorch 1.8.0+


Documentation
-------------
See [this page](https://sp-nitech.github.io/diffsptk/latest/) for a reference manual.


Installation
------------
```sh
git clone https://github.com/sp-nitech/diffsptk.git
pip install -e diffsptk
```


Examples
--------
### Cepstral analysis
```python
import diffsptk
import torch

x = torch.randn(100)

# Compute STFT of x.
stft = diffsptk.STFT(frame_length=12, frame_period=10, fft_length=16)
X = stft(x)

# Estimate 4-th order cepstrum of x.
fftcep = diffsptk.CepstralAnalysis(cep_order=4, fft_length=16, n_iter=1)
c = fftcep(X)
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
- [x] ~~aeq~~
- [x] ~~amgcep~~
- [x] ~~average~~ (*torch.mean*)
- [x] b2mc
- [x] ~~bcp~~
- [x] ~~bcut~~
- [ ] c2acr
- [x] c2mpir
- [ ] c2ndps
- [x] cdist
- [x] ~~clip~~ (*torch.clip*)
- [ ] csm2acr
- [ ] dct
- [x] decimate
- [ ] delay
- [ ] delta
- [ ] dequantize
- [ ] df2
- [ ] dfs
- [x] ~~dmp~~
- [ ] dtw
- [x] ~~dtw_merge~~
- [ ] entropy
- [ ] excite
- [x] ~~extract~~
- [ ] fbank
- [x] ~~fd~~
- [x] ~~fdrw~~
- [x] ~~fft~~ (*torch.fft.fft*)
- [x] ~~fft2~~ (*torch.fft.fft2*)
- [x] fftcep
- [x] ~~fftr~~ (*torch.fft.rfft*)
- [x] ~~fftr2~~ (*torch.fft.rfft2*)
- [x] frame
- [x] freqt
- [x] ~~glogsp~~
- [x] ~~gmm~~
- [x] ~~gmmp~~
- [ ] gnorm
- [x] ~~gpolezero~~
- [x] ~~grlogsp~~
- [ ] grpdelay
- [x] ~~gseries~~
- [x] ~~gspecgram~~
- [x] ~~gwave~~
- [x] ~~histogram~~ (*torch.histogram*)
- [x] ~~huffman~~
- [x] ~~huffman_decode~~
- [x] ~~huffman_encode~~
- [ ] idct
- [x] ~~ifft~~ (*torch.fft.ifft*)
- [x] ~~ifft2~~ (*torch.fft.ifft2*)
- [ ] ignorm
- [ ] imglsadf
- [x] impulse
- [ ] imsvq
- [x] interpolate
- [x] ipqmf
- [x] iulaw
- [ ] lar2par
- [x] ~~lbg~~
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
- [x] ~~median~~ (*torch.median*)
- [x] ~~merge~~
- [ ] mfcc
- [ ] mgc2mgc
- [ ] mgc2sp
- [ ] mgcep
- [ ] mglsadf
- [ ] mglsp2sp
- [x] ~~minmax~~
- [ ] mlpg
- [ ] mlsacheck
- [ ] mpir2c
- [ ] mseq
- [ ] msvq
- [x] ~~nan~~ (*torch.isnan*)
- [ ] ndps2c
- [ ] norm0
- [ ] nrand
- [ ] par2lar
- [ ] par2lpc
- [ ] pca
- [ ] pcas
- [ ] phase
- [ ] pitch
- [x] ~~pitch_mark~~
- [ ] poledf
- [x] pqmf
- [ ] quantize
- [x] ramp
- [x] ~~reverse~~
- [ ] rlevdur
- [x] ~~rmse~~
- [ ] root_pol
- [ ] sin
- [ ] smcep
- [ ] snr
- [x] ~~sopr~~
- [x] spec
- [x] step
- [x] ~~swab~~
- [x] ~~symmetrize~~
- [ ] train
- [x] ~~transpose~~
- [x] ulaw
- [x] ~~vc~~
- [x] ~~vopr~~
- [x] ~~vstat~~ (*torch.var_mean*)
- [x] ~~vsum~~ (*torch.sum*)
- [x] window
- [x] ~~x2x~~
- [ ] zcross
- [ ] zerodf


License
-------
This software is released under the Apache License 2.0.
