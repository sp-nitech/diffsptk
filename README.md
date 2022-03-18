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
stft = diffsptk.STFT(frame_legnth=12, frame_period=10, fft_length=16)
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
- [ ] ~~aeq~~
- ~~[ ] amgcep~~
- [ ] average
- [x] b2mc
- [ ] bcp
- [ ] bcut
- [ ] c2acr
- [x] c2mpir
- [ ] c2ndps
- [x] cdist
- [ ] clip
- [ ] csm2acr
- [ ] dct
- [x] decimate
- [ ] delay
- [ ] delta
- [ ] dequantize
- [ ] df2
- [ ] dfs
- [ ] dmp
- [ ] dtw
- [ ] dtw_merge
- [ ] entropy
- [ ] excite
- [ ] extract
- [ ] fbank
- [ ] fd
- [ ] fdrw
- [ ] fft
- [ ] fft2
- [x] fftcep
- [ ] fftr
- [ ] fftr2
- [x] frame
- [x] freqt
- [ ] glogsp
- [ ] gmm
- [ ] gmmp
- [ ] gnorm
- [ ] gpolezero
- [ ] grlogsp
- [ ] grpdelay
- [ ] gseries
- [ ] gspecgram
- [ ] gwave
- [ ] histogram
- [ ] huffman
- [ ] huffman_decode
- [ ] huffman_encode
- [ ] idct
- [ ] ifft
- [ ] ifft2
- [ ] ignorm
- [ ] imglsadf
- [ ] impulse
- [ ] imsvq
- [ ] interpolate
- [x] ipqmf
- [x] iulaw
- [ ] lar2par
- [ ] lbg
- [x] levdur
- [ ] linear_intpl
- [ ] lpc
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
- [ ] median
- [ ] merge
- [ ] mfcc
- [ ] mgc2mgc
- [ ] mgc2sp
- [ ] mgcep
- [ ] mglsadf
- [ ] mglsp2sp
- [ ] minmax
- [ ] mlpg
- [ ] mlsacheck
- [ ] mpir2c
- [ ] mseq
- [ ] msvq
- [ ] nan
- [ ] ndps2c
- [ ] norm0
- [ ] nrand
- [ ] par2lar
- [ ] par2lpc
- [ ] pca
- [ ] pcas
- [ ] phase
- [ ] pitch
- [ ] pitch_mark
- [ ] poledf
- [x] pqmf
- [ ] quantize
- [ ] ramp
- [ ] reverse
- [ ] rlevdur
- [ ] rmse
- [ ] root_pol
- [ ] sin
- [ ] smcep
- [ ] snr
- [ ] sopr
- [ ] spec
- [ ] step
- [ ] swab
- [ ] symmetrize
- [ ] train
- [ ] transpose
- [x] ulaw
- [ ] vc
- [ ] vopr
- [ ] vstat
- [ ] vsum
- [x] window
- [ ] x2x
- [ ] zcross
- [ ] zerodf


License
-------
This software is released under the Apache License 2.0.
