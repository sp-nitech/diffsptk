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


License
-------
This software is released under the Apache License 2.0.
