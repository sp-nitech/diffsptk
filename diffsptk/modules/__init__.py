# ------------------------------------------------------------------------ #
# Copyright 2022 SPTK Working Group                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

from .acorr import Autocorrelation
from .acr2csm import AutocorrelationToCompositeSinusoidalModelCoefficients
from .alaw import ALawCompression
from .ap import Aperiodicity
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .c2acr import CepstrumToAutocorrelation
from .c2mpir import CepstrumToMinimumPhaseImpulseResponse
from .c2ndps import CepstrumToNegativeDerivativeOfPhaseSpectrum
from .cdist import CepstralDistance
from .chroma import ChromaFilterBankAnalysis
from .cqt import ConstantQTransform
from .cqt import ConstantQTransform as CQT
from .csm2acr import CompositeSinusoidalModelCoefficientsToAutocorrelation
from .dct import DiscreteCosineTransform
from .dct import DiscreteCosineTransform as DCT
from .decimate import Decimation
from .delay import Delay
from .delta import Delta
from .dequantize import InverseUniformQuantization
from .df2 import SecondOrderDigitalFilter
from .dfs import InfiniteImpulseResponseDigitalFilter
from .dfs import InfiniteImpulseResponseDigitalFilter as IIR
from .dht import DiscreteHartleyTransform
from .dht import DiscreteHartleyTransform as DHT
from .drc import DynamicRangeCompression
from .drc import DynamicRangeCompression as DRC
from .dst import DiscreteSineTransform
from .dst import DiscreteSineTransform as DST
from .dtw import DynamicTimeWarping
from .dtw import DynamicTimeWarping as DTW
from .entropy import Entropy
from .excite import ExcitationGeneration
from .fbank import MelFilterBankAnalysis
from .fbank import MelFilterBankAnalysis as FBANK
from .fftcep import CepstralAnalysis
from .fftr import RealValuedFastFourierTransform
from .flux import Flux
from .frame import Frame
from .freqt import FrequencyTransform
from .freqt2 import SecondOrderAllPassFrequencyTransform
from .gammatone import GammatoneFilterBankAnalysis
from .gmm import GaussianMixtureModeling
from .gmm import GaussianMixtureModeling as GMM
from .gnorm import GeneralizedCepstrumGainNormalization
from .griffin import GriffinLim
from .grpdelay import GroupDelay
from .hilbert import HilbertTransform
from .hilbert2 import TwoDimensionalHilbertTransform
from .histogram import Histogram
from .ialaw import ALawExpansion
from .ica import IndependentComponentAnalysis
from .ica import IndependentComponentAnalysis as ICA
from .icqt import InverseConstantQTransform
from .icqt import InverseConstantQTransform as ICQT
from .idct import InverseDiscreteCosineTransform
from .idct import InverseDiscreteCosineTransform as IDCT
from .idht import InverseDiscreteHartleyTransform
from .idht import InverseDiscreteHartleyTransform as IDHT
from .idst import InverseDiscreteSineTransform
from .idst import InverseDiscreteSineTransform as IDST
from .ifbank import InverseMelFilterBankAnalysis
from .ifbank import InverseMelFilterBankAnalysis as IFBANK
from .ifftr import RealValuedInverseFastFourierTransform
from .ifreqt2 import SecondOrderAllPassInverseFrequencyTransform
from .igammatone import GammatoneFilterBankSynthesis
from .ignorm import GeneralizedCepstrumInverseGainNormalization
from .imdct import InverseModifiedDiscreteCosineTransform
from .imdct import InverseModifiedDiscreteCosineTransform as IMDCT
from .imdst import InverseModifiedDiscreteSineTransform
from .imdst import InverseModifiedDiscreteSineTransform as IMDST
from .imglsadf import PseudoInverseMGLSADigitalFilter
from .imglsadf import PseudoInverseMGLSADigitalFilter as IMLSA
from .imsvq import InverseMultiStageVectorQuantization
from .interpolate import Interpolation
from .ipnorm import MelCepstrumInversePowerNormalization
from .ipqmf import PseudoQuadratureMirrorFilterBankSynthesis
from .ipqmf import PseudoQuadratureMirrorFilterBankSynthesis as IPQMF
from .is2par import InverseSineToParcorCoefficients
from .istft import InverseShortTimeFourierTransform
from .istft import InverseShortTimeFourierTransform as ISTFT
from .iulaw import MuLawExpansion
from .ivq import InverseVectorQuantization
from .lar2par import LogAreaRatioToParcorCoefficients
from .lbg import LindeBuzoGrayAlgorithm
from .lbg import LindeBuzoGrayAlgorithm as LBG
from .levdur import LevinsonDurbin
from .linear_intpl import LinearInterpolation
from .lpc import LinearPredictiveCodingAnalysis
from .lpc import LinearPredictiveCodingAnalysis as LPC
from .lpc2lsp import LinearPredictiveCoefficientsToLineSpectralPairs
from .lpc2par import LinearPredictiveCoefficientsToParcorCoefficients
from .lpccheck import LinearPredictiveCoefficientsStabilityCheck
from .lsp2lpc import LineSpectralPairsToLinearPredictiveCoefficients
from .lsp2sp import LineSpectralPairsToSpectrum
from .lspcheck import LineSpectralPairsStabilityCheck
from .magic_intpl import MagicNumberInterpolation
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients
from .mcpf import MelCepstrumPostfiltering
from .mdct import ModifiedDiscreteCosineTransform
from .mdct import ModifiedDiscreteCosineTransform as MDCT
from .mdst import ModifiedDiscreteSineTransform
from .mdst import ModifiedDiscreteSineTransform as MDST
from .mfcc import MelFrequencyCepstralCoefficientsAnalysis
from .mfcc import MelFrequencyCepstralCoefficientsAnalysis as MFCC
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum
from .mgc2sp import MelGeneralizedCepstrumToSpectrum
from .mcep import MelCepstralAnalysis
from .mgcep import MelGeneralizedCepstralAnalysis
from .mglsadf import PseudoMGLSADigitalFilter
from .mglsadf import PseudoMGLSADigitalFilter as MLSA
from .mlpg import MaximumLikelihoodParameterGeneration
from .mlpg import MaximumLikelihoodParameterGeneration as MLPG
from .mlsacheck import MLSADigitalFilterStabilityCheck
from .mpir2c import MinimumPhaseImpulseResponseToCepstrum
from .msvq import MultiStageVectorQuantization
from .ndps2c import NegativeDerivativeOfPhaseSpectrumToCepstrum
from .nmf import NonnegativeMatrixFactorization
from .nmf import NonnegativeMatrixFactorization as NMF
from .norm0 import AllPoleToAllZeroDigitalFilterCoefficients
from .oband import FractionalOctaveBandAnalysis
from .par2is import ParcorCoefficientsToInverseSine
from .par2lar import ParcorCoefficientsToLogAreaRatio
from .par2lpc import ParcorCoefficientsToLinearPredictiveCoefficients
from .pca import PrincipalComponentAnalysis
from .pca import PrincipalComponentAnalysis as PCA
from .phase import Phase
from .pitch import Pitch
from .pitch_spec import PitchAdaptiveSpectralAnalysis
from .plp import PerceptualLinearPredictiveCoefficientsAnalysis
from .plp import PerceptualLinearPredictiveCoefficientsAnalysis as PLP
from .pnorm import MelCepstrumPowerNormalization
from .pol_root import RootsToPolynomial
from .poledf import AllPoleDigitalFilter
from .pqmf import PseudoQuadratureMirrorFilterBankAnalysis
from .pqmf import PseudoQuadratureMirrorFilterBankAnalysis as PQMF
from .quantize import UniformQuantization
from .rlevdur import ReverseLevinsonDurbin
from .rmse import RootMeanSquareError
from .rmse import RootMeanSquareError as RMSE
from .root_pol import PolynomialToRoots
from .smcep import SecondOrderAllPassMelCepstralAnalysis
from .snr import SignalToNoiseRatio
from .snr import SignalToNoiseRatio as SNR
from .spec import Spectrum
from .stft import ShortTimeFourierTransform
from .stft import ShortTimeFourierTransform as STFT
from .ulaw import MuLawCompression
from .unframe import Unframe
from .vq import VectorQuantization
from .wht import WalshHadamardTransform
from .wht import WalshHadamardTransform as WHT
from .wht import WalshHadamardTransform as InverseWalshHadamardTransform
from .wht import WalshHadamardTransform as IWHT
from .window import Window
from .world_synth import WorldSynthesis
from .yingram import Yingram
from .zcross import ZeroCrossingAnalysis
from .zerodf import AllZeroDigitalFilter
