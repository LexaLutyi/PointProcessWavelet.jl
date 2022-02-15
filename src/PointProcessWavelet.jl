module PointProcessWavelet

using StaticArrays: SVector
using Statistics: mean, std
using FFTW: fft, ifft, fftshift, ifftshift, fftfreq
using LinearAlgebra: I, norm
using Distributions: MvNormal, pdf

include("wavelets.jl")
include("wavelet_params.jl")
include("wavelet_functions.jl")

export WaveletParams
export v_λ_k_all, K_all, W_all

end
