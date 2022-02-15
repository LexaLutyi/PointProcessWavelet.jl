using PointProcessWavelet
using Test

@testset "PointProcessWavelet.jl" begin
    x0 = rand(2, 100)
    wp = WaveletParams(0.5, 128, 3, 8, 2, 1 / 128)

    vs = v_λ_k_all(x0, wp)
    k0 = K_all(x0, wp, vs)
    w0 = W_all(x0, wp, vs)

    k1 = K_all(x0, x0, wp, vs, vs)
    w1 = W_all(x0, x0, wp, vs, vs)

    @test k0 ≈ k1
    @test w0 ≈ w1
end
