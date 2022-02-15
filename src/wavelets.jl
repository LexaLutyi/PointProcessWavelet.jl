"Fourier transform of 2D Wavelet"
function Ψ(w; ξ0, L, c)
    ϕ = atan(w[2], w[1])
    t = (norm(w) - ξ0) / ξ0
    if (-π / 2 < ϕ < π / 2) && (-1 < t < 1)
        c * exp(-abs2(t) / (1 - abs2(t))) * cos(ϕ)^(L - 1)
    else
        0.
    end
end

rotate(w, θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * w
scale(w, j) = 2^j * w

"Fourier transform of 2D Wavelet with scaling and rotation"
Ψ(w, j, θ; ξ0, L, c) = Ψ(scale(rotate(w, θ), j); ξ0, L, c)

"Square matrix of wavelet in frequency domain"
wavelet_matrix(ws, j=0, θ=0.; ξ0, L, c) = 
    [Ψ(SVector(w1, w2), j, θ; ξ0, L, c) for w1 in ws, w2 in ws]
