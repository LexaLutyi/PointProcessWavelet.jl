function points_to_field(x::AbstractArray{T}, wp::WaveletParams) where T <: AbstractFloat
    ws = wp.ws
    n = size(x, 2)
    N = length(ws)
    
    a = x[1, :]
    b = x[2, :]
    aw = a * ws'
    bw = b * ws'
    c = reshape(aw, n, 1, N) .+ reshape(bw, n, N, 1)

    m = sum(t -> cis(-T(2π) * t), c, dims=1)
    reshape(m, N, N)
end


function phase_harmonics(z::T, k::Int) where T
    if k < 0
        ph(conj.(z), -k)
    elseif k == 0
        T(@. abs(z))
    elseif k == 1
        copy(z)
    elseif k == 2
        @. z * z / abs(z)
    elseif k == 3
        @. z ^ 3 / abs2(z)
    else
        if k % 2 == 0
            @. z ^ k / abs(z) ^ (k - 1)
        else
            @. z ^ k / abs2(z) ^ (k ÷ 2)
        end
    end
end


all_phase_harmonics(w_jl, K) = mapreduce(k -> phase_harmonics.(w_jl, k), vcat, 0:K - 1)


function wavelet_phase_harmonics(μ, wp, vs)
    M = points_to_field(μ, wp)
    
    w_jl = map(Ψ -> ifft(M .* Ψ), wp.Ψs)
    w_jlk = map((w, v) -> w .- v, all_phase_harmonics(w_jl, wp.K), vs[2])
    
    m = ifft(M) .- vs[1]
    
    m, w_jlk
end


function v_λ_k_all(μ, wp::WaveletParams{TF}) where TF <: AbstractFloat
    m, w_jlk = wavelet_phase_harmonics(μ, wp, (zero(TF), zeros(TF, wp.J * wp.L * wp.K)))
    mean(m), mean.(w_jlk)
end


cross_correlation(x, y::AbstractArray{<:Complex}) = mean(@. x * conj(y))
cross_correlation(x, y::AbstractArray{<:Real}) = mean(@. x * y)


cc_subset(w1, w2, ix) = map(ix) do (i, j)
    cross_correlation(w1[i], w2[j])
end


cc_subset(w1, w2, ix::Vector{Tuple{Int, Int, Tuple{Int, Int}}}) = map(ix) do (i, j, shift)
    cross_correlation(w1[i], circshift(w2[j], shift))
end


function K_all(μ, wp, vs)
    m, w = wavelet_phase_harmonics(μ, wp, vs)

    c1 = cc_subset(w, w, wp.ix)
    c2 = cc_subset(w, w, wp.ix_shift)
    c3 = cc_subset([m], w, wp.ix_0)

    [c1; c2; c3]
end


function K_all(x, y, wp, vs_x, vs_y)
    mx, wx = wavelet_phase_harmonics(x, wp, vs_x)
    my, wy = wavelet_phase_harmonics(y, wp, vs_y)

    cxy1 = cc_subset(wx, wy, wp.ix)
    cxy2 = cc_subset(wx, wy, wp.ix_shift)
    cxy3 = cc_subset([mx], wy, wp.ix_0)

    [cxy1; cxy2; cxy3]
end


ww_subset(w1, w2, ix) = map(ix) do (i, j)
    x = w1[i]
    y = w2[j]
    std(@. x * conj(y))
end


ww_subset(w1, w2, ix::Vector{Tuple{Int, Int, Tuple{Int, Int}}}) = map(ix) do (i, j, shift)
    x = w1[i]
    y = circshift(w2[j], shift)
    std(@. x * conj(y))
end


function W_all(μ, wp, vs)
    m, w = wavelet_phase_harmonics(μ, wp, vs)

    c1 = ww_subset(w, w, wp.ix)
    c2 = ww_subset(w, w, wp.ix_shift)
    c3 = ww_subset([m], w, wp.ix_0)

    [c1; c2; c3]
end


function W_all(x, y, wp, vs_x, vs_y)
    mx, wx = wavelet_phase_harmonics(x, wp, vs_x)
    my, wy = wavelet_phase_harmonics(y, wp, vs_y)

    cxy1 = ww_subset(wx, wy, wp.ix)
    cxy2 = ww_subset(wx, wy, wp.ix_shift)
    cxy3 = ww_subset([mx], wy, wp.ix_0)

    [cxy1; cxy2; cxy3]
end
