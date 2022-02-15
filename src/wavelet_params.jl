struct WaveletParams{TF, Tw}
    s::TF
    N::Int   
    J::Int
    L::Int
    K::Int
    σ::TF

    ws::Tw
    
    Ψs::Vector{Matrix{TF}}

    Γ_H::Vector{NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int, Int, TF, Int, Int, 
        Int, TF, Int, Tuple{Int, Int}}}
        }
    
    ix::Vector{Tuple{Int, Int}}
    ix_shift::Vector{Tuple{Int, Int, Tuple{Int, Int}}}
    ix_0::Vector{Tuple{Int, Int}}
end


function WaveletParams(s, N, J, L, K, σ, Γ_H = default_Γ_H(J, L, K, θtype=typeof(s)))
    dx = 2s / N
    xs = range(-s; length=N, step=dx)
    ws = fftfreq(N, 1 / dx)

    g = MvNormal([0, 0], σ^2 * I)
    gaus = [pdf(g, [x, y]) for x in xs, y in xs]
    Gaus = gaus |> ifftshift |> fft |> real

    float_type = typeof(s)

    # wavelet_matrices(ws, J, L; ξ0=maximum(ws) / 2, c=1) =
    ξ0 = maximum(ws) / 2
    c = 1
    wavelet_matrices = [wavelet_matrix(ws, j, θ; ξ0, L, c) for j in 0:J - 1, θ in range(0., length=L, step=2π / L)]
    Ψs = map(Ψ -> float_type.(Ψ .* Gaus), reshape(wavelet_matrices, :))

    ix, ix_shift, ix_0, perm_ix, perm_ix_shift = get_ix_subsets_from_Γ_H(Γ_H, J, L, K)

    WaveletParams(s, N, J, L, K, σ, ws, Ψs, Γ_H, ix, ix_shift, ix_0)
end


e_θ(l, L) = (cos(π / 2 + 2π * l / L), sin(π / 2 + 2π * l / L))
τ_θ_j(l, j_, L) = round.(Int, 2^j_ .* e_θ(l, L))


function default_Γ_H(J, L, K; θtype=Float64)
    Γ_H = NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int, Int, θtype, Int, Int, 
        Int, θtype, Int, Tuple{Int, Int}}
    }[]
    p = Iterators.product(
        0:L - 1, 
        0:L - 1,
        0:J - 1,
        0:J - 1,
        [false, true],
        0:K - 1,
        0:K - 1
    )

    for (l, l_, j, j_, isshift, k, k_) in p
        l, l_, j, j_, isshift, k, k_
        if l_ > l
            continue
        end
        if (j > j_) || (j < j_ - 2)
            continue
        end
        if k > min(k_, 1)
            continue
        end

        if isshift
            τ_ = τ_θ_j(l, j_, L)
        else
            τ_ = (0, 0)
        end

        θ = 2π * l / L
        θ_ = 2π * l_ / L
        dl = abs(l - l_)

        if j == j_
            if k == 0 && k_ > 1
                continue
            end
            if k == 1
                if k_ != 1
                    continue
                elseif dl > 2
                    continue
                end
            end
        else
            if k == 0 && k_ == 4
                continue
            end
            if k == 1
                if k_ != 2^(j_ - j)
                    continue
                elseif dl > 2
                    continue
                end
            end
        end

        push!(Γ_H, (;j, l, θ, k, j_, l_, θ_, k_, τ_))
    end
    Γ_H
end


function get_ix_from_Γ_H(p, J, L)
    ix1 = (p.j + 1) + J * (p.l) + J * L * (p.k)
    ix2 = (p.j_ + 1) + J * (p.l_) + J * L * (p.k_)
    (p.τ_ == (0, 0), ix1, ix2, p.τ_)
end


function get_ix_subsets_from_Γ_H(Γ_H, J, L, K)
    ix_all = map(p -> get_ix_from_Γ_H(p, J, L), Γ_H)

    perm_ix = findall(s -> s[1] == true, ix_all)
    ix = map(s -> s[2:3], ix_all[perm_ix])

    perm_ix_shift = findall(s -> s[1] == false, ix_all)
    ix_shift = map(s -> s[2:4], ix_all[perm_ix_shift])

    ix_0 = map(i -> (1, i), 1:J * L * K)
    ix, ix_shift, ix_0, perm_ix, perm_ix_shift
end
