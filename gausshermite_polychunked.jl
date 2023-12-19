
export GaussHermitePolyChunkedLSF


struct GaussHermitePolyChunkedLSF <: LSFModel
    deg::Int
    bounds::Vector{<:Vector{<:Real}}
    n_chunks::Int
    order_deg::Int
end

function GaussHermitePolyChunkedLSF(;deg::Int=0, bounds::Vector{<:Vector{<:Real}}, n_chunks::Int, order_deg::Int)
    @assert length(bounds) >= deg + 1
    return GaussHermitePolyChunkedLSF(deg, bounds, n_chunks, order_deg)
end

# Primary build method
function build(lsf::GaussHermitePolyChunkedLSF, coeffs::Vector{<:Vector{<:Real}}, λlsf::AbstractVector{<:Real}, chunks::AbstractVector{<:Real}, j::Int; zero_centroid=nothing)
    # create polynomial for sigmas across the order
    σpoly=Polynomial(coeffs[:,1])

    # option 1 - work the polynomial in wavelength space - con: does not translate across orders well --> normalize by center of order wavelength
    # find the center wavelength of the order's chunks in wavelength; probably an easier way to do this
    #if iseven(lsf.n_chunks)
    #   jmid = lsf.n_chunks/2
    #   λmid = (chunks[jmid][2]+chunks[jmid+1][1])/2
    #else
    #   jmid = floor(lsf.n_chunks/2)
    #   λmid = (chunks[jmid][1]+chunks[jmid][2])/2
    # find the wavelength center of the chunk we are working on
    # λchunk = (chunks[j][2]+chunks[j][1])/2
    # σ = evaluate(σpoly,(λchunk-λmid)/λmid)
    
    # option 2 - work the polynomial in chunk number space - pro - can drop passing the chunks array; con - does not translate well when changing n_chunks --> normalize by n_chunks from 0..1
    σ = evaluate(σpoly,j/lsf.n_chunks)
    herm = gauss_hermite(λlsf ./ σ, lsf.deg)
    kernel = herm[:, 1]
    if lsf.deg == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end

    for k=2:lsf.deg+1
        σpoly=Polynomial(coeffs[:,k])
        # option 1 - work in λchunk space
        # hk = evaluate(polyk,(λchunk-λmid)/λmid)
        # option 2 - work in n_chunks space for the polynomial, instead of lambda; simpler
        hk = evaluate(polyk,j/lsf.n_chunks)
        kernel .+=  hk .* herm[:, k]
    end
    if isnothing(zero_centroid)
        zero_centroid = lsf.deg > 0
    end
    if zero_centroid
        λcen = sum(abs.(kernel) .* λlsf) ./ sum(abs.(kernel))
        return build(lsf, coeffs, λlsf .+ λcen, zero_centroid=false)
    end
    kernel ./= sum(kernel)
    return kernel
end

# API Build method
function build(lsf::GaussHermitePolyChunkedLSF, templates::Dict{String, Any}, params::Parameters, ::SpecData1D; zero_centroid=nothing)
    kernels = Vector{Vector{Float64}}(undef, length(templates["lsf"].chunks))
    for j=1:length(kernels)
        coeffs = [[params["a_$(j)_$(i)"].value for i=0:lsf.deg] for k=0:lsf.order_deg]
        kernels[j] = build(lsf, coeffs, templates["lsf"].λlsf, templates["lsf"].chunks, j; zero_centroid) # templates["lsf"].chunks not needed if we work in n_chunks space for the polynomial for the LSF params
    end
    return kernels
end

# function gauss_hermite(x, deg)
#     herm0 = π^-0.25 .* exp.(-0.5 .* x.^2)
#     if deg == 0
#         return herm0
#     elseif deg == 1
#         herm1 = sqrt(2) .* herm0 .* x
#         return [herm0 herm1]
#     else
#         herm = zeros(length(x), deg+1)
#         herm[:, 1] .= herm0
#         herm[:, 2] .= sqrt(2) .* herm0 .* x
#         for k=3:deg+1
#             herm[:, k] .= sqrt(2 / (k - 1)) .* (x .* herm[:, k-1] .- sqrt((k - 2) / 2) .* herm[:, k-2])
#         end
#         return herm
#     end
# end

function initialize_params!(params::Parameters, lsf::GaussHermitePolyChunkedLSF, templates::Dict{String, Any}, ::SpecData1D)
    for j=0:length(templates["lsf"].order_deg) # was .chunks; changed 1 to 0.
        for i=0:lsf.deg
            # decay/tighten bounds as one goes to higher order terms in the polynomial for each deg of the hermite polynomial, on general principle as you go to higher order terms in a polynomial fits the coefficients shrink because the values are getting raised to the jth power. May not work.  
            hardness=1.0 # adjust to make the polynomial coefficient bounds shrink faster or slower
            if j == 0
                decayfactor=1
            else
                decayfactor = 1/(hardness*j)  
            end
            t = lsf.bounds[i+1].^decayfactor
            lbi = t[1]
            ubi = t[2]
            ai = (lbi + ubi) / 2
            if ai == 0
                ai = 0.1 * (ubi - lbi)
            end
            params["a_$(j)_$(i)"] = Parameter(value=ai, lower_bound=lbi, upper_bound=ubi)
        end
    end
    return params
end


function initialize_templates!(templates::Dict{String, Any}, lsf::GaussHermitePolyChunkedLSF, data::SpecSeries1D)
    δλ = templates["λ"][3] - templates["λ"][2]
    λlsf = get_lsfkernel_λgrid(lsf.bounds[1][2] * 2.355, δλ)
    chunks, n_distrust = generate_chunks(lsf, templates["λ"], data)
    templates["lsf"] = (;λlsf, chunks, n_distrust)
    return templates
end

function generate_chunks(lsf::GaussHermiteChunkedLSF, λ, data)
    λi, λf = get_data_λ_bounds(data)
    λi -= 0.2
    λf += 0.2
    xi, xf = nanargmin(abs.(λ .- λi)), nanargmin(abs.(λ .- λf))
    nx = xf - xi + 1
    δλ = λ[3] - λ[2]
    Δλ = 10 * lsf.bounds[1][2]
    n_distrust = Int(ceil(Δλ / δλ / 2))
    chunks = []
    chunk_overlap = Int(round(2.5 * n_distrust))
    chunk_width = Int(round(nx / lsf.n_chunks + chunk_overlap))
    push!(chunks, (xi, Int(round(min(xi + chunk_width, xf)))))
    if chunks[1][2] == xf
        return chunks
    end
    for i=2:nx
        _xi = chunks[i-1][2] - chunk_overlap
        _xf = Int(floor(min(_xi + chunk_width, xf)))
        push!(chunks, (_xi, _xf))
        if _xf == xf
            break
        end
    end
    if (chunks[end][2] - chunks[end][1]) <= chunk_width / 2
        deleteat!(chunks, lastindex(chunks))
        deleteat!(chunks, lastindex(chunks))
        _xi = chunks[end][2] - chunk_overlap
        push!(chunks, (_xi, xf))
    end
    chunks[1] = (1, chunks[1][2])
    chunks[end] = (chunks[end][1], length(λ))
    return chunks, n_distrust
end

enforce_positivity(lsf::GaussHermitePolyChunkedLSF) = lsf.deg > 0

function convolve_spectrum(lsf::GaussHermitePolyChunkedLSF, templates::Dict{String, Any}, params::Parameters, model_spec::AbstractVector{<:Real}, data::SpecData1D)
    kernels = build(lsf, templates, params, data)
    model_specc = fill(NaN, length(model_spec), length(templates["lsf"].chunks))
    for j=1:length(templates["lsf"].chunks)
        xi, xf = templates["lsf"].chunks[j]
        model_specc[xi:xf, j] .= convolve1d(view(model_spec, xi:xf), kernels[j])
        k1 = xi + templates["lsf"].n_distrust - 1
        k2 = xf - templates["lsf"].n_distrust + 1
        model_specc[1:k1, j] .= NaN
        model_specc[k2:end, j] .= NaN
    end
    model_specc = nanmean(model_specc, dim=2)
    return model_specc
end