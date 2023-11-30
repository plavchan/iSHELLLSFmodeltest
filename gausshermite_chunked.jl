export GaussHermiteChunkedLSF

"""
An LSF model determined by a summation of Gaussian functions with identical widths and Hermite Polynomials as coefficients.  The LSF is chunked into N chunks across the order to allow for a variable LSF while not 

# Fields
- `deg::Int`: The degree of the Hermite-Gaussian function. `deg=0` corresponds to a single Gaussian.
- `bounds::Vector{Vector{Float64}}`: The bounds on each of the the LSF coeffs.
- `zero_centroid::Bool` Whether or not to force the centroid of the kernel to be zero in order to lift the degeneracy between the LSF and wavelength solution.
- `Nchunks::Int:` the number of chunks for the LSF model

# Constructors
    GaussHermiteChunkedLSF(;deg::Int, bounds::NamedTuple)
"""
struct GaussHermiteChunkedLSF <: LSFModel
    deg::Int
    bounds::Vector{<:Vector{<:Real}}
    Nchunks::Int
end

function GaussHermiteChunkedLSF(;deg::Int=0, bounds::Vector{<:Vector{<:Real}}, Nchunks::Int=1)
    @assert length(bounds) >= deg + 1
    return GaussHermiteChunkedLSF(deg, bounds,Nchunks)
end

# Primary build method
function build(lsf::GaussHermiteChunkedLSF, coeffs::Vector{Vector{<:Real}}, λlsf::AbstractVector{<:Real}; zero_centroid=nothing)
    kernelarray = Vector{Vector{Float64}}(undef, lsf.Nchunks)
    for j in range(1, lsf.Nchunks)
        σ = coeffs[j][1]
        herm = gauss_hermite(λlsf[j] ./ σ, lsf.deg)
        kernelarray[j] = herm[:, 1]
        if lsf.deg == 0  # just a Gaussian
            return kernelarray[j] ./ sum(kernelarray[j])
        end
        for k=2:lsf.deg+1
            kernelarray[j] .+= coeffs[j][k] .* herm[:, k]
        end
        kernelarray[j] ./= sum(kernelarray[j])
    end
    if isnothing(zero_centroid)
        zero_centroid = lsf.deg > 0
    end

    if zero_centroid
        λlsfshifted = deepcopy(λlsf)
        for j in range(1, lsf.Nchunks)
            λcen = sum(abs.(kernelarray[j]) .* λlsf[j]) ./ sum(abs.(kernelarray[j]))
            λlsfshifted[j] .+= λcen
        end 
     	  return build(lsf, coeffs, λlsfshifted, zero_centroid=false)
    end
    return kernelarray
end

# API Build method
function build(lsf::GaussHermiteChunkedLSF, templates::Dict{String, Any}, params::Parameters, ::SpecData1D; zero_centroid=nothing)
    coeffs = Vector{Vector{Float64}}(undef, lsf.Nchunks)
    # this pulls the named variables out of the params array
    for j in range(1, lsf.Nchunks)
        coeffs[j] = [params["a$j$i"].value for i=0:lsf.deg]
    end
    return build(lsf, coeffs, templates["lsf"].λlsf; zero_centroid)
end

function gauss_hermite(x, deg)
    herm0 = π^-0.25 .* exp.(-0.5 .* x.^2)
    if deg == 0
        return herm0
    elseif deg == 1
        herm1 = sqrt(2) .* herm0 .* x
        return [herm0 herm1]
    else
        herm = zeros(length(x), deg+1)
        herm[:, 1] .= herm0
        herm[:, 2] .= sqrt(2) .* herm0 .* x
        for k=3:deg+1
            herm[:, k] .= sqrt(2 / (k - 1)) .* (x .* herm[:, k-1] .- sqrt((k - 2) / 2) .* herm[:, k-2])
        end
        return herm
    end
end

function initialize_params!(params::Parameters, lsf::GaussHermiteChunkedLSF, ::Dict{String, Any}, ::SpecData1D)
    for j=1:lsf.Nchunks
        for i=0:lsf.deg
            t = lsf.bounds[i+1] # if needed in future, swap this to lsf.bounds[j][i+1]
            lbi = t[1]
            ubi = t[2]
            ai = (lbi + ubi) / 2
            if ai == 0
                ai = 0.1 * (ubi - lbi)
            end
            params["a$j$i"] = Parameter(value=ai, lower_bound=lbi, upper_bound=ubi)
        end
    end

    # Return
    return params
end

function initialize_templates!(templates::Dict{String, Any}, lsf::GaussHermiteChunkedLSF, ::SpecSeries1D)
    δλ = templates["λ"][3] - templates["λ"][2]
    λlsf = get_lsfkernel_λgrid(lsf.bounds[1][2] * 2.355, δλ)
    templates["lsf"] = (;λlsf = [copy(λlsf) for l in lsf.Nchunks])
    return templates
end

enforce_positivity(lsf::GaussHermiteChunkedLSF) = true


function convolve_spectrum(lsf::GaussHermiteChunkedLSF, templates::Dict{String, Any}, params::Parameters, model_spec::AbstractVector{<:Real}, data::SpecData1D)
    kernelarray = build(lsf, templates, params, data)
    model_specc_chunk = Vector{Float64}(undef, lsf.Nchunks)
    datalen = length(data.spec)
    padding = # chunkpadding value
    # chunks overlap by small amount, and average in the overlap
    for j=1:lsf.Nchunks
       chunki = 
       chunkf = 
       # enforce checking that not going past edges of array with chunks
       model_spec_chunk = model_spec[chunki:chunkf]
       model_specc_chunk[j] = convolve1d(model_spec_chunk, kernelarray[j])
    end
    model_specc = #reconstructed from the chunks.  average in overlap.
    return model_specc
end


