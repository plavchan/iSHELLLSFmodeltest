
export GaussHermiteLSF

"""
An LSF model determined by a summation of Gaussian functions with identical widths and Hermite Polynomials as coefficients.

# Fields
- `deg::Int`: The degree of the Hermite-Gaussian function. `deg=0` corresponds to a single Gaussian.
- `bounds::Vector{Vector{Float64}}`: The bounds on each of the the LSF coeffs.
- `zero_centroid::Bool` Whether or not to force the centroid of the kernel to be zero in order to lift the degeneracy between the LSF and wavelength solution.

# Constructors
    GaussHermiteLSF(;deg::Int, bounds::NamedTuple)
"""
struct GaussHermiteLSF <: LSFModel
    deg::Int
    bounds::Vector{<:Vector{<:Real}}
end

function GaussHermiteLSF(;deg::Int=0, bounds::Vector{<:Vector{<:Real}})
    @assert length(bounds) >= deg + 1
    return GaussHermiteLSF(deg, bounds)
end

# Primary build method
function build(lsf::GaussHermiteLSF, coeffs::Vector{<:Real}, λlsf::AbstractVector{<:Real}; zero_centroid=nothing)
    σ = coeffs[1]
    herm = gauss_hermite(λlsf ./ σ, lsf.deg)
    kernel = herm[:, 1]
    if lsf.deg == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end
    for k=2:lsf.deg+1
        kernel .+= coeffs[k] .* herm[:, k]
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
function build(lsf::GaussHermiteLSF, templates::Dict{String, Any}, params::Parameters, ::SpecData1D; zero_centroid=nothing)
    coeffs = [params["a$i"].value for i=0:lsf.deg]
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

function initialize_params!(params::Parameters, lsf::GaussHermiteLSF, ::Dict{String, Any}, ::SpecData1D)
    for i=0:lsf.deg
        t = lsf.bounds[i+1]
        lbi = t[1]
        ubi = t[2]
        ai = (lbi + ubi) / 2
        if ai == 0
            ai = 0.1 * (ubi - lbi)
        end
        params["a$i"] = Parameter(value=ai, lower_bound=lbi, upper_bound=ubi)
    end

    # Return
    return params
end

function initialize_templates!(templates::Dict{String, Any}, lsf::GaussHermiteLSF, ::SpecSeries1D)
    δλ = templates["λ"][3] - templates["λ"][2]
    λlsf = get_lsfkernel_λgrid(lsf.bounds[1][2] * 2.355, δλ)
    templates["lsf"] = (;λlsf)
    return templates
end

enforce_positivity(lsf::GaussHermiteLSF) = true