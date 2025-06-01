module CUDAHist

export CUDAHist1D, nentries, sumw2, binerrors, bincounts, binedges, bincenters, integral, atomic_push!, append!, push!, normalize, cumulative, nbins

using CUDA
using Base.Threads: SpinLock
using StatsBase
using Statistics
using FHist

_sturges(x) = StatsBase.sturges(length(x))

mutable struct CUDAHist1D{T<:Union{Int32, UInt32, Float32}} <: AbstractHistogram{T,1,CuArray{T,1}}
    binedges::CuArray{Float32,1}
    bincounts::CuArray{T,1}
    sumw2::CuArray{Float32,1}
    nentries::Base.RefValue{Int32}
    overflow::Bool
    hlock::SpinLock
    function CUDAHist1D(;
        counttype::Type{T}=Float32,
        binedges,
        bincounts=CUDA.zeros(counttype, length(binedges) - 1),
        sumw2=zero(bincounts),
        nentries=Int32(0),
        overflow=false) where {T}

        return new{T}(binedges, bincounts, sumw2, Ref(round(Int32, nentries)), overflow, SpinLock())
    end

    function CUDAHist1D(ary::E;
        counttype::Type{T}=Float32,
        binedges=nothing,
        weights=nothing,
        nbins=nothing,
        overflow=false) where {T,E<:NTuple{1,Any}}

        length(ary) == 1 || throw(DimensionMismatch("Data must be a tuple of 1 vectors"))
        isnothing(weights) || length(ary[1]) == length(weights) || throw(DimensionMismatch("Data and weights must have the same length"))

        binedges = if !isnothing(binedges)
            CuArray(binedges)
        else
            auto_bins(ary, Val(1); nbins)
        end

        h = CUDAHist1D(; counttype, binedges, overflow)
        _fast_bincounts!(h, ary, binedges, weights)
        return h
    end

    function CUDAHist1D(h::Hist1D)
        binedges = CUDA.CuArray(Vector(FHist.binedges(h)))
        bincounts = CUDA.CuArray(Vector(FHist.bincounts(h)))
        sumw2 = CUDA.CuArray(FHist.sumw2(h))
        nentries = FHist.nentries(h)
        overflow = h.overflow
        return CUDAHist1D(; counttype=Float32, binedges, bincounts, sumw2, nentries, overflow)
    end

    function CUDAHist1D(ary; kws...)
        CUDAHist1D((ary, ); kws...)
    end
end

function Hist1D(h::CUDAHist1D)
    binedges = Vector(CUDAHist.binedges(h))
    bincounts = Vector(CUDAHist.bincounts(h))
    sumw2 = Vector(CUDAHist.sumw2(h))
    nentries = CUDAHist.nentries(h)
    overflow = h.overflow
    return FHist.Hist1D(; binedges, bincounts, sumw2, nentries, overflow)
end

function auto_bins(ary, ::Val{1}; nbins=nothing)
    xs = only(ary)
    E = eltype(xs)
    F = E <: Number ? float(E) : Float64
    nbins = isnothing(nbins) ? _sturges(xs) : nbins
    lo, hi = minimum(xs), maximum(xs)
    CuArray(collect(StatsBase.histrange(F(lo), F(hi), nbins)))
end

Base.lock(h::CUDAHist1D) = lock(h.hlock)
Base.unlock(h::CUDAHist1D) = unlock(h.hlock)
bincounts(h::CUDAHist1D) = h.bincounts
binedges(h::CUDAHist1D) = h.binedges
bincenters(h::CUDAHist1D) = length(h.binedges) > 1 ? (binedges(h)[1:end-1] + diff(binedges(h))./ 2) : nothing
@doc """
    nentries(h::$(CUDAHist1D))
Get the number of times a histogram is filled (`push!`ed)
"""
nentries(h::CUDAHist1D) = h.nentries[]
@doc """
    sumw2(h)
Get the sum of weights squared of the histogram, it has the same shape as `bincounts(h)`.
"""
sumw2(h::CUDAHist1D) = h.sumw2

@doc """
    binerrors(f=sqrt, h)
Get the error (uncertainty) of each bin. By default, calls `sqrt` on `sumw2(h)` bin by bin as an approximation.
"""
binerrors(f::T, h::CUDAHist1D) where T<:Function = f.(sumw2(h))
binerrors(h::CUDAHist1D) = binerrors(sqrt, h)

@doc raw"""
    effective_entries(h) -> scalar

Get the number of effective entries for the entire histogram:

```math
n_{eff} = \frac{(\sum Weights )^2}{(\sum Weight^2 )}
```

This is also equivalent to `integral(hist)^2 / sum(sumw2(hist))`, this is the same as `TH1::GetEffectiveEntries()`
"""
effective_entries(h::CUDAHist1D) = abs2(integral(h)) / sum(sumw2(h))

import Base: ==, +, -, *, /

function Base.:(==)(h1::CUDAHist1D, h2::CUDAHist1D)
    bincounts(h1) == bincounts(h2) &&
        binedges(h1) == binedges(h2) &&
        nentries(h1) == nentries(h2) &&
        sumw2(h1) == sumw2(h2) &&
        h1.overflow == h2.overflow
end

function Base.:(==)(h1::CUDAHist1D, h2::Hist1D)
        Vector(CUDAHist.bincounts(h1)) == FHist.bincounts(h2) &&
        Vector(CUDAHist.binedges(h1)) == FHist.binedges(h2) &&
        Int(CUDAHist.nentries(h1)) == FHist.nentries(h2) &&
        Vector(CUDAHist.sumw2(h1)) == FHist.sumw2(h2) &&
        h1.overflow == h2.overflow
end

for op in (:+, :-)
    @eval function ($op)(h1::CUDAHist1D, h2::CUDAHist1D)
        edge1 = h1.binedges
        edge1 != h2.binedges && throw(DimensionMismatch("Binedges don't match"))
        h1.overflow != h2.overflow && throw("Can't $op histograms with different overflow settings.")
        newcounts = broadcast($op, bincounts(h1),  bincounts(h2))

        (CUDAHist1D)(; binedges = copy.(edge1), bincounts = newcounts, sumw2 = sumw2(h1) + sumw2(h2), nentries = nentries(h1) + nentries(h2), overflow = h1.overflow)
    end
end

function Base.empty!(h1::CUDAHist1D)
    bincounts(h1) .= false
    sumw2(h1) .= false
end

# fall back one-shot implementation
function _fast_bincounts!(h::CUDAHist1D, A, binedges, weights)
    xs = A[1]
    if isnothing(weights)
        for x in xs
            push!(h, x)
        end
    else
        for (x, w) in zip(xs, weights)
            push!(h, x, w)
        end
    end
end

"""
    nbins(h::CUDAHist1D)

Get the number of bins of a histogram.
"""
function nbins(h::CUDAHist1D)
    length(bincounts(h))
end

"""
    integral(h; width=false)

Get the integral a histogram; `width` means multiply each bincount
by their bin width when calculating the integral.

!!! warning
    `width` keyword argument only works with 1D histogram at the moment.

!!! warning
    Be aware of the approximation you make
    when using `width=true` with histogram with overflow bins, the overflow
    bins (i.e. the left/right most bins) width will be taken "as is".
"""
function integral(h::CUDAHist1D; width=false)
    if width
        sum(bincounts(h) .* diff(binedges(h)))
    else
        sum(bincounts(h))
    end
end

"""
    push!(h::CUDAHist1D, val::Real, wgt::Real=1)
    atomic_push!(h::CUDAHist1D, val::Real, wgt::Real=1)

Adding one value at a time into histogram.
`sumw2` (sum of weights^2) accumulates `wgt^2` with a default weight of 1.
`atomic_push!` is a slower version of `push!` that is thread-safe.

N.B. To append multiple values at once, use broadcasting via
`push!.(h, [-3.0, -2.9, -2.8])` or `push!.(h, [-3.0, -2.9, -2.8], 2.0)`
"""
@inline function atomic_push!(h::CUDAHist1D, val::Real, wgt::Real=1)
    lock(h)
    push!(h, val, wgt)
    unlock(h)
    return nothing
end

@inline function Base.push!(h::CUDAHist1D, val::Real, wgt::Real=1)
    r = binedges(h)
    L = nbins(h)
    binidx = (val .>= r[1:end-1]) .& (val .< r[2:end])
    
    if sum(binidx) == L && h.overflow
        h.bincounts += wgt * vcat(CUDA.zeros(L-1), 1)
        h.sumw2 += wgt^2 * vcat(CUDA.zeros(L-1), 1)
        h.nentries[] += 1
    else
        h.bincounts += wgt * binidx
        h.sumw2 += wgt^2 * binidx
        if sum(binidx) != 0
            h.nentries[] += 1
        end
    end
    return nothing
end

function Base.append!(h::CUDAHist1D, val::AbstractVector, wgt::AbstractVector)
    length(val) == length(wgt) || throw(DimensionMismatch("append! to histogram expect same length values and weights"))
    lock(h)
    for (v, w) in zip(val, wgt)
        push!(h, v, w)
    end
    unlock(h)
    return h
end

function Base.append!(h::CUDAHist1D, val::AbstractVector)
    lock(h)
    for v in val
        push!(h, v)
    end
    unlock(h)
    return h
end

function _cuda_append_kernel!(data::CuDeviceArray{Float32}, 
    wgt::CuDeviceArray{Float32}, 
    binedges::CuDeviceArray{Float32}, 
    bincounts::CuDeviceMatrix{Float32},
    sumw2::CuDeviceMatrix{Float32},
    nentries::CuDeviceArray{Int32},
    overflow::Bool)

    #initialize bincounts to zero
    for i in 1:length(binedges) - 1
        @inbounds bincounts[i] = 0.0
        @inbounds sumw2[i] = 0.0
    end

    thread_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    nentries[thread_index] = 0

    for i = thread_index:stride:length(data)
        for i_bin in 1:length(binedges)-1
            if data[i] >= binedges[i_bin] && data[i] < binedges[i_bin + 1]
                @inbounds bincounts[i_bin, thread_index] += wgt[i]
                @inbounds sumw2[i_bin, thread_index] += wgt[i]^2
                @inbounds nentries[thread_index] += 1
                break
            end
        end

        if overflow && data[i] > binedges[end]
            @inbounds bincounts[end, thread_index] += wgt[i]
            @inbounds sumw2[end, thread_index] += wgt[i]^2
            @inbounds nentries[thread_index] += 1
        end
    end
    return
end

function cuda_append!(hist::CUDAHist1D, data::Vector{Float32}, Nthreads::Int; wgt::Vector{Float32}=ones(Float32, length(data)))
    if length(data) != length(wgt)
        throw(DimensionMismatch("Data and weights must have the same length"))
    end

    #Set up arrays for kernel
    dev = CUDA.device()
    max_threads = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_per_block = min(max_threads, Nthreads)
    blocks = cld(Nthreads, threads_per_block)


    data_cu = CuArray(data)
    wgt_cu = CuArray(wgt)
    nbins = length(binedges(hist)) - 1
    bincounts = CUDA.zeros(Float32, nbins, Nthreads)
    sumw2   = CUDA.zeros(Float32, nbins, Nthreads)
    nentries = CUDA.zeros(Int32, Nthreads)

    #Launch kernel
    @cuda threads=threads_per_block blocks=blocks _cuda_append_kernel!(data_cu, wgt_cu, binedges(hist), bincounts, sumw2, nentries, hist.overflow)

    #Wait for all histograms to finish filling
    CUDA.synchronize()

    #Add up the results of each thread
    hist.bincounts += vec(sum(bincounts, dims=2))
    hist.sumw2 += vec(sum(sumw2, dims=2))
    hist.nentries[] += length(data)

    return
end


Base.broadcastable(h::CUDAHist1D) = Ref(h)

Statistics.mean(h::CUDAHist1D) = Statistics.mean(bincenters(h), Weights(bincounts(h)))
Statistics.std(h::CUDAHist1D) = Statistics.mean((bincenters(h) .- mean(h)*CUDA.ones(length(bincenters(h)))).^2, Weights(bincounts(h))) |> sqrt 

"""
    normalize(h::CUDAHist1D; width=true)

Create a normalized histogram via division by `integral(h)`, when `width==true`, the
resultant histogram has area under the curve equals 1.

!!! warning
    Implicit approximation is made when using `width=true` with histograms
    that have overflow bins: the overflow data lives inthe left/right most bins
    and the bin width is taken "as is".
"""
function normalize(h::CUDAHist1D; width=true)
    return h*(1/integral(h; width=width))
end

"""
    cumulative(h::CUDAHist1D; forward=true)

Create a cumulative histogram. If `forward`, start
summing from the left.
"""
function cumulative(h::CUDAHist1D; forward=true)
    # https://root.cern.ch/doc/master/TH1_8cxx_source.html#l02608
    f = forward ? identity : reverse
    h = deepcopy(h)
    bc = bincounts(h)
    bc .= f(cumsum(f(bc)))

    s2 = sumw2(h)
    s2 .= f(cumsum(f(s2)))
    return h
end

end