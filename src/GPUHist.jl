module GPUHist

using FHist
import FHist: Hist1D

# INCLUDE ROCM
using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace, @Const
import CUDA
using CUDA.CUDAKernels
CUDA.allowscalar(false)

@kernel unsafe_indices = true function histogram_naive_kernel!(histogram_output, @Const(output_L), @Const(input_raw), @Const(weights), @Const(firstr), @Const(invstep))
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)

    gs = Int32(prod(@groupsize()))
    tid = (gid - Int32(1)) * gs + lid
    if tid <= length(input_raw)
        x = input_raw[tid]
        cursor = floor(Int32, (x - firstr) * invstep)
        bin = cursor + Int32(1)
        @atomic histogram_output[bin] += isnothing(weights) ? one(Float32) : weights[tid]
    end
end

@kernel unsafe_indices = true function histogram_sharemem_kernel!(histogram_output, @Const(output_L),
    @Const(input_raw), @Const(weights), @Const(firstr), @Const(invstep)
)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)

    gs = Int32(prod(@groupsize()))
    tid = (gid - Int32(1)) * gs + lid
    N = Int32(length(histogram_output))

    shared_histogram = @localmem eltype(histogram_output) (gs)

    bin = Int32(0)
    if tid <= length(input_raw)
        x = input_raw[tid]
        cursor = floor(Int32, (x - firstr) * invstep)
        bin = cursor + Int32(1)
    end
    min_element = Int32(1)
    while min_element <= N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = Int32(0)
        @synchronize()

        max_element = min_element + gs
        if max_element > N
            max_element = N + Int32(1)
        end

        # Defining bin on shared memory and writing to it if possible
        if bin >= min_element && bin < max_element
            bin -= min_element - Int32(1)
            @atomic shared_histogram[bin] += isnothing(weights) ? one(Float32) : weights[tid]
        end

        @synchronize()

        if ((lid + min_element - Int32(1)) <= N)
            @atomic histogram_output[lid+min_element-Int32(1)] += shared_histogram[lid]
        end

        min_element += gs
    end
end

@kernel unsafe_indices = true function histogram_sharemem_v2_kernel!(histogram_output, ::Val{N},
    @Const(input_raw), @Const(weights), @Const(firstr), @Const(invstep)
) where {N}
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)

    @uniform gs = Int32(prod(@groupsize()))
    tid = (gid - Int32(1)) * gs + lid
    max_oid = Int32(length(histogram_output))

    shared_histogram = @localmem eltype(histogram_output) (N)

    # Setting shared_histogram to 0
    min_element = Int32(1)
    while min_element < N
        oid = min_element + lid - Int32(1)
        if oid <= max_oid
            @inbounds shared_histogram[oid] = Int32(0)
        end
        min_element += gs
    end
    @synchronize()

    # Defining bin on shared memory and writing to it if possible
    if tid <= length(input_raw)
        x = input_raw[tid]
        cursor = floor(Int32, (x - firstr) * invstep)
        bin = cursor + Int32(1)
        if bin >= Int32(1) && bin <= N
            @atomic shared_histogram[bin] += isnothing(weights) ? one(Float32) : weights[tid]
        end
    end
    @synchronize()

    min_element = Int32(1)
    while min_element < N
        oid = min_element + lid - Int32(1)
        if oid <= max_oid
            @atomic histogram_output[oid] += shared_histogram[oid]
        end

        min_element += gs
    end
end

@kernel unsafe_indices = true function histogram_sharemem_v3_kernel!(histogram_output, @Const(output_L),
    @Const(input_raw), @Const(weights), @Const(binedges)
)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)

    gs = Int32(prod(@groupsize()))
    tid = (gid - Int32(1)) * gs + lid
    N = Int32(length(histogram_output))
    nedges = Int32(length(binedges))

    shared_histogram = @localmem eltype(histogram_output) (gs)

    bin = Int32(0)
    if tid <= length(input_raw)
        x = input_raw[tid]
        bin += Int32(1)  # Start bin at 1 to match histogram_output indexing
        #Before performing the search, check x is actually within the range of binedges
        if x < binedges[1] || x ≥ binedges[nedges]
            bin = -Int32(1)  # Set bin to -1 if x is out of bounds
        else
            bin = searchsortedlast(binedges, x)
        end
    end

    min_element = Int32(1)
    while min_element <= N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = Int32(0)
        @synchronize()

        max_element = min_element + gs
        if max_element > N
            max_element = N + Int32(1)
        end

        # Defining bin on shared memory and writing to it if possible
        if bin >= min_element && bin < max_element
            bin -= min_element - Int32(1)
            @atomic shared_histogram[bin] += isnothing(weights) ? one(Float32) : weights[tid]
        end

        @synchronize()

        if ((lid + min_element - Int32(1)) <= N)
            @atomic histogram_output[lid+min_element-Int32(1)] += shared_histogram[lid]
        end

        min_element += gs
    end
end

# input already on gpu
function gpu_bincounts(input; weights=nothing, sync=false, sharemem=true, v2=false, binedges, blocksize=256, backend=CUDABackend())
    cu_bincounts = KernelAbstractions.zeros(backend, Float32, length(binedges) - 1)
    if binedges isa AbstractRange
        firstr = Float32(first(binedges))
        invstep = Float32(inv(step(binedges)))
        histogram!(cu_bincounts, input; binedges=nothing, firstr, invstep, weights, blocksize, sharemem, v2)
    else
        histogram!(cu_bincounts, input; binedges, blocksize, sharemem, v2, weights)
    end

    if sync
        synchronize(backend)
    end
    return cu_bincounts
end

function histogram!(histogram_output, input; binedges=nothing, firstr=nothing, invstep=nothing, sharemem=false, v2=false, weights=nothing, blocksize=256)
    backend = get_backend(histogram_output)
    # Need static block size
    if !isnothing(weights)
        @assert get_backend(weights) == backend "Weights must be on the same backend as histogram_output"
    end
    if binedges !== nothing
        kernel! = histogram_sharemem_v3_kernel!(backend, (blocksize))
        kernel!(histogram_output, Val(length(histogram_output)), input, weights, binedges, ndrange=size(input))
    else
        kernel! = if sharemem && v2
            histogram_sharemem_v2_kernel!(backend, (blocksize,))
        elseif sharemem
            histogram_sharemem_kernel!(backend, (blocksize,))
        else
            histogram_naive_kernel!(backend, (blocksize,))
        end
        kernel!(histogram_output, Val(length(histogram_output)), input, weights, firstr, invstep, ndrange=size(input))
    end
    return
end

function move(backend, input)
    # TODO replace with adapt(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

end
