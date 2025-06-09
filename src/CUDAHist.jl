module CUDAHist

using FHist
import FHist: Hist1D

# INCLUDE ROCM
using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace, @Const
import CUDA
using CUDA.CUDAKernels
CUDA.allowscalar(false)

# Function to use as a baseline for CPU metrics
function create_histogram(input)
    histogram_output = zeros(Int, maximum(input))
    for i in input
        histogram_output[i] += 1
    end
    return histogram_output
end

# input already on gpu
function gpu_bincounts(input; weights=nothing, sync=false, binedges, blocksize=256, backend=CUDABackend())
    cu_bincounts = KernelAbstractions.zeros(backend, Float32, length(binedges) - 1)
    firstr = Float32(first(binedges))
    invstep = Float32(inv(step(binedges)))
    # binindexs = naive_binidxs(input, binedges)
    # synchronize(backend)
    histogram!(cu_bincounts, input, firstr, invstep; weights, blocksize)
    if sync
        synchronize(backend)
    end
    return cu_bincounts
end

function naive_binidxs(input, binedges)
    firstr = first(binedges)
    invstep = inv(step(binedges))
    cursor = floor.(Int, (input .- firstr) .* invstep)
    binidxs = cursor .+ 1

    return binidxs
end

# This a 1D histogram kernel where the histogramming happens on shmem
@kernel unsafe_indices = true function histogram_kernel!(histogram_output, @Const(input_raw), @Const(weights), firstr, invstep)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)

    @uniform gs = Int32(prod(@groupsize()))
    tid = (gid - Int32(1)) * gs + lid
    @uniform N = Int32(length(histogram_output))

    shared_histogram = @localmem Float32 (gs)

    # This will go through all input elements and assign them to a location in
    # shmem. Note that if there is not enough shem, we create different shmem
    # blocks to write to. For example, if shmem is of size 256, but it's
    # possible to get a value of 312, then we will have 2 separate shmem blocks,
    # one from 1->256, and another from 256->512
    for min_element in Int32(1):gs:N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = Int32(0)
        @synchronize()

        max_element = min_element + gs
        if max_element > N
            max_element = N + Int32(1)
        end

        # Defining bin on shared memory and writing to it if possible
        if tid <= length(input_raw)
            x = input_raw[tid]
            cursor = floor(Int32, (x - firstr) * invstep)
            bin = cursor + Int32(1)
            if bin >= min_element && bin < max_element
                bin -= min_element - Int32(1)
                @atomic shared_histogram[bin] += isnothing(weights) ? one(Float32) : weights[tid]
            end
        end

        @synchronize()

        if ((lid + min_element - Int32(1)) <= N)
            @atomic histogram_output[lid+min_element-Int32(1)] += shared_histogram[lid]
        end

    end
end

function histogram!(histogram_output, input, firstr, invstep; weights=nothing, blocksize=256)
    backend = get_backend(histogram_output)
    # Need static block size
    if !isnothing(weights)
        @assert get_backend(weights) == backend "Weights must be on the same backend as histogram_output"
    end
    kernel! = histogram_kernel!(backend, (blocksize,))
    kernel!(histogram_output, input, weights, firstr, invstep, ndrange=size(input))
    return
end

function move(backend, input)
    # TODO replace with adapt(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

end
