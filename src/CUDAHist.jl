module CUDAHist

using FHist
import FHist: Hist1D

# INCLUDE ROCM
using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
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
function gpu_bincounts(input; sync=false, binedges, blocksize=256, backend = CUDABackend())
    cu_bincounts = KernelAbstractions.zeros(backend, Int, length(binedges) - 1)
    binindexs = naive_binidxs(input, binedges)
    synchronize(backend)
    histogram!(cu_bincounts, binindexs; blocksize)
    if sync
        synchronize(backend)
    end
    return cu_bincounts
end

function naive_binidxs(input, binedges)
    firstr = first(binedges)
    invstep = inv(step(binedges))
    cursor = floor.(Int32, (input .- firstr) .* invstep)
    binidxs = cursor .+ 1

    return binidxs
end

# This a 1D histogram kernel where the histogramming happens on shmem
@kernel function histogram_kernel!(histogram_output, input)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @uniform warpsize = Int(32)

    @uniform gs = @groupsize()[1]
    @uniform N = length(histogram_output)

    shared_histogram = @localmem Int (gs)

    # This will go through all input elements and assign them to a location in
    # shmem. Note that if there is not enough shem, we create different shmem
    # blocks to write to. For example, if shmem is of size 256, but it's
    # possible to get a value of 312, then we will have 2 separate shmem blocks,
    # one from 1->256, and another from 256->512
    @uniform max_element = 1
    for min_element in 1:gs:N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = 0
        @synchronize()

        max_element = min_element + gs
        if max_element > N
            max_element = N + 1
        end

        # Defining bin on shared memory and writing to it if possible
        bin = input[tid]
        if bin >= min_element && bin < max_element
            bin -= min_element - 1
            @atomic shared_histogram[bin] += 1
        end

        @synchronize()

        if ((lid + min_element - 1) <= N)
            @atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
        end

    end

end

function histogram!(histogram_output, input; blocksize = 256)
    backend = get_backend(histogram_output)
    # Need static block size
    kernel! = histogram_kernel!(backend, (blocksize,))
    kernel!(histogram_output, input, ndrange = size(input))
    return
end

function move(backend, input)
    # TODO replace with adapt(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

end
