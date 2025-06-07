using Test
using FHist
using KernelAbstractions
using CUDA
using CUDAHist: gpu_bincounts, create_histogram, histogram!, move
const backend = CUDABackend()

# fhist = Hist1D(randn(1000))
# cuda_hist = CUDAHist1D(fhist);

# warm up
begin
    wi_cpu = rand(1:128, 512)
    Hist1D(wi_cpu; binedges=1:129)
    wi = move(backend, wi_cpu)
    wi_h = KernelAbstractions.zeros(backend, Int, 128)
    histogram!(wi_h, wi)
    KernelAbstractions.synchronize(backend)
end

@testset "basic binning" begin
    be1 = [0, 1, 4]
    data1 = [0.5]

    ref_hist1 = FHist.Hist1D(data1; binedges=be1)
end

@testset "trivial integer binning" begin
    for N = 0:8
        rand_input = rand(1:128, 1024 * 2^N)

        binedges = 1:129
        @info N
        histogram_rand_baseline = @time Hist1D(rand_input; binedges)
        rand_input = move(backend, rand_input)
        @time begin
            cu_bcs = gpu_bincounts(rand_input; binedges)
            KernelAbstractions.synchronize(backend)
        end

        @test isapprox(Array(cu_bcs), bincounts(histogram_rand_baseline))
    end
end

@testset "histogram tests" begin
    rand_input = [rand(1:128) for i in 1:1000]
    linear_input = [i for i in 1:1024]
    all_two = [2 for i in 1:512]

    histogram_rand_baseline = create_histogram(rand_input)
    histogram_linear_baseline = create_histogram(linear_input)
    histogram_two_baseline = create_histogram(all_two)

    rand_input = move(backend, rand_input)
    linear_input = move(backend, linear_input)
    all_two = move(backend, all_two)

    rand_histogram = KernelAbstractions.zeros(backend, Int, 128)
    linear_histogram = KernelAbstractions.zeros(backend, Int, 1024)
    two_histogram = KernelAbstractions.zeros(backend, Int, 2)

    histogram!(rand_histogram, rand_input)
    histogram!(linear_histogram, linear_input)
    histogram!(two_histogram, all_two)
    KernelAbstractions.synchronize(backend)

    @test isapprox(Array(rand_histogram), histogram_rand_baseline)
    @test isapprox(Array(linear_histogram), histogram_linear_baseline)
    @test isapprox(Array(two_histogram), histogram_two_baseline)
end
