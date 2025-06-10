using Test
using FHist
using KernelAbstractions
using CUDA
using CUDAHist: gpu_bincounts, create_histogram, histogram!, move
const backend = isnothing(get(ENV, "CI", nothing)) ? CUDABackend() : CPU()

# fhist = Hist1D(randn(1000))
# cuda_hist = CUDAHist1D(fhist);

# warm up
begin
    wi_cpu = rand(1:128, 512)
    Hist1D(wi_cpu; binedges=1:129)
    wi = move(backend, wi_cpu)
    gpu_bincounts(wi; binedges=1:129, backend)
    KernelAbstractions.synchronize(backend)
end

@testset "basic binning" begin
    be1 = [0, 1, 4]
    data1 = [0.5]

    ref_hist1 = FHist.Hist1D(data1; binedges=be1)
end

@testset "trivial integer binning" begin
    for N = 0:8
        rand_input = Float32.(rand(1:128, 1024 * 2^N))
        binedges = 1:129
        @info N
        histogram_rand_baseline = @time Hist1D(rand_input; binedges)
        cu_input = move(backend, rand_input)
        @time begin
            cu_bcs = gpu_bincounts(cu_input; binedges, backend)
            KernelAbstractions.synchronize(backend)
        end

        @test isapprox(Array(cu_bcs), bincounts(histogram_rand_baseline))
    end
end

@testset "weird input length" begin
    for N = [1,2,3,4,10,128, 1000, 10000]
        rand_input = Float32.(rand(1:128, N))
        binedges = 1:129
        @info N
        histogram_rand_baseline = Hist1D(rand_input; binedges)
        cu_input = move(backend, rand_input)
        cu_bcs = gpu_bincounts(cu_input; binedges, backend)
        KernelAbstractions.synchronize(backend)

        @test isapprox(Array(cu_bcs), bincounts(histogram_rand_baseline))
    end
end

@testset "trivial integer binning with weights" begin
    for N = 0:8
        rand_input = Float32.(rand(1:128, 1024 * 2^N))
        rand_weights = Float32.(rand(1024 * 2^N))

        binedges = 1:129
        histogram_rand_baseline = Hist1D(rand_input; weights=rand_weights, binedges)
        cu_input = move(backend, rand_input)
        cu_weights = move(backend, rand_weights)
        cu_bcs = gpu_bincounts(cu_input; weights=cu_weights, sharemem=true, binedges, backend)
        KernelAbstractions.synchronize(backend)

        @test isapprox(Array(cu_bcs), bincounts(histogram_rand_baseline))
    end
end

@testset "naive-algo, trivial integer binning with weights" begin
    for N = 0:8
        rand_input = Float32.(rand(1:128, 1024 * 2^N))
        rand_weights = Float32.(rand(1024 * 2^N))

        binedges = 1:129
        histogram_rand_baseline = Hist1D(rand_input; weights=rand_weights, binedges)
        cu_input = move(backend, rand_input)
        cu_weights = move(backend, rand_weights)
        cu_bcs = gpu_bincounts(cu_input; weights=cu_weights, sharemem=false, binedges, backend)
        KernelAbstractions.synchronize(backend)

        @test isapprox(Array(cu_bcs), bincounts(histogram_rand_baseline))
    end
end

@testset "simple binning" begin
    rand_input = rand(1:128, 1024 * 2)
    binedges = 1:4:129
    hist_ref = Hist1D(rand_input; binedges)
    rand_input = move(backend, rand_input)
    cu_bcs = gpu_bincounts(rand_input; binedges, backend)
    KernelAbstractions.synchronize(backend)

    @test isapprox(Array(cu_bcs), bincounts(hist_ref))
end

@testset "uniform binning" begin
    rand_input = rand(1024 * 2)
    binedges = 0:0.1:1
    hist_ref = Hist1D(rand_input; binedges)
    rand_input = move(backend, rand_input)
    cu_bcs = gpu_bincounts(rand_input; binedges, backend)
    KernelAbstractions.synchronize(backend)

    @test isapprox(Array(cu_bcs), bincounts(hist_ref))
end

@testset "trivial integer binning (1000 bins)" begin
    for N = 0:8
        rand_input = Float32.(rand(1024 * 2^N)) .* 10
        binedges = 0:0.1:100
        hist_ref = Hist1D(rand_input; binedges)
        cu_input = move(backend, rand_input)
        cu_bcs = gpu_bincounts(cu_input; binedges, backend)
        KernelAbstractions.synchronize(backend)

        @test isapprox(Array(cu_bcs), bincounts(hist_ref))
    end
end

# @testset "histogram tests" begin
#     rand_input = [rand(1:128) for i in 1:1000]
#     linear_input = [i for i in 1:1024]
#     all_two = [2 for i in 1:512]

#     histogram_rand_baseline = create_histogram(rand_input)
#     histogram_linear_baseline = create_histogram(linear_input)
#     histogram_two_baseline = create_histogram(all_two)

#     rand_input = move(backend, rand_input)
#     linear_input = move(backend, linear_input)
#     all_two = move(backend, all_two)

#     rand_histogram = KernelAbstractions.zeros(backend, Int, 128)
#     linear_histogram = KernelAbstractions.zeros(backend, Int, 1024)
#     two_histogram = KernelAbstractions.zeros(backend, Int, 2)

#     histogram!(rand_histogram, rand_input)
#     histogram!(linear_histogram, linear_input)
#     histogram!(two_histogram, all_two)
#     KernelAbstractions.synchronize(backend)

#     @test isapprox(Array(rand_histogram), histogram_rand_baseline)
#     @test isapprox(Array(linear_histogram), histogram_linear_baseline)
#     @test isapprox(Array(two_histogram), histogram_two_baseline)
# end
