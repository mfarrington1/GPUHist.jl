using Test
using FHist
using KernelAbstractions
using CUDA
using GPUHist: gpu_bincounts, histogram!, move
using Random
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
    for N = [1, 2, 3, 4, 10, 128, 1000, 10000]
        rand_input = Float32.(rand(1:128, N))
        binedges = 1:129
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

@testset "trivial integer binning (2000 bins)" begin
    for Nbins = [256, 512, 512*2, 512*3, 512*4, 512*5]
        for N = 4:8, v2=(true, false)
            rand_input = Float32.(rand(1024 * 2^N)) .* Nbins
            binedges = 0:1.0:Nbins
            hist_ref = Hist1D(rand_input; binedges)
            cu_input = move(backend, rand_input)
            cu_bcs = gpu_bincounts(cu_input; blocksize=1024, sync=true, binedges, backend, v2)

            @test isapprox(Array(cu_bcs), bincounts(hist_ref))
        end
    end
end

@testset "non-uniform binning" begin
    for Nbins = [256, 512, 512*2, 512*3, 512*4, 512*5]    
        for N = 4:8
            rand_input = Float32.(rand(1024 * 2^N))
            binedges = Float32.(sort(randperm(1024 * 2^N)[1:Nbins]))
            hist_ref = Hist1D(rand_input; binedges)
            cu_input = move(backend, rand_input)
            cu_bcs = gpu_bincounts(cu_input; blocksize=1024, sync=true, binedges, backend)

            @test isapprox(Array(cu_bcs), bincounts(hist_ref))
        end
    end
end
