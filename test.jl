using Test
using FHist
using CUDAHist

fhist = Hist1D(randn(1000))
cuda_hist = CUDAHist1D(fhist);

@testset "Equality Tests" begin
    @test fhist.overflow == cuda_hist.overflow
    @test Vector(FHist.binedges(fhist)) == Vector(CUDAHist.binedges(cuda_hist))
    @test FHist.bincounts(fhist) == Vector(CUDAHist.bincounts(cuda_hist))
    @test FHist.bincenters(fhist) == Vector(CUDAHist.bincenters(cuda_hist))
    @test FHist.sumw2(fhist) == Vector(CUDAHist.sumw2(cuda_hist))
    @test FHist.nentries(fhist) == CUDAHist.nentries(cuda_hist)
    @test FHist.binerrors(fhist) ≈ Vector(CUDAHist.binerrors(cuda_hist))
    @test FHist.nbins(fhist) == CUDAHist.nbins(cuda_hist)
    @test FHist.integral(fhist) == CUDAHist.integral(cuda_hist)
    @test FHist.mean(fhist) ≈ CUDAHist.mean(cuda_hist)
    @test FHist.std(fhist) ≈ CUDAHist.std(cuda_hist)
    @test cuda_hist == fhist
end

@testset "Append Tests" begin
    data = Float32.(randn(1000))
    FHist.append!(fhist, data)
    CUDAHist.append!(cuda_hist, data)
    @test cuda_hist == fhist;

    CUDAHist.cuda_append!(cuda_hist, data, 100)
    FHist.append!(fhist, data)
    @test cuda_hist == fhist;
end