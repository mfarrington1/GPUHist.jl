using Test
using FHist, BenchmarkTools
using KernelAbstractions
using CUDA
using CUDAHist: gpu_bincounts, move
const backend = isnothing(get(ENV, "CI", nothing)) ? CUDABackend() : CPU()

using PythonCall
const np = pyimport("numpy")

const input_Ls = [2^N for N in 7:3:25]
const rand_inputs = map(input_Ls) do N
    rand(Float32, N)
end
const rand_weights = map(input_Ls) do N
    rand(Float32, N)
end
const rannd_inputs_np = np.array.(rand_inputs)
const rannd_weights_np = np.array.(rand_weights)

const EVALS = 1
const SAMPLES = 300

const SUITE = BenchmarkGroup()
SUITE["N_input_scan"] = BenchmarkGroup()
const L_binedges = 1000
SUITE["N_bins_scan"] = BenchmarkGroup()

for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rannd_inputs_np, rannd_weights_np)
    binedges = range(0.0, 1.0; length=L_binedges)
    SUITE["NumpyBaseline"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; binedges=$binedges);
        evals=EVALS,
        samples=SAMPLES
    )
    SUITE["NumpyBaseline"]["Numpy (CPU, v2.3.0)"][N] = @benchmarkable(
    np.histogram($input_np; bins=$(L_binedges+1), range=$((0.0, 1.0)));
        evals=EVALS,
        samples=SAMPLES
    )
end

for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rannd_inputs_np, rannd_weights_np)
    binedges = range(0.0, 1.0; length=L_binedges)
    SUITE["NumpyWeights"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; weights=$weights, binedges=$binedges);
        evals=EVALS,
        samples=SAMPLES
    )
    SUITE["NumpyWeights"]["Numpy (CPU, v2.3.0)"][N] = @benchmarkable(
    np.histogram($input_np; weights=$(weights_np), bins=$(L_binedges+1), range=$((0.0, 1.0)));
        evals=EVALS,
        samples=SAMPLES
    )
end

const binedges_Ls = [256, 512, 512*2, 512*4, 512*8, 512*12]

for sharemem in (true, false)
    for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rannd_inputs_np, rannd_weights_np)
        binedges = range(0.0, 1.0; length=L_binedges)
        SUITE["N_input_scan_sharemem$sharemem"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        SUITE["N_input_scan_sharemem$sharemem-v2"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        for bs in (32, 128, 512, 1024)
            SUITE["N_input_scan_sharemem$sharemem"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=false, sharemem=$sharemem, binedges=$binedges);
                evals=EVALS,
                samples=SAMPLES
            )
            SUITE["N_input_scan_sharemem$sharemem-v2"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=true, sharemem=$sharemem, binedges=$binedges);
                evals=EVALS,
                samples=SAMPLES
            )
        end
    end


    for N in binedges_Ls
        input = rand_inputs[5]
        weights = rand_weights[5]

        binedges = range(0.; stop=1.0, length=N)

        SUITE["N_bins_scan_sharemem$sharemem"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        SUITE["N_bins_scan_sharemem$sharemem-v2"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        for bs in (32, 128, 512, 1024)
            SUITE["N_bins_scan_sharemem$sharemem"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=false, sharemem=$sharemem, binedges=$binedges);
                evals=EVALS,
                samples=SAMPLES
            )
            SUITE["N_bins_scan_sharemem$sharemem-v2"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), v2=true, backend, sharemem=$sharemem, binedges=$binedges);
                evals=EVALS,
                samples=SAMPLES
            )
        end
    end
end


results = run(SUITE, verbose=true, seconds=10)
# BenchmarkTools.save("benchmark_params.json", params(SUITE));
BenchmarkTools.save("benchmark_result_$(L_binedges)bins_all.json", results)

