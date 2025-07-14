using Test
using FHist, BenchmarkTools
using KernelAbstractions
using CUDA
using GPUHist: gpu_bincounts, move
const backend = isnothing(get(ENV, "CI", nothing)) ? CUDABackend() : CPU()

using PythonCall
const np = pyimport("numpy")
const cupy = pyimport("cupy")

function cupy_histogram_sync(input; weights, bins, range)
    a = cupy.histogram(input; weights=weights, bins=bins, range=range)
    cupy.cuda.Device().synchronize()
    a
end

const input_Ls = [2^N for N in 7:3:22]
const rand_inputs = map(input_Ls) do N
    rand(Float32, N)
end
const rand_weights = map(input_Ls) do N
    rand(Float32, N)
end
const rand_inputs_np = np.array.(rand_inputs)
const rand_weights_np = np.array.(rand_weights)
const rand_inputs_cupy = cupy.array.(rand_inputs)
const rand_weights_cupy = cupy.array.(rand_weights)

const EVALS = 1
const SAMPLES = 300

const SUITE = BenchmarkGroup()
SUITE["N_input_scan"] = BenchmarkGroup()
const L_binedges = 1000
SUITE["N_bins_scan"] = BenchmarkGroup()

const binedges_Ls = [256, 512, 512*2, 512*4, 512*8, 512*12]

for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rand_inputs_np, rand_weights_np)
    uniform_binedges = range(0.0, 1.0; length=L_binedges)
    SUITE["NumpyBaseline"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; binedges=$uniform_binedges);
        evals=EVALS,
        samples=SAMPLES
    )
    SUITE["NumpyBaseline"]["Numpy (CPU, v1.26)"][N] = @benchmarkable(
    np.histogram($input_np; bins=$(L_binedges+1), range=$((0.0, 1.0)));
        evals=EVALS,
        samples=SAMPLES
    )
end

for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rand_inputs_np, rand_weights_np)
    uniform_binedges = range(0.0, 1.0; length=L_binedges)
    SUITE["NumpyWeights"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; weights=$weights, binedges=$uniform_binedges);
        evals=EVALS,
        samples=SAMPLES
    )
    SUITE["NumpyWeights"]["Numpy (CPU, v1.26)"][N] = @benchmarkable(
    np.histogram($input_np; weights=$(weights_np), bins=$(L_binedges+1), range=$((0.0, 1.0)));
        evals=EVALS,
        samples=SAMPLES
    )
end

for sharemem in (true, false)
    for (N, input, weights, input_np, weights_np, input_cupy, weights_cupy) in zip(input_Ls, rand_inputs, rand_weights, rand_inputs_np, rand_weights_np, rand_inputs_cupy, rand_weights_cupy)
        uniform_binedges = range(0.0, 1.0; length=L_binedges)
        SUITE["N_input_scan_sharemem$sharemem"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$uniform_binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        SUITE["N_input_scan_sharemem$sharemem-v2"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$uniform_binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        SUITE["N_input_scan_sharemem$sharemem-v2"]["CuPy (v12)"][N] = @benchmarkable(
            cupy_histogram_sync($input_cupy; weights=$weights_cupy, bins=$(L_binedges+1), range=$((0.0, 1.0)));
            evals=EVALS,
            samples=SAMPLES
        )
        for bs in (32, 128, 512, 1024)
            SUITE["N_input_scan_sharemem$sharemem"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=false, sharemem=$sharemem, binedges=$uniform_binedges);
                evals=EVALS,
                samples=SAMPLES
            )
            SUITE["N_input_scan_sharemem$sharemem-v2"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=true, sharemem=$sharemem, binedges=$uniform_binedges);
                evals=EVALS,
                samples=SAMPLES
            )
        end
    end


    for N in binedges_Ls
        input = rand_inputs[5]
        weights = rand_weights[5]
        input_cupy = rand_inputs_cupy[5]
        weights_cupy = rand_weights_cupy[5]

        uniform_binedges = range(0.; stop=1.0, length=N)

        SUITE["N_bins_scan_sharemem$sharemem"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$uniform_binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        SUITE["N_bins_scan_sharemem$sharemem-v2"]["FHist.jl (CPU)"][N] = @benchmarkable(
            Hist1D($input; weights=$weights, binedges=$uniform_binedges);
            evals=EVALS,
            samples=SAMPLES
        )
        for bs in (32, 128, 512, 1024)
            SUITE["N_bins_scan_sharemem$sharemem"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, v2=false, sharemem=$sharemem, binedges=$uniform_binedges);
                evals=EVALS,
                samples=SAMPLES
            )
            SUITE["N_bins_scan_sharemem$sharemem-v2"]["GPU-blocksize$bs"][N] = @benchmarkable(
                gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), v2=true, backend, sharemem=$sharemem, binedges=$uniform_binedges);
                evals=EVALS,
                samples=SAMPLES
            )
            SUITE["N_bins_scan_sharemem$sharemem-v2"]["CuPy (v12)"][N] = @benchmarkable(
                cupy_histogram_sync($input_cupy; weights=$weights_cupy, bins=$(N+1), range=$((0.0, 1.0)));
                evals=EVALS,
                samples=SAMPLES
            )
        end
    end
end

for (N, input, weights, input_np, weights_np) in zip(input_Ls, rand_inputs, rand_weights, rand_inputs_np, rand_weights_np)
    fake_non_uniform_binedges = Float32.(range(0.0, 1.0; length=L_binedges))
    sharemem=true
    SUITE["N_input_scan_non_uniform_binedges"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; weights=$weights, binedges=$fake_non_uniform_binedges);
        evals=EVALS,
        samples=SAMPLES
    )

    for bs in (512, 1024)
        SUITE["N_input_scan_non_uniform_binedges"]["GPU-blocksize$bs"][N] = @benchmarkable(
        gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), backend, sharemem=$sharemem, binedges=$(move(backend, fake_non_uniform_binedges)));
            evals=EVALS,
            samples=SAMPLES
        )
    end
end


for N in binedges_Ls
    input = rand_inputs[5]
    weights = rand_weights[5]
    fake_non_uniform_binedges = Float32.(range(0.0, 1.0; length=N))
    sharemem = true

    SUITE["N_bins_scan_non_uniform_binedges"]["FHist.jl (CPU)"][N] = @benchmarkable(
        Hist1D($input; weights=$weights, binedges=$fake_non_uniform_binedges);
        evals=EVALS,
        samples=SAMPLES
    )
    for bs in (512, 1024)
        SUITE["N_bins_scan_non_uniform_binedges"]["GPU-blocksize$bs"][N] = @benchmarkable(
        gpu_bincounts($(move(backend, input)); blocksize=$bs, sync=true, weights=$(move(backend, weights)), v2=true, backend, sharemem=$sharemem, binedges=$(move(backend, fake_non_uniform_binedges)));
            evals=EVALS,
            samples=SAMPLES
        )
    end
end

results = run(SUITE, verbose=true, seconds=10)
# BenchmarkTools.save("benchmark_params.json", params(SUITE));
BenchmarkTools.save("benchmark_result_new_$(L_binedges)bins_all_v3.json", results)
