module SoftActorCritic

__precompile__(false)

export
	SAC,
    solve,
    MLPActorCritic,
    Master2Worker,
    Worker2Master,
    simulation_task,
    distributed_solve

using BSON: @save
using CommonRLInterface
if VERSION >= v"1.3"
	using CUDA
	const CuModule = CUDA
else
	# For compatibility with legacy sXu Julia
	using CUDAnative
	using CuArrays
	const CuModule = CuArrays
end
using Dates
using Distributed
using Distributions
using Flux
using Flux: params
using LinearAlgebra
using ProgressMeter
using Random
using Statistics

const DEFAULT_SAVE_DIR = joinpath(@__DIR__, "../checkpoints")

include("core.jl")
include("replay_buffer.jl")
include("sac.jl")
include("distributed.jl")

end
