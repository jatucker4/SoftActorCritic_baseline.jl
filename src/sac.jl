"""
Computes loss for value function under current policy.
"""
function compute_loss_q(
	ac::MLPActorCritic,
	ac_targ::MLPActorCritic,
	data::NamedTuple,
	gamma::Float64,
	alpha::Vector{Float64}
)
	o, a, r, o2, d = data
    qs = [q(o, a) for q in ac.qs]
    a2, logp_a2 = ac.pi(o2)

    # Target q-values
    qs_pi_targ = [q(o2, a2) for q in ac_targ.qs]
    q_pi_targ = sum(qs_pi_targ) / length(qs_pi_targ) # min.(qs_pi_targ...)
    backup = @. r + gamma * (1.0 - d) * (q_pi_targ - alpha[] * logp_a2)

    # MSE loss against Bellman backup
    loss_qs = [mean((q .- backup).^2) for q in qs]
    loss_q = sum(loss_qs)
    return loss_q
end

"""
Computes loss for current policy.
"""
function compute_loss_pi(ac::MLPActorCritic, data::NamedTuple, alpha::Vector{Float64})
    o = data.obs
    pi, logp_pi = ac.pi(o)
    qs_pi = [q(o, pi) for q in ac.qs]
    q_pi = sum(qs_pi) / length(qs_pi) # min.(qs_pi...)

    # Entropy-regularized policy loss
    loss_pi = mean(alpha[] .* logp_pi .- q_pi)
    return loss_pi
end

"""
Computes loss for current alpha.
"""
function compute_loss_alpha(ac::MLPActorCritic, data::NamedTuple, alpha::Vector{Float64}, target_entropy::Float64)
    o = data.obs
    _, logp_pi = ac.pi(o)
    loss_alpha = mean(-1.0 .* alpha[] .* (logp_pi .+ target_entropy))
    return loss_alpha
end

"""
Updates policy with observations.
"""
function update(
	ac::MLPActorCritic,
	ac_targ::MLPActorCritic,
	data::NamedTuple,
	q_optimizer::Any,
	pi_optimizer::Any,
	polyak::Float64,
	gamma::Float64,
	alpha_optimizer::Any,
	alpha::Vector{Float64},
	target_entropy::Float64
)

    # Transfer data to GPU
    data = (; zip(keys(data), gpu.(values(data)))...)

    # Gradient descent step for value networks
    loss_q = 0.0
    q_ps = params((q -> q.q).(ac.qs))
    q_gs = gradient(q_ps) do
        loss_q = compute_loss_q(ac, ac_targ, data, gamma, alpha)
        return loss_q
    end
    Flux.update!(q_optimizer, q_ps, q_gs)

    # Gradient descent step for policy network
    loss_pi = 0.0
    pi_ps = params([ac.pi.net, ac.pi.mu_layer, ac.pi.log_std_layer])
    pi_gs = gradient(pi_ps) do
        loss_pi = compute_loss_pi(ac, data, alpha)
        return loss_pi
    end
    Flux.update!(pi_optimizer, pi_ps, pi_gs)

    # Gradient descent step for alpha
    loss_alpha = 0.0
    alpha_ps = params(alpha)
    alpha_gs = gradient(alpha_ps) do
        loss_alpha = compute_loss_alpha(ac, data, alpha, target_entropy)
        return loss_alpha
    end
    Flux.update!(alpha_optimizer, alpha_ps, alpha_gs)

    # Update target networks with in-place Polyak averaging
    for (dest, src) in zip(params((q -> q.q).(ac_targ.qs)), params((q -> q.q).(ac.qs)))
        dest .= polyak .* dest .+ (1.0 .- polyak) .* src
    end

    # Logging
    @debug "Losses" loss_q loss_pi loss_alpha
end

"""
Tests current policy and generates display statistics.
TODO: Statistics can be gathered more efficiently from existing rollouts.
"""
function test_agent(
    ac::MLPActorCritic,
    test_env::CommonRLInterface.AbstractEnv,
    displays::Vector{<:Tuple},
    max_ep_len::Int,
    num_test_episodes::Int
)

    # Copy AC agent and transfer to CPU
    ac = deepcopy(ac)
    to_cpu!(ac)

    # Initialize collections for displayed values
    rets = []
    stdevs = []
    dispvals = [[] for _ in displays]

    # Perform rollouts
    for _ in 1:num_test_episodes

        # Single trajectory from deterministic (mean) policy
        d, ep_ret, ep_len = false, 0.0, 0
        reset!(test_env)
        o = observe(test_env)
        while !(d || ep_len == max_ep_len)
            a = ac(o, true)
          	qs = [q.q(vcat(o, a))[] for q in ac.qs]
            r = act!(test_env, a)
            o = observe(test_env)
            d = terminated(test_env)
            ep_ret += r
            ep_len += 1
            push!(stdevs, std(qs))
        end

        # Compute custom statistics and add to collections
        push!(rets, ep_ret)
        push!.(dispvals, f(test_env) for (_, f) in displays)
    end

    # Average displayed values
    dispvals = [rets, stdevs, dispvals...]
    dispvals_avg = mean.(dispvals)
    return dispvals_avg
end

"""
Converts DateTime to valid cross-platform filename.
"""
function dt_to_fn(dt::DateTime)
    dt = round(dt, Dates.Second)
    str = replace("$dt", ":" => "-")
    return "saved_" * str * ".bson"
end

"""
Converts filename to corresponding unix time (or NaN)
"""
function fn_to_t(fn::String)
    str = replace(fn, r"saved_(.*T)(.*)-(.*)-(.*).bson" => s"\1\2:\3:\4")
    return try datetime2unix(DateTime(str)) catch; NaN end
end

"""
Saves AC networks to specified directory, with optional maximum number of saves.
"""
function checkpoint(ac::MLPActorCritic, save_dir::String, max_saved::Int)
    ac = deepcopy(ac)
    to_cpu!(ac)

    # Delete earliest save if maximum number is exceeded
    if max_saved > 0
        files = readdir(save_dir)
        times = [fn_to_t(file) for file in files]
        if !isempty(times) && sum(@. !isnan(times)) >= max_saved
            i_min = argmin((x -> isnan(x) ? Inf : x).(times))
            rm(joinpath(save_dir, files[i_min]))
        end
    end

    # Save AC agent
    filename = joinpath(save_dir, dt_to_fn(now()))
	@save filename ac
end

"""
Generates values to display and save from display tuples
"""
function gen_showvalues(epoch::Int64, disptups::Vector{<:Tuple})
    return () -> [(:epoch, epoch), ((sym, isempty(hist) ? NaN : hist[end]) for (sym, hist) in disptups)...]
end

"""
Defines SAC solver
"""
Base.@kwdef mutable struct SAC
    # Environment
    env_fn::Function                                # zero-argument function to create new MDP
    obs_dim::Int                                    # dimension of observation space
    act_dim::Int                                    # dimension of action space
    act_mins::Vector{Float64}                       # minimum values of actions
    act_maxs::Vector{Float64}                       # maximum values of actions
    gamma::Float64 = 0.95                           # discount factor

    # Replay buffer
    max_buffer_size::Int = 100000                   # maximum number of timesteps in buffer

    # Actor-critic network
    hidden_sizes::Vector{Int} = [100,100,100]       # dimensions of any hidden layers
    num_q::Int = 2                                  # size of critic ensemble
    activation::Function = SoftActorCritic_baseline.relu     # activation after each hidden layer

    # Training
    q_optimizer::Any = AdaBelief(1e-3)              # optimizer for value networks
    pi_optimizer::Any = AdaBelief(1e-3)             # optimizer for policy network
    alpha_optimizer::Any = AdaBelief(1e-3)          # optimizer for alpha
    batch_size::Int = 64                            # size of each update to networks
    epochs::Int = 200                               # number of epochs
    steps_per_epoch::Int = 200                      # steps of simulation per epoch
    start_steps::Int = 1000                         # steps before following policy
    max_ep_len::Int = 50                            # maximum number of steps per episode
    update_after::Int = 300                         # steps before networks begin to update
    update_every::Int = 50                          # steps between updates
    num_batches::Int = update_every                 # number of batches per update
    polyak::Float64 = 0.995                         # target network averaging parameter
    target_entropy::Float64 = -act_dim              # target entropy (default is heuristic)
    rng::AbstractRNG = Random.GLOBAL_RNG            # random number generator

    # Testing
    num_test_episodes::Int = 100                    # number of test episodes
    displays::Vector{<:Tuple} = Tuple[]             # display values (list of tuples of
                                                    # name and function to apply to MDP
                                                    # after each trajectory)

    # Checkpointing
    save::Bool = false                              # to enable checkpointing
    save_every::Int = 10000                         # steps between checkpoints
    save_dir::String = DEFAULT_SAVE_DIR             # directory to save checkpoints
    max_saved::Int = 100                            # maximum number of checkpoints; set to
                                                    # zero or negative for unlimited
end

"""
Solves MDP with soft actor critic method.
"""
function solve(sac::SAC)

    # Initialize AC agent and auxiliary data structures
    env = sac.env_fn()
    test_env = sac.env_fn()
    replay_buffer = ReplayBuffer(sac.obs_dim, sac.act_dim, sac.max_buffer_size)
    ac = MLPActorCritic(sac.obs_dim, sac.act_dim, sac.act_mins, sac.act_maxs,
        sac.hidden_sizes, sac.num_q, sac.activation, sac.rng)
    ac_targ = deepcopy(ac)
    alpha = [1.0]
    total_steps = sac.steps_per_epoch * sac.epochs

    # Initialize displayed information and progress meter
    @debug "Solve" total_steps
    disptups = [(:score, []), (:stdev, []), ((sym, []) for (sym, _) in sac.displays)...]
    p = Progress(total_steps - sac.update_after)

    ep_ret, ep_len = 0.0, 0
    reset!(env)
    o = observe(env)
    for t in 1:total_steps
        # Choose action
    	random_policy = t <= sac.start_steps
    	a = random_policy ? rand(actions(env)) : ac(gpu(o)) |> cpu

        # Step environment
        r = act!(env, a)
        o2 = observe(env)
        d = terminated(env)
        ep_ret += r
        ep_len += 1

        # Ignore done signal if due to overtime
        d = (ep_len == sac.max_ep_len) ? false : d

        store!(replay_buffer, o, a, r, o2, d)
        o = o2

        # End of trajectory handling
        if d || ep_len == sac.max_ep_len
            ep_ret, ep_len = 0.0, 0
            reset!(env)
            o = observe(env)
        end

        # Actor-critic update
        if t > sac.update_after && t % sac.update_every == 0
            @debug "Updating models" t
            for _ in 1:sac.num_batches
                batch = sample_batch(replay_buffer, sac.batch_size)
                update(ac, ac_targ, batch, sac.q_optimizer, sac.pi_optimizer, sac.polyak,
                    sac.gamma, sac.alpha_optimizer, alpha, sac.target_entropy)
            end
        end

        # End of epoch handling
        epoch = (t - 1) ?? sac.steps_per_epoch + 1
        if t % sac.steps_per_epoch == 0
            # Update display values
            dispvals = test_agent(ac, test_env, sac.displays, sac.max_ep_len, sac.num_test_episodes)
            for ((_, hist), val) in zip(disptups, dispvals)
                push!(hist, val)
            end

            # Log info about epoch
            @debug("Evaluation",
            	  alpha[],
                  mean(mean.(params([q.q for q in ac.qs]))),
            	  mean(mean.(params([ac.pi.net, ac.pi.mu_layer, ac.pi.log_std_layer])))
            )
        end

        # Checkpointing
        if sac.save && t > sac.update_after && t % sac.save_every == 0
            checkpoint(ac, sac.save_dir, sac.max_saved)
        end

        # Progress meter
        if t > sac.update_after
            ProgressMeter.next!(p; showvalues=gen_showvalues(epoch, disptups))
        end
    end

    # Save display values and replay buffer
    info = Dict{String,Any}()
    for (sym, hist) in disptups
        info[String(sym)] = hist
    end
    info["replay_buffer"] = replay_buffer

    return ac, info, env
end
