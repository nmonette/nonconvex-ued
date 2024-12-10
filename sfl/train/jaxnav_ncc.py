import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax import core,struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState
import chex
from enum import IntEnum
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import hydra
from omegaconf import OmegaConf
import os
import wandb
import functools
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle
from functools import partial

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance
from jaxmarl.environments.jaxnav.jaxnav_ued_utils import make_level_mutator

from sfl.util.jaxued.jaxued_utils import compute_max_returns, l1_value_loss, max_mc, positive_value_loss

from sfl.train.train_utils import save_params
from sfl.train.jaxnav_plr import Transition, RolloutBatch, update_actor_critic_rnn, batchify, unbatchify, ScannedRNN
from sfl.train.minigrid_ncc import LevelSampler
from sfl.util.ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated
from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.util.rl.plr import PLRManager, PLRBuffer
from sfl.util.rl.ued_scores import UEDScore, compute_ued_scores

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    fc_dim_size: int = 512
    hidden_size: int = 512
    is_recurrent: bool = True
    use_layer_norm: bool = False
    use_elu: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.elu(embedding) if self.use_elu else nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.elu(actor_mean) if self.use_elu else nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.elu(critic) if self.use_elu else nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), None

def update_actor_critic_rnn(
    rng,
    train_state: TrainState,
    init_hstate,
    traj_batch,
    advantages,
    targets,    
    config,
    update_grad=True,
):
    
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                                        
                # RERUN NETWORK
                _, pi, value, _ = train_state.apply_fn(
                    params,
                    init_hstate.transpose(),
                    (traj_batch.obs, traj_batch.last_done),
                )
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(
                    value_losses, value_losses_clipped
                ).mean(where=(1 - traj_batch.mask))
                
                # CALCULATE ACTOR LOSS
                logratio = log_prob - traj_batch.log_prob
                ratio = jnp.exp(logratio)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean(where=(1 - traj_batch.mask))
                entropy = pi.entropy().mean(where=(1 - traj_batch.mask))

                # debug
                approx_kl = jax.lax.stop_gradient(
                    ((ratio - 1) - logratio).mean()
                )
                clipfrac = jax.lax.stop_gradient(
                    (jnp.abs(ratio - 1) > config["CLIP_EPS"]).mean()
                )

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

            grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, init_hstate, traj_batch, advantages, targets
            )
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            total_loss = jax.tree_map(lambda x: x.mean(), total_loss)
            return train_state, total_loss

        (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        ) = update_state
        rng, _rng = jax.random.split(rng)

        init_hstate = jnp.reshape(
            init_hstate, (config["HIDDEN_SIZE"], config["NUM_ACTORS"])
        )
        batch = (
            init_hstate,
            traj_batch,
            advantages.squeeze(),
            targets.squeeze(),
        )
        permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], config["NUM_MINIBATCHES"], -1]
                    + list(x.shape[2:]),
                ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        # total_loss = jax.tree_map(lambda x: x.mean(), total_loss)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        return update_state, total_loss 
    
    update_state = (
        train_state, 
        init_hstate[None, :].squeeze().transpose(),
        traj_batch,
        advantages,
        targets,
        rng,
    )
    (train_state, _, _, _, _, rng), loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
    )
    return (rng, train_state), loss_info

def sample_trajectories_rnn(
    rng, 
    env: JaxNav,
    train_state,
    init_hstate,
    init_obs,
    init_env_state,
    config,
    num_envs,
    gamma: float = 1.0,
):
    def _env_step(runner_state, unused):
        train_state, env_state, last_obs, last_done, hstate, rng, disc_factor, disc_return = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        obs_batch = batchify(last_obs, env.agents, num_envs * env.num_agents)
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, value, _ = train_state.apply_fn(train_state.params, hstate, ac_in)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        env_act = unbatchify(
            action, env.agents, num_envs, env.num_agents
        )
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, 0)
        )(rng_step, env_state, env_act, init_env_state) 
        done_batch = batchify(done, env.agents, num_envs * env.num_agents).squeeze()
        train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
        transition = Transition(
            jnp.tile(done["__all__"], env.num_agents),
            last_done,
            done_batch,
            action.squeeze(),
            value.squeeze(),
            batchify(reward, env.agents, num_envs * env.num_agents).squeeze(),
            log_prob.squeeze(),
            obs_batch,
            train_mask, # 0 if valid, 1 if not
            info,
        )

        disc_factor *= gamma
        rewards = jnp.stack(jax.tree_util.tree_leaves(reward))
        disc_return = disc_return + gamma * rewards

        runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, disc_factor, disc_return)
        return runner_state, (transition)

    (train_state, last_env_state, last_obs, last_done, hstate, rng, _, disc_returns), transitions = jax.lax.scan(
        _env_step,
        (
            train_state, 
            init_env_state,
            init_obs,
            jnp.zeros((num_envs * env.num_agents), dtype=bool),
            init_hstate,
            rng,
            1.0,  # gamma
            jnp.zeros((env.num_agents, num_envs), dtype=float) # disc_return
        ),
        None,
        config["NUM_STEPS"]
    )      
            
    last_obs_batch = batchify(last_obs, env.agents, num_envs * env.num_agents)
    ac_in = (
        last_obs_batch[np.newaxis, :],
        last_done[np.newaxis, :],
    )
    _, _, last_val, _ = train_state.apply_fn(train_state.params, hstate, ac_in)
    last_val = last_val.squeeze()
    
    return (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng, disc_returns), transitions

@hydra.main(version_base=None, config_path="config", config_name="jaxnav-ncc")
def main(config):
    config = OmegaConf.to_container(config)
    t_config = config["learning"]
    
    tags = ["RNN", "ts: "+config["env"]["test_set"], "sf: "+"learnability", "NCC"]

    run = wandb.init(
        group=config['GROUP_NAME'],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    
    env = JaxNav(num_agents=config["env"]["num_agents"],
                        **config["env"]["env_params"])
        
    t_config["NUM_ACTORS"] = env.num_agents * t_config["NUM_ENVS"]
    t_config["NUM_UPDATES"] = (
        t_config["TOTAL_TIMESTEPS"] // t_config["NUM_STEPS"] // t_config["NUM_ENVS"]
    )
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * t_config["NUM_STEPS"] // t_config["NUM_MINIBATCHES"]
    )
    t_config["CLIP_EPS"] = (
        t_config["CLIP_EPS"] / env.num_agents
        if t_config["SCALE_CLIP_EPS"]
        else t_config["CLIP_EPS"]
    )
    
    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    network = ActorCriticRNN(env.agent_action_space().shape[0],
                             fc_dim_size=t_config["FC_DIM_SIZE"],
                             hidden_size=t_config["HIDDEN_SIZE"],
                             use_layer_norm=t_config["USE_LAYER_NORM"],
                             use_elu=t_config["USE_ELU"])

    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, t_config["NUM_ENVS"], env.lidar_num_beams+5)  # NOTE hardcoded
        ),
        jnp.zeros((1, t_config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ENVS"], t_config["HIDDEN_SIZE"])
    network_params = network.init(_rng, init_hstate, init_x)

    if config["OPTIMISTIC"]:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            ti_ada(vy0 = jnp.zeros(config["ued"]["PLR_PARAMS"]["capacity"]), eta=linear_schedule),
            optax.scale_by_optimistic_gradient()
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            ti_ada(vy0 = jnp.zeros(config["ued"]["PLR_PARAMS"]["capacity"]), eta=linear_schedule),
        )

    eval_singleton_runner = EvalSingletonsRunner(
        config["env"]["test_set"],
        network,
        init_carry=ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        env_kwargs=config["env"]["env_params"]
    )

    with open(config["EVAL_SAMPLED_SET_PATH"], "rb") as f:
        eval_env_instances = pickle.load(f)
    _, eval_init_states = jax.vmap(env.set_env_instance, in_axes=(0))(eval_env_instances)

    eval_sampled_runner = EvalSampledRunner(
        None,
        env,
        network,
        ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        greedy=False,
        env_init_states=eval_init_states,
        n_episodes=10,
    )

    level_sampler = LevelSampler(**config["ued"]["PLR_PARAMS"])

    def sample_random_level(rng):
        obsv, env_state = env.reset(rng)
        return EnvInstance(
            agent_pos=env_state.pos,
            agent_theta=env_state.theta,
            goal_pos=env_state.goal,
            map_data=env_state.map_data,
            rew_lambda=env_state.rew_lambda,
        )
    
    rng, _rng = jax.random.split(rng)
    init_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["ued"]["PLR_PARAMS"]["capacity"]))
    sampler = level_sampler.initialize(init_levels, {"max_return": jnp.full(config["ued"]["PLR_PARAMS"]["capacity"], -jnp.inf)})

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
    )

    # Set up y optimizer state
    y_ti_ada = scale_y_by_ti_ada(eta=config["META_LR"])
    y_opt_state = y_ti_ada.init(jnp.zeros_like(train_state.sampler["scores"]))

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.global_done, 
                transition.value,
                transition.reward,
            )
            delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
            gae = (
                delta
                + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def log_eval(metrics):
        
        wandb.log(metrics, step=metrics["update_count"])

    def learnability_fn(rng, levels, num_envs, train_state):

        init_obs, new_levels = jax.vmap(env.set_env_instance)(levels)
        init_env_state = new_levels

        def rollout_fn(rng):

            rng, _rng = jax.random.split(rng)
            (_, last_env_state, last_obs, last_done, hstate, last_val, rng, disc_return), traj_batch = sample_trajectories_rnn(
                _rng,
                env,
                train_state,
                ScannedRNN.initialize_carry(num_envs, t_config["HIDDEN_SIZE"]),
                init_obs,
                init_env_state,
                t_config,
                num_envs,
                t_config["GAMMA"]
            )

            return traj_batch, jax.tree_map(lambda x: x.swapaxes(2, 1).reshape((-1, num_envs * env.num_agents)), traj_batch.info)
        
        @partial(jax.vmap, in_axes=(None, 2, 2, 2))
        @partial(jax.jit, static_argnums=(0,))
        def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
            idxs = jnp.arange(max_steps)
            
            @partial(jax.vmap, in_axes=(0, 0))
            def __ep_outcomes(start_idx, end_idx): 
                mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                r = jnp.sum(returns * mask)
                success = jnp.sum(info["GoalR"] * mask)
                collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                timeo = jnp.sum(info["TimeO"] * mask)
                l = end_idx - start_idx
                return r, success, collision, timeo, l
            
            done_idxs = jnp.argmax(dones, axis=1)
            mask_done = jnp.where(done_idxs == max_steps, 0, 1)
            ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                    
            jax.debug.breakpoint()
            return {"ep_return": ep_return.mean(where=mask_done),
                    "num_episodes": mask_done.sum(),
                    "success_rate": success.mean(where=mask_done),
                    "collision_rate": collision.mean(where=mask_done),
                    "timeout_rate": timeo.mean(where=mask_done),
                    "ep_len": length.mean(where=mask_done),
                    }

        rng, _rng = jax.random.split(rng)
        traj_batch, info_by_actor = jax.vmap(rollout_fn)(jax.random.split(rng, 10))
        
        o = _calc_outcomes_by_agent(t_config["NUM_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
        success_by_env = o["success_rate"].reshape((env.num_agents, num_envs))
        learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
        
        return learnability_by_env, o["ep_return"]

    def replace_fn(rng, train_state, old_level_scores):
        # NOTE: scores here are the actual UED scores, NOT the probabilities induced by the projection

        # Sample new levels
        rng, _rng = jax.random.split(rng)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, t_config["NUM_ENVS"]))

        rng, _rng = jax.random.split(rng)
        new_level_scores, max_returns = learnability_fn(_rng, new_levels, t_config["NUM_ENVS"], train_state)

        idxs = jnp.flipud(jnp.argsort(new_level_scores))

        new_levels = jax.tree_util.tree_map(
            lambda x: x[idxs], new_levels
        )
        new_level_scores = new_level_scores[idxs]

        update_sampler = {**train_state.sampler,"scores": old_level_scores}

        sampler, _ = level_sampler.insert_batch(update_sampler, new_levels, new_level_scores, {"max_return": max_returns})
        
        return sampler

    @jax.jit
    def train_step(carry, t):

        def callback(metrics):
            wandb.log(metrics, step=metrics["update_count"])

        rng, train_state, xhat, prev_grad, y_opt_state = carry

        new_score = projection_simplex_truncated(xhat + prev_grad, config["META_TRUNC"]) if config["META_OPTIMISTIC"] else xhat
        sampler = {**train_state.sampler, "scores": new_score}
            
        rng, rng_levels = jax.random.split(rng)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, t_config["NUM_ENVS"])
        init_obs, new_levels = jax.vmap(env.set_env_instance)(levels)
        init_env_state = new_levels
        
        init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng, _), traj_batch = sample_trajectories_rnn(
            rng,
            env,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            t_config,
            t_config["NUM_ENVS"]
        )                
        advantages, targets = _calculate_gae(traj_batch, last_val)
        train_info = traj_batch.info

        (rng, train_state), loss_info = update_actor_critic_rnn(rng, train_state, init_hstate, traj_batch, advantages, targets, t_config, update_grad=True)
                    
        # Update the level sampler
        levels = sampler["levels"]
        rng, _rng = jax.random.split(rng)
        scores, _ = learnability_fn(_rng, levels, config["ued"]["PLR_PARAMS"]["capacity"], train_state)

        rng, _rng = jax.random.split(rng)
        new_sampler = replace_fn(_rng, train_state, scores)
        sampler = {**new_sampler, "scores": new_score}

        grad, y_opt_state = y_ti_ada.update(new_sampler["scores"], y_opt_state)
        xhat = projection_simplex_truncated(xhat + grad, config["META_TRUNC"])

        train_state = train_state.replace(
            opt_state = jax.tree_util.tree_map(
                lambda x: x if type(x) is not ScaleByTiAdaState else x.replace(vy = y_opt_state.vy), train_state.opt_state
            ),
            sampler = sampler
        )

        # LOG
        train_info = jax.tree_map(lambda x: x.sum(axis=-1).reshape((t_config["NUM_STEPS"], t_config["NUM_ENVS"])).sum(), train_info)
        ratio_0 = loss_info[1][3].at[0,0].get().mean()
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
        metrics = {
            "meta_entropy": -jnp.dot(sampler["scores"], jnp.log(sampler["scores"] + 1e-6)),
            "meta_loss": new_sampler["scores"].T @ new_score,
            "loss/": {
               "total_loss": loss_info[0],
               "value_loss": loss_info[1][0],
               "actor_loss": loss_info[1][1],
               "entropy": loss_info[1][2],
               "ratio": loss_info[1][3],
               "ratio_0": ratio_0,
               "approx_kl": loss_info[1][4],
               "clipfrac": loss_info[1][5],
            },
            "terminations": {
                k: train_info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]
            },
        }
        
        metrics["update_count"] = t
        metrics["num_env_steps"] = t * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]
        
        jax.experimental.io_callback(callback, None, metrics)
        return (rng, train_state, xhat, grad, y_opt_state), metrics

    @jax.jit
    def train_and_eval_step(runner_state, t):
        
        # TRAIN
        (rng, train_state, xhat, grad, y_opt_state), metrics = jax.lax.scan(train_step, runner_state, jnp.arange(0, t_config["EVAL_FREQ"]) + t + 1, t_config["EVAL_FREQ"])
        
        # EVAL
        rng, eval_singleton_rng, eval_sampled_rng = jax.random.split(rng, 3)
        test_metrics = {}
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, train_state.params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, train_state.params)
        
        test_metrics["update_count"] = t + t_config["EVAL_FREQ"]
        
        return (rng, train_state, xhat, grad, y_opt_state), test_metrics

    def log_buffer(sampler, epoch):
        
        sorted_scores = jnp.argsort(sampler["scores"])
        top = sorted_scores[-20:]
        bottom = sorted_scores[:20]
        
        num_samples = 20
        rows_per = 2 
        fig, axes = plt.subplots(2*rows_per, int(num_samples/rows_per), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes[:num_samples]):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            idx = top[i]
            score = sampler["scores"][idx]
            level = jax.tree_map(lambda x: x[idx], sampler["levels"])
                        
            env.init_render(ax, level, lidar=False, ticks_off=True)
            ax.set_title(f'regret: {score:.3f}, \ntimestamp: {sampler["timestamps"][i]}')
            ax.set_aspect('equal', 'box')
            
        for i, ax in enumerate(axes[num_samples:]):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            idx = bottom[i]
            score = sampler["scores"][idx]
            level = jax.tree_map(lambda x: x[idx], sampler["levels"])
                        
            env.init_render(ax, level, lidar=False, ticks_off=True)
            ax.set_title(f'regret: {score:.3f}, \ntimestamp: {sampler["timestamps"][i]}')
            ax.set_aspect('equal', 'box')
            
        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) 
        wandb.log({"maps": wandb.Image(im)}, step=epoch)

    print('eval step', t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])
    print('num updates', t_config["NUM_UPDATES"])
    
    checkpoint_steps = t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"]
    xhat = sampler["scores"]
    grad = jnp.zeros_like(xhat)
    runner_state = (rng, train_state, xhat, grad, y_opt_state)
    for eval_step in range(0, int(t_config["NUM_UPDATES"]), t_config["EVAL_FREQ"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, eval_step)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta']
        log_eval(metrics)  #, train_state_to_log_dict(runner_state[1], level_sampler) add?

        sampler_with_states = {**runner_state[1].sampler, "levels": jax.vmap(env.set_env_instance)(runner_state[1].sampler["levels"])[1]}
        log_buffer(sampler_with_states, metrics["update_count"])
        # if (eval_step % checkpoint_steps == 0) & (eval_step > 0):    
        #     if config["SAVE_PATH"] is not None:
        #         params = runner_state[1].params
                
        #         save_dir = os.path.join(config["SAVE_PATH"], run.name)
        #         os.makedirs(save_dir, exist_ok=True)
        #         save_params(params, f'{save_dir}/model.safetensors')
        #         print(f'Parameters of saved in {save_dir}/model.safetensors')
                
        #         # upload this to wandb as an artifact   
        #         artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        #         artifact.add_file(f'{save_dir}/model.safetensors')
        #         artifact.save()
    
    if config["SAVE_PATH"] is not None:
        params = runner_state[1].params
        
        save_dir = os.path.join(config["SAVE_PATH"], wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()

if __name__ == "__main__":
    main()
