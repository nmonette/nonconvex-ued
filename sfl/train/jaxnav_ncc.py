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

from jaxmarl.environments.jaxnav import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_ued_utils import make_level_mutator

from sfl.util.jaxued.level_sampler import LevelSampler
from sfl.util.jaxued.jaxued_utils import compute_max_returns, l1_value_loss, max_mc, positive_value_loss

from sfl.train.train_utils import save_params
from sfl.train.minigrid_ncc import TrainState, LevelSampler, replace_fn
from sfl.train.jaxnav_plr import update_actor_critic_rnn, sample_trajectories_rnn
from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.util.rl.plr import PLRManager, PLRBuffer
from sfl.util.rl.ued_scores import UEDScore, compute_ued_scores
from sfl.util.ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    fc_dim_size: int = 512
    hidden_size: int = 512
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
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
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    last_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: jnp.ndarray

class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    targets: jnp.ndarray
    advantages: jnp.ndarray
    # carry: jnp.ndarray
    mask: jnp.ndarray

def compute_score(config, dones, values, max_returns, advantages):
    if config['SCORE_FUNCTION'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['SCORE_FUNCTION'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['SCORE_FUNCTION'] == "l1vl":
        return l1_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['SCORE_FUNCTION']}")
    
def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

hydra.main(version_base=None, config_path="config", config_name="jaxnav-plr")
def main(config):
    config = OmegaConf.to_container(config)
    t_config = config["learning"]
    
    tags = ["RNN", "ts: "+config["env"]["test_set"], "sf: "+config["ued"]["SCORE_FUNCTION"], "ncc"]
    
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
    
    # get ued score
    print('Using UED Score:', config["ued"]["SCORE_FUNCTION"])

    network = ActorCriticRNN(env.agent_action_space().shape[0],
                             fc_dim_size=t_config["FC_DIM_SIZE"],
                             hidden_size=t_config["HIDDEN_SIZE"],
                             use_layer_norm=t_config["USE_LAYER_NORM"],)

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
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            ti_ada(vy0 = jnp.zeros(config["PLR_PARAMS"]["capacity"]), eta=linear_schedule),
            optax.scale_by_optimistic_gradient()
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            ti_ada(vy0 = jnp.zeros(config["PLR_PARAMS"]["capacity"]), eta=linear_schedule),
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

    sample_random_level = env.sample_test_case
    mutate_level = make_level_mutator(50, env.map_obj)
    level_sampler = LevelSampler(**config["ued"]["PLR_PARAMS"])
    
    rng, _rng = jax.random.split(rng)
    pholder_level = sample_random_level(_rng) 
    sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_map(lambda x: jnp.array([x]).repeat(t_config["NUM_ENVS"], axis=0), pholder_level)
    
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler  
    )

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
        
        wandb.log(metrics)
    
    @jax.jit
    def replace_fn(rng, train_state, old_level_scores):
        # NOTE: scores here are the actual UED scores, NOT the probabilities induced by the projection

        # Sample new levels
        rng, _rng = jax.random.split(rng)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["NUM_ENVS"]))

        # Get the scores of the levels
        rng, _rng = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, config["NUM_ENVS"]), new_levels, env_params)
        # Rollout
        (
            train_state, last_env_state, last_obs, last_done, hstate, last_val, rng,
        ) = sample_trajectories_rnn(
            rng,
            env,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            t_config,
        )
        advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        max_returns = compute_max_returns(dones, rewards)
        new_level_scores = compute_score(config, dones, values, max_returns, advantages)

        update_sampler = {**train_state.sampler,"scores": old_level_scores}

        sampler, _ = level_sampler.insert_batch(update_sampler, new_levels, new_level_scores, {"max_return": max_returns})
        
        return sampler
    
    @jax.jit
    def train_step(carry, unused):
        
        rng, train_state, xhat, prev_grad, y_opt_state = carry

        # COLLECT TRAJECTORIES
        def callback(metrics):
            wandb.log(metrics)

        new_score = projection_simplex_truncated(xhat + prev_grad, config["META_TRUNC"]) if config["META_OPTIMISTIC"] else xhat
        sampler = {**train_state.sampler, "scores": new_score}
        
        rng, rng_levels = jax.random.split(rng)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, t_config["NUM_ENVS"])
        init_obs, new_levels = jax.vmap(env.set_state)(levels)
        init_env_state = new_levels
        
        init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng), traj_batch_dormancy = sample_trajectories_rnn(
            rng,
            env,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            t_config,
        )
        traj_batch, dormancy = traj_batch_dormancy
        dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
                
        advantages, targets = _calculate_gae(traj_batch, last_val)
        
        max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
        max_returns_by_level = max_returns.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)
        max_returns_by_level = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], max_returns_by_level)
        scores = compute_score(config["ued"], traj_batch.done, traj_batch.value, max_returns, advantages)
        scores_by_level = scores.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)

        sampler = level_sampler.update_batch(sampler, level_inds, scores_by_level, {"max_return": max_returns_by_level})
        (rng, train_state), loss_info = update_actor_critic_rnn(rng, train_state, init_hstate, traj_batch, advantages, targets, t_config, update_grad=True)
                    
        train_state = train_state.replace(
            sampler=sampler,
            update_state=1, # UpdateState.REPLAY
            num_replay_updates=train_state.num_replay_updates + 1,
            replay_last_level_batch=levels,
        )

        rng, rng_levels, rng_reset = jax.random.split(rng, 3)

        level_inds = jnp.arange(config["PLR_PARAMS"]["capacity"])
        # levels = jax.tree_util.tree_map(
        #     lambda x: jnp.repeat(x, 5, axis=0), sampler["levels"]
        # )
        levels = sampler["levels"]
        
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["PLR_PARAMS"]["capacity"]), levels, env_params)
        (
            (rng, train_state, _, _, _, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((config["PLR_PARAMS"]["capacity"],)),
            init_obs,
            init_env_state,
            config["PLR_PARAMS"]["capacity"],
            config["NUM_STEPS"],
        )
        advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
        
        scores = compute_score(config, dones, values, max_returns, advantages)

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

        return (rng, train_state), loss_info