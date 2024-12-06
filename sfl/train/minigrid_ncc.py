"""
Nonconvex-Concave Regret Optimization for UED
"""

import json
import time
from functools import partial
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
import chex
from enum import IntEnum
import hydra
from omegaconf import OmegaConf

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import BaseLevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

from sfl.util.jaxued.jaxued_utils import l1_value_loss
from sfl.train.train_utils import save_params
from sfl.train.minigrid_plr import (
    compute_gae,
    sample_trajectories_rnn,
    evaluate_rnn,
    update_actor_critic_rnn,
    ActorCritic,
    setup_checkpointing,
    train_state_to_log_dict,
    compute_score
)
from sfl.util.ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

class LevelSampler(BaseLevelSampler):

    def level_weights(self, sampler, *args,**kwargs):
        return sampler.scores
    
    def initialize(self, pholder_level, pholder_level_extra):
        sampler = {
                "levels": jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level),
                "scores": jnp.full(self.capacity, 1 / self.capacity, dtype=jnp.float32),
                "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
                "size": 0,
                "episode_count": 0,
        }
        if pholder_level_extra is not None:
            sampler["levels_extra"] = jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level_extra)
        return sampler


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)

@hydra.main(version_base=None, config_path="config", config_name="minigrid-ncc")
def main(config):

    d = {}
    for k, v in config.items():
        if isinstance(v, dict):
            d = d | v
        else:
            d[k] = v
    config = d
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    run = wandb.init(
        group=config.get('GROUP_NAME'),
        entity=config["ENTITY"],
        project=config["PROJECT"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["NUM_ENVS"] * config["NUM_STEPS"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["EVAL_LEVELS"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["EVAL_LEVELS"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        
        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["EVAL_LEVELS"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)

    # Setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["AGENT_VIEW_SIZE"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["N_WALLS"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # And the level sampler    
    # TODO: maybe need to subclass the LevelSampler to get rid of staleness and such
    level_sampler = LevelSampler(
        capacity=config["PLR_PARAMS"]["capacity"],
        replay_prob=config["PLR_PARAMS"]["replay_prob"],
        staleness_coeff=config["PLR_PARAMS"]["staleness_coeff"],
        minimum_fill_ratio=config["PLR_PARAMS"]["minimum_fill_ratio"],
        prioritization=config["PLR_PARAMS"]["prioritization"],
        prioritization_params=config["PLR_PARAMS"]["prioritization_params"],
        duplicate_check=config["PLR_PARAMS"]['duplicate_check'],
    )

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
            (rng, train_state, hstate, last_obs, last_env_state, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            init_obs,
            init_env_state,
            config["NUM_ENVS"],
            config["NUM_STEPS"],
        )
        advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        max_returns = compute_max_returns(dones, rewards)
        new_level_scores = compute_score(config, dones, values, max_returns, advantages)

        sampler, _ = level_sampler.insert_batch(sampler.replace(scores = old_level_scores), new_levels, new_level_scores, {"max_return": max_returns})
        
        return sampler

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["NUM_ENVS"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["NUM_ENVS"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["NUM_ENVS"],)))

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
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
        )
    
    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):

        rng, train_state, xhat, prev_grad, y_opt_state = carry

        sampler = train_state.sampler.replace(
            score = projection_simplex_truncated(xhat + prev_grad, config["META_TRUNC"]) if config["META_OPTISMISTIC"] else xhat
        )
            
        # Collect trajectories on replay levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["NUM_ENVS"])
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), levels, env_params)
        (
            (rng, train_state, hstate, last_obs, last_env_state, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            init_obs,
            init_env_state,
            config["NUM_ENVS"],
            config["NUM_STEPS"],
        )
        advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        
        # Update the policy using trajectories collected from replay levels
        (rng, train_state), losses = update_actor_critic_rnn(
            rng,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            (obs, actions, dones, log_probs, values, targets, advantages),
            config["NUM_ENVS"],
            config["NUM_STEPS"],
            config["NUM_MINIBATCHES"],
            config["UPDATE_EPOCHS"],
            config["CLIP_EPS"],
            config["ENT_COEF"],
            config["VF_COEF"],
            update_grad=True,
        )
        
        # Update the level sampler
        # Collect trajectories on replay levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["NUM_ENVS"])

        level_inds = jnp.arange(config["PLR_PARAMS"]["capacity"])
        levels = sampler["levels"]
        
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), levels, env_params)
        (
            (rng, train_state, _, _, _, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            init_obs,
            init_env_state,
            config["NUM_ENVS"],
            config["NUM_STEPS"],
        )
        advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
        
        scores = compute_score(config, dones, values, max_returns, advantages)

        grad, y_opt_state = ti_ada.update(scores, y_opt_state)

        xhat = projection_simplex_truncated(xhat + grad, config["META_TRUNC"])

        rng, _rng = jax.random.split(rng)
        sampler = replace_fn(_rng, train_state, scores)
                        
        metrics = {
            "losses": jax.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": levels.wall_map.sum() / config["NUM_ENVS"],
            "meta_entropy": -jnp.dot(sampler["scores"], jnp.log(sampler["scores"] + 1e-6)),
        }
        
        train_state = train_state.replace(
            opt_state = jax.tree_util.tree_map(
                lambda x: x if type(x) is not ScaleByTiAdaState else x.replace(vy = y_opt_state.vy), train_state.opt_state
            ),
            sampler = sampler
        )

        return (rng, train_state, xhat, grad, y_opt_state), metrics
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["EVAL_LEVELS"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["EVAL_LEVELS"])
        num_levels = len(config["EVAL_LEVELS"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state, xhat, prev_grad, y_opt_state), metrics = jax.lax.scan(train_step, runner_state, None, config["EVAL_FREQ"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["EVAL_NUM_ATTEMPTS"]), train_state)
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
         
        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
        
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state, xhat, prev_grad, y_opt_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["EVAL_NUM_ATTEMPTS"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['EVAL_LEVELS'])
        return states, cum_rewards, episode_lengths
    
    # Set up the train states
    rng = jax.random.PRNGKey(config["SEED"])
    rng_init, rng_train = jax.random.split(rng)
    train_state = create_train_state(rng_init)

    # Set up y optimizer state
    y_ti_ada = scale_y_by_ti_ada(eta=config["Y_LR"])
    y_opt_state = y_ti_ada.init(jnp.zeros_like(train_state.sampler["scores"]))
        
    xhat = grad = jnp.zeros_like(train_state.sampler["scores"])
    runner_state = (rng_train, train_state, xhat, grad, y_opt_state)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["NUM_UPDATES"] // config["EVAL_FREQ"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

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

    return runner_state[1]

if __name__=="__main__":
    main()