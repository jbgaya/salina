#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import hydra
import os
import random
import torch
from salina import Workspace, instantiate_class
import salina.rl.functional as RLF
from salina.agents import Agents, TemporalAgent
from salina_examples.rl.LoP.agents import AlphaAgent, Normalizer, CustomBraxAgent
from salina.rl.replay_buffer import ReplayBuffer

def run_line_ppo(policy_agent, critic_agent, logger, cfg):

    # Instantiating acquisition agent
    normalizer_agent = Normalizer(cfg.acquisition.env)
    env_agent = CustomBraxAgent(cfg.acquisition.n_envs, **cfg.acquisition.env)
    alpha_agent = AlphaAgent(cfg.device, cfg.algorithm.n_models, cfg.algorithm.geometry, cfg.algorithm.distribution)
    acquisition_agent = TemporalAgent(Agents(env_agent, normalizer_agent, alpha_agent, policy_agent)).to(cfg.device)
    acquisition_agent.seed(cfg.acquisition.seed)
    workspace = Workspace()
    acquisition_agent(workspace, t = 0, n_steps = cfg.acquisition.n_timesteps, replay=False, update_normalizer = True, action_std=cfg.algorithm.action_std)

    # Instantiating validation agent
    env_agent = CustomBraxAgent(cfg.validation.n_envs, **cfg.validation.env)
    validation_agent = TemporalAgent(Agents(env_agent, normalizer_agent, alpha_agent, policy_agent)).to(cfg.device)
    validation_agent.seed(cfg.validation.seed)
    validation_workspace = Workspace()

    # Initializing optimizers
    optimizer_policy = torch.optim.Adam(policy_agent.parameters(), lr=cfg.algorithm.lr_policy)
    optimizer_critic = torch.optim.Adam(critic_agent.parameters(), lr=cfg.algorithm.lr_critic)

    # Using replay buffer
    print("[LinePPO] Initializing replay buffer")
    buffer_size = cfg.algorithm.minibatch_size * cfg.algorithm.num_minibatches
    replay_buffer = ReplayBuffer(buffer_size,device=cfg.device)
    replay_buffer.put(workspace)
    while replay_buffer.size() < buffer_size:
        workspace.copy_n_last_steps(1)
        acquisition_agent(workspace, t = 1, n_steps = cfg.acquisition.n_timesteps - 1, replay=False, update_normalizer = True, action_std=cfg.algorithm.action_std)
        replay_buffer.put(workspace)

    # Running algorithm
    epoch = 0
    iteration = 0
    while (epoch < cfg.algorithm.max_epochs):
        env_interactions = buffer_size * cfg.acquisition.n_timesteps * (epoch  + 1)
        logger.add_scalar("monitor/env_interactions", env_interactions, epoch)

        # Evaluating the training policy
        if (epoch % cfg.validation.evaluate_every == 0) and (epoch > 0):
            validation_agent(validation_workspace, t=0, stop_variable="env/done", replay=False, action_std=0.0, update_normalizer=False)
            creward, done = validation_workspace["env/cumulated_reward", "env/done"]
            creward = creward[done].reshape(cfg.validation.n_envs)
            logger.add_scalar("reward/mean", creward.mean().item(), env_interactions)
            logger.add_scalar("reward/max", creward.max().item(), env_interactions)
            print("reward at epoch", epoch, ":\t", round(creward.mean().item(), 0))

        for _ in range(cfg.algorithm.update_epochs * cfg.algorithm.num_minibatches):
            miniworkspace = replay_buffer.get(cfg.algorithm.minibatch_size)

            # Updating policy with a cosine similarity penalty
            critic_agent(miniworkspace, replay = True)
            critic, done, reward = miniworkspace["critic", "env/done", "env/reward"]
            old_action_lp = miniworkspace["old_action_logprobs"].detach()
            reward = reward * cfg.algorithm.reward_scaling
            gae = RLF.gae(critic, reward, done, cfg.algorithm.discount_factor, cfg.algorithm.gae).detach()
            policy_agent(miniworkspace, t = None, replay=True, action_std = cfg.algorithm.action_std)
            action_lp = miniworkspace["action_logprobs"]
            ratio = action_lp - old_action_lp
            ratio = ratio.exp()
            ratio = ratio[:-1]
            clip_adv = torch.clamp(ratio, 1 - cfg.algorithm.clip_ratio, 1 + cfg.algorithm.clip_ratio) * gae
            loss_policy = -(torch.min(ratio * gae, clip_adv)).mean()        
            j,k = random.sample(range(cfg.algorithm.n_models),2)
            penalty = policy_agent.cosine_similarity(j,k)
            logger.add_scalar("loss/weight_penalty_policy", penalty.item(), iteration)
            loss = loss_policy + penalty * cfg.algorithm.beta
            optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), cfg.algorithm.clip_grad)
            optimizer_policy.step()
            logger.add_scalar("loss/policy", loss_policy.item(), iteration)

            # Updating critic
            td0 = RLF.temporal_difference(critic, reward, done, cfg.algorithm.discount_factor)
            loss = (td0 ** 2).mean()
            optimizer_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_agent.parameters(), cfg.algorithm.clip_grad)
            optimizer_critic.step()
            logger.add_scalar("loss/critic", loss.item(), iteration)
            iteration += 1

        # Acquisition
        for _ in range(buffer_size // cfg.acquisition.n_envs):
            workspace.copy_n_last_steps(1)
            acquisition_agent(workspace, t = 1, n_steps = cfg.acquisition.n_timesteps - 1, replay=False, update_normalizer = True, action_std=cfg.algorithm.action_std)
            replay_buffer.put(workspace)
        epoch += 1

    # Saving model
    
    os.makedirs(cfg.logger.log_dir +"/model")
    torch.save(policy_agent.state_dict(),cfg.logger.log_dir +"/model/policy")
    torch.save(normalizer_agent.state_dict(),cfg.logger.log_dir +"/model/normalizer")
            
@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    # For initializing brax
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    policy_agent = instantiate_class(cfg.policy_agent).to(cfg.device)
    critic_agent = instantiate_class(cfg.critic_agent).to(cfg.device)
    logger = instantiate_class(cfg.logger)
    mp.set_start_method("spawn")
    run_line_ppo(policy_agent, critic_agent, logger, cfg)

if __name__ == "__main__":
    main()