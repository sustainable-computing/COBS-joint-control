import pickle, pdb, os

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.optim as optim


class Replay_Memory():
    def __init__(self, memory_size=10):
        self.memory_size = memory_size
        self.len = 0
        self.rewards = []
        self.states = []
        self.n_states = []
        self.log_probs = []
        self.actions = []
        self.disturbance = []
        self.CC = []
        self.cc = []

    def sample_batch(self, batch_size):
        rand_idx = np.arange(-batch_size, 0, 1)
        batch_rewards = torch.stack([self.rewards[i] for i in rand_idx]).reshape(-1)
        batch_states = torch.stack([self.states[i] for i in rand_idx])
        batch_nStates = torch.stack([self.n_states[i] for i in rand_idx])
        batch_actions = torch.stack([self.actions[i] for i in rand_idx])
        batch_logprobs = torch.stack([self.log_probs[i] for i in rand_idx]).reshape(-1)
        batch_disturbance = torch.stack([self.disturbance[i] for i in rand_idx])
        batch_CC = torch.stack([self.CC[i] for i in rand_idx])
        batch_cc = torch.stack([self.cc[i] for i in rand_idx])
        # Flatten
        _, _, n_state = batch_states.shape
        batch_states = batch_states.reshape(-1, n_state)
        batch_nStates = batch_nStates.reshape(-1, n_state)
        _, _, n_action = batch_actions.shape
        batch_actions = batch_actions.reshape(-1, n_action)
        _, _, T, n_dist = batch_disturbance.shape
        batch_disturbance = batch_disturbance.reshape(-1, T, n_dist)
        _, _, T, n_tau, n_tau = batch_CC.shape
        batch_CC = batch_CC.reshape(-1, T, n_tau, n_tau)
        batch_cc = batch_cc.reshape(-1, T, n_tau)
        return batch_states, batch_actions, batch_nStates, batch_disturbance, batch_rewards, batch_logprobs, batch_CC, batch_cc

    def append(self, states, actions, next_states, rewards, log_probs, dist, CC, cc):
        self.rewards.append(rewards)
        self.states.append(states)
        self.n_states.append(next_states)
        self.log_probs.append(log_probs)
        self.actions.append(actions)
        self.disturbance.append(dist)
        self.CC.append(CC)
        self.cc.append(cc)
        self.len += 1

        if self.len > self.memory_size:
            self.len = self.memory_size
            self.rewards = self.rewards[-self.memory_size:]
            self.states = self.states[-self.memory_size:]
            self.log_probs = self.log_probs[-self.memory_size:]
            self.actions = self.actions[-self.memory_size:]
            self.nStates = self.n_states[-self.memory_size:]
            self.disturbance = self.disturbance[-self.memory_size:]
            self.CC = self.CC[-self.memory_size:]
            self.cc = self.cc[-self.memory_size:]


class PPOLearner:
    def __init__(self, planning_steps, n_ctrl, n_state, target, disturbance, eta, u_upper, u_lower,
                 step=300*3, lr=5e-4, clip_param=0.1, F_hat=None, Bd_hat=None):

        self.replay_memory = Replay_Memory()
        self.clip_param = clip_param

        self.T = planning_steps
        self.step = step
        self.lr = lr
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        self.eta = eta

        self.target = target
        self.dist = disturbance
        self.n_dist = self.dist.shape[1]

        if F_hat is not None:  # Load pre-trained F if provided
            print("Load pretrained F")
            self.F_hat = torch.tensor(F_hat).double().requires_grad_()
            print(self.F_hat)
        else:
            self.F_hat = torch.ones((self.n_state, self.n_state + self.n_ctrl))
            self.F_hat = self.F_hat.double().requires_grad_()

        if Bd_hat is not None:  # Load pre-trained Bd if provided
            print("Load pretrained Bd")
            self.Bd_hat = Bd_hat
        else:
            self.Bd_hat = 0.1 * np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        print(self.Bd_hat)

        self.Bd_hat_old = self.Bd_hat.detach().clone()
        self.F_hat_old = self.F_hat.detach().clone()

        self.optimizer = optim.RMSprop([self.F_hat, self.Bd_hat], lr=self.lr)

        self.u_lower = u_lower * torch.ones(n_ctrl).double()
        self.u_upper = u_upper * torch.ones(n_ctrl).double()

    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, x_init, ft, C, c, current=True, n_iters=20):
        T, n_batch, n_dist = ft.shape
        if current == True:
            F_hat = self.F_hat
            Bd_hat = self.Bd_hat
        else:
            F_hat = self.F_hat_old
            Bd_hat = self.Bd_hat_old

        x_lqr, u_lqr, objs_lqr = mpc.MPC(n_state=self.n_state,
                                         n_ctrl=self.n_ctrl,
                                         T=self.T,
                                         u_lower=self.u_lower.repeat(self.T, n_batch, 1),
                                         u_upper=self.u_upper.repeat(self.T, n_batch, 1),
                                         lqr_iter=n_iters,
                                         backprop=True,
                                         verbose=0,
                                         exit_unconverged=False,
                                         )(x_init.double(), QuadCost(C.double(), c.double()),
                                           LinDx(F_hat.repeat(self.T - 1, n_batch, 1, 1), ft.double()))
        return x_lqr, u_lqr

    def select_action(self, mu, sigma):
        if self.n_ctrl > 1:
            sigma_sq = torch.ones(mu.size()).double() * sigma ** 2
            dist = MultivariateNormal(mu, torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            dist = Normal(mu, torch.ones_like(mu) * sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, mu, actions, sigma):
        n_batch = len(mu)
        if self.n_ctrl > 1:
            cov = torch.eye(self.n_ctrl).double() * sigma ** 2
            cov = cov.repeat(n_batch, 1, 1)
            dist = MultivariateNormal(mu, cov)
        else:
            dist = Normal(mu, torch.ones_like(mu) * sigma)
        log_prob = dist.log_prob(actions.double())
        entropy = dist.entropy()
        return log_prob, entropy

    def update_parameters(self, loader, sigma):
        for i in range(1):
            for states, actions, next_states, dist, advantage, old_log_probs, C, c in loader:
                n_batch = states.shape[0]
                advantage = advantage.double()
                f = self.Dist_func(dist, current=True)  # T-1 x n_batch x n_state
                opt_states, opt_actions = self.forward(states, f, C.transpose(0, 1), c.transpose(0, 1),
                                                       current=True)  # x, u: T x N x Dim.
                log_probs, entropies = self.evaluate_action(opt_actions[0], actions, sigma)

                tau = torch.cat([states, actions], 1)  # n_batch x (n_state + n_ctrl)
                nState_est = torch.bmm(self.F_hat.repeat(n_batch, 1, 1), tau.unsqueeze(-1)).squeeze(-1) + f[
                    0]  # n_batch x n_state
                mse_loss = torch.mean((nState_est - next_states) ** 2)

                ratio = torch.exp(log_probs.squeeze() - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                loss = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_([self.F_hat, self.Bd_hat], 100)
                self.optimizer.step()

            self.F_hat_old = self.F_hat.detach().clone()
            self.Bd_hat_old = self.Bd_hat.detach().clone()
            print(self.F_hat)
            print(self.Bd_hat)

    def Dist_func(self, d, current=False):
        if current:  # d in n_batch x n_dist x T-1
            n_batch = d.shape[0]
            ft = torch.bmm(self.Bd_hat.repeat(n_batch, 1, 1), d)  # n_batch x n_state x T-1
            ft = ft.transpose(1, 2)  # n_batch x T-1 x n_state
            ft = ft.transpose(0, 1)  # T-1 x n_batch x n_state
        else:  # d in n_dist x T-1
            ft = torch.mm(self.Bd_hat_old, d).transpose(0, 1)  # T-1 x n_state
            ft = ft.unsqueeze(1)  # T-1 x 1 x n_state
        return ft

    def Cost_function(self, cur_time):
        diag = torch.zeros(self.T, self.n_state + self.n_ctrl)
        occupied = self.dist["Occupancy Flag"][cur_time: cur_time + pd.Timedelta(seconds=(self.T - 1) * self.step)]  # T
        eta_w_flag = torch.tensor([self.eta[int(flag)] for flag in occupied]).unsqueeze(1).double()  # Tx1
        diag[:, :self.n_state] = eta_w_flag
        diag[:, self.n_state:] = 1e-6

        # pdb.set_trace()
        C = []
        for i in range(self.T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1)  # T x 1 x (m+n) x (m+n)

        x_target = self.target[cur_time: cur_time + pd.Timedelta(seconds=(self.T - 1) * self.step)]  # in pd.Series
        x_target = torch.tensor(np.array(x_target))

        c = torch.zeros(self.T, self.n_state + self.n_ctrl)  # T x (m+n)
        c[:, :self.n_state] = -eta_w_flag * x_target
        c[:, self.n_state:] = 1  # L1-norm now!

        c = c.unsqueeze(1)  # T x 1 x (m+n)
        return C, c

class Dataset(data.Dataset):
    def __init__(self, states, actions, next_states, disturbance, rewards, old_logprobs, CC, cc):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.disturbance = disturbance
        self.rewards = rewards
        self.old_logprobs = old_logprobs
        self.CC = CC
        self.cc = cc

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.next_states[index], self.disturbance[index], self.rewards[index], self.old_logprobs[index], self.CC[index], self.cc[index]


class PPOAgent:
    def __init__(self, tol_eps, learner, multiplier, gamma, update_episode, obs_name, save_path):
        self.tol_eps = tol_eps
        self.learner = learner
        self.multiplier = multiplier
        self.gamma = gamma
        self.update_episode = update_episode
        self.obs_name = obs_name
        # globals vars
        self.timeStamp = []
        self.observations = []
        self.actions_taken = []
        self.perf = []
        # episode vars
        self.log_probs = []
        self.rewards = []
        self.real_rewards = []
        self.old_log_probs = []
        self.states = []
        self.disturbance = []
        self.actions = []  # Save for Parameter Updates
        self.CC = []
        self.cc = []
        self.sigma = 1
        self.save_path = save_path

    def agent_start(self, obvs, i_episode):
        (last_state, obs_dict, obs, cur_time) = obvs
        self.log_probs = []
        self.rewards = []
        self.real_rewards = []
        self.old_log_probs = []
        self.states = []
        self.disturbance = []
        self.actions = []  # Save for Parameter Updates
        self.CC = []
        self.cc = []
        self.sigma = 1 - 0.9 * i_episode / self.tol_eps

        td = pd.Timedelta(seconds=(self.learner.T - 2) * self.learner.step)
        dt = np.array(self.learner.dist[cur_time: cur_time + td])  # T-1 x n_dist
        dt = torch.tensor(dt).transpose(0, 1)  # n_dist x T-1
        ft = self.learner.Dist_func(dt)  # T-1 x 1 x n_state
        C, c = self.learner.Cost_function(cur_time)
        opt_states, opt_actions = self.learner.forward(last_state, ft, C, c, current=False)  # x, u: T x 1 x Dim.
        action, old_log_prob = self.learner.select_action(opt_actions[0], self.sigma)
        if action.item() < 0:
            action = torch.zeros_like(action)
        SAT_stpt = obs_dict["MA Temp."] + max(0, action.item())
        # If the room gets too warm during occupied period, uses outdoor air for free cooling.
        if (obs_dict["Indoor Temp."] > obs_dict["Indoor Temp. Setpoint"]) & (obs_dict["Occupancy Flag"] == 1):
            SAT_stpt = obs_dict["Outdoor Temp."]

        self.old_log_probs.append(old_log_prob)
        self.CC.append(C)
        self.cc.append(c)
        # vv state vars
        self.disturbance.append(dt)
        self.states.append(last_state)
        if len(self.observations) == 0: self.observations.append(obs)
        # vv action vars
        self.actions.append(action)
        self.actions_taken.append([action.item(), SAT_stpt])
        if len(self.timeStamp) == 0: self.timeStamp.append(cur_time)

        return action, SAT_stpt

    def agent_step(self, reward, obvs):
        (last_state, obs_dict, obs, cur_time) = obvs
        self.real_rewards.append(reward)
        self.rewards.append(reward.double() / self.multiplier)

        dt = np.array(self.learner.dist[cur_time: cur_time + pd.Timedelta(
            seconds=(self.learner.T - 2) * self.learner.step)])  # T-1 x n_dist
        dt = torch.tensor(dt).transpose(0, 1)  # n_dist x T-1
        ft = self.learner.Dist_func(dt)  # T-1 x 1 x n_state
        C, c = self.learner.Cost_function(cur_time)
        opt_states, opt_actions = self.learner.forward(last_state, ft, C, c, current=False)  # x, u: T x 1 x Dim.
        action, old_log_prob = self.learner.select_action(opt_actions[0], self.sigma)
        if action.item() < 0:
            action = torch.zeros_like(action)
        SAT_stpt = obs_dict["MA Temp."] + max(0, action.item())
        # If the room gets too warm during occupied period, uses outdoor air for free cooling.
        if (obs_dict["Indoor Temp."] > obs_dict["Indoor Temp. Setpoint"]) & (obs_dict["Occupancy Flag"] == 1):
            SAT_stpt = obs_dict["Outdoor Temp."]

        self.old_log_probs.append(old_log_prob)
        self.CC.append(C)
        self.cc.append(c)
        # vv state vars
        self.disturbance.append(dt)
        self.states.append(last_state)
        self.observations.append(obs)
        self.timeStamp.append(cur_time)
        # vv action vars
        self.actions.append(action)
        self.actions_taken.append([action.item(), SAT_stpt])

        return action, SAT_stpt

    def agent_end(self, reward, obvs, i_episode):
        (last_state, obs_dict, obs, cur_time) = obvs
        dt = np.array(self.learner.dist[cur_time: cur_time + pd.Timedelta(
            seconds=(self.learner.T - 2) * self.learner.step)])
        dt = torch.tensor(dt).transpose(0, 1)
        (last_state, obs_dict, obs, _) = obvs
        self.real_rewards.append(reward)
        self.rewards.append(reward.double() / self.multiplier)
        # vv state vars
        self.disturbance.append(dt)
        self.states.append(last_state)
        self.observations.append(obs)
        self.timeStamp.append(cur_time)

        self.store_memory()
        self.end_episode(cur_time, i_episode)

    def end_episode(self, cur_time, i_episode):
        # if -1, do not update parameters
        # print('lengths', len(self.observations), len(self.timeStamp), len(self.actions_taken))

        if self.update_episode == -1:
            pass
        elif (self.learner.memory.len >= self.update_episode) & (i_episode % self.update_episode == 0):
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, \
            batch_old_logprobs, batch_CC, batch_cc = self.learner.memory.sample_batch(self.update_episode)
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards,
                                batch_old_logprobs, batch_CC, batch_cc)
            # pdb.set_trace()
            batch_loader = data.DataLoader(batch_set, batch_size=48, shuffle=True, num_workers=2)
            self.learner.update_parameters(batch_loader, self.sigma)

        self.perf.append([np.mean(self.real_rewards), np.std(self.real_rewards)])
        print("{}, reward: {}".format(cur_time, np.mean(self.real_rewards)))

    def store_memory(self):
        def advantage_func(rewards, gamma):
            r = torch.zeros(1, 1).double()
            t = len(rewards)
            advantage = torch.zeros((t, 1)).double()

            for i in reversed(range(len(rewards))):
                r = gamma * r + rewards[i]
                advantage[i] = r
            return advantage

        advantages = advantage_func(self.rewards, self.gamma)
        old_log_probs = torch.stack(self.old_log_probs).squeeze().detach().clone()
        next_states = torch.stack(self.states[1:]).squeeze(1)
        states = torch.stack(self.states[:-1]).squeeze(1)
        actions = torch.stack(self.actions).squeeze(1).detach().clone()
        CC = torch.stack(self.CC).squeeze()  # n_batch x T x (m+n) x (m+n)
        cc = torch.stack(self.cc).squeeze()  # n_batch x T x (m+n)
        disturbance = torch.stack(self.disturbance)  # n_batch x T x n_dist
        self.learner.memory.append(states, actions, next_states, advantages, old_log_probs, disturbance, CC, cc)
