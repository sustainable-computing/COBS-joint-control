class Logger:
    def __init__(self, fname):
        self.fname = fname
        if self.fname is None:
            print('Will not log')
        else:
            with open(self.fname, 'w') as f:
                f.write('Starting ====\n')

    def __call__(self, s):
        if self.fname is not None:
            with open(self.fname, 'a+') as f:
                f.write(s + '\n')

    def log_start(self, i_episode, state, action):
        if self.fname is not None:
            s = f"Starting episode {i_episode}:\nCalling agent_start with state:"
            with open(self.fname, 'a+') as f:
                f.write(s + '\n')
            self.log_state(state)
            with open(self.fname, 'a+') as f:
                f.write(f'    and Action: {action}\n')

    def log_step(self, i_episode_step, action, state, reward, term):
        if self.fname is not None:
            s = f"Episode step {i_episode_step}:"
            with open(self.fname, 'a+') as f:
                f.write('  ' + s + '\n')
                f.write(f'    Action Taken: {action}\n')
                f.write(f'    Reward: {reward}\n')
                f.write(f'    Terminal: {term}\n')
            self.log_state(state)

    def log_state(self, s):
        if self.fname is not None:
            (state, obs_dict, cur_time) = s
            with open(self.fname, 'a+') as f:
                f.write(f"\tState:      {state}\n")
                f.write(f"\tObs. Dict:  {obs_dict}\n")
                f.write(f"\tCur. Time:  {cur_time}\n")
