import os, random, pickle, pdb
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        try:
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
        except ValueError:
            pdb.set_trace()
        return state, action, reward, next_state, done

    def save(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, f'memory_{num}')
        dump_dict = {
            'capacity': self.capacity,
            'buffer': self.buffer,
            'position': self.position
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(dump_dict, f)

    def load(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, f'memory_{num}')
        with open(checkpoint_file, "rb") as f:
            dump_dict = pickle.load(f)
            self.capacity = dump_dict['capacity']
            self.buffer = dump_dict['buffer']
            self.position = dump_dict['position']

    def __len__(self):
        return len(self.buffer)
