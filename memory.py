import numpy as np

class Memory(object):
    def __init__(self, dimension=10, size=10000):
        self.memory = np.empty((size,dimension), dtype=np.float32)
        self.size = size
        self.index = 0 # keeps track of current size
        self.full = False
    def add(self, memory):
        self.memory[self.index,:] = memory
        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.full = True
    def sample(self, n):
        # data stored in columns
        # each column is one entry
        if self.full:
            idx = np.random.randint(self.size, size=n)
        else:
            idx = np.random.randint(self.index, size=n)
        return self.memory[idx,:]

