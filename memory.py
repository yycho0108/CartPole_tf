import numpy as np
import random

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

class TraceMemory(object):
    """
    Dequeue Memory with Traced-Sample Support.
    """
    def __init__(self, size=10000):
        self._memory = [[] for _ in range(size)]
        self._size = size
        self._index = 0
        self._full = False

    def save(self, path):
        np.save(path, {
            'memory' : self._memory,
            'size' : self._size,
            'index' : self._index,
            'full' : self._full
            })

    def load(self, path):
        data = np.load(path)
        self._memory = data['memory']
        self._size = data['size']
        self._index = data['index']
        self._full = data['full']

    def add(self, memory):
        self._memory[self._index] = memory
        self._index += 1
        if self._index >= self._size:
            self._index = 0
            self._full = True

    def sample(self, n, s):
        """
        n : batch_size
        s : trace length
        """
        if self._full:
            idx = np.random.randint(self._size, size=n)
        else:
            idx = np.random.randint(self._index, size=n)

        res = []
        for i in idx:
            m = self._memory[i]
            i0 = np.random.randint(0, len(m)+1-s)
            res.append(m[i0:i0+s])

        res = np.asarray(res, dtype=np.float32)
        return res
