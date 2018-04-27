import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
#import cv2

class BatchCartPoleEnv(gym.Env):
    metadata = {

    }
    def __init__(self,
            theta_thresh = np.deg2rad(12),
            x_thresh = 2.4,
            batch_size = 1
            ):
        self._batch_size = batch_size

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = theta_thresh
        self.x_threshold = x_thresh

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.asarray(action)
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state # [N, 4]
        x,v,q,w = state.T
        sq = np.sin(q)
        cq = np.cos(q)

        g = self.gravity
        m, pm, pml = self.total_mass, self.masspole, self.polemass_length
        l = self.length

        f = self.force_mag * (2 * action - 1.0)
        tmp = f + (l*pm/m)*w*w*sq

        qa = (g*sq-cq*tmp) / (l*(4.0/3.0 - pm*cq*cq/m))
        xa = tmp - (l*pm/m)*qa*cq

        x += self.tau*v
        v += self.tau*xa
        q += self.tau*w
        w += self.tau*qa

        done = np.logical_or(
                np.abs(x) > self.x_threshold,
                np.abs(q) > self.theta_threshold_radians
                )

        reward = np.ones_like(done, dtype=np.float32)
        return self.state.copy(), reward, done, {}

    def reset(self, index=None):
        if index is None:
            state = self.np_random.uniform(low=-0.05, high=0.05, size=(self._batch_size, 4))
            self.state = state
        else:
            state = self.np_random.uniform(low=-0.05, high=0.05, size=(len(index), 4))
            self.state[index] = state
        self.steps_beyond_done = None
        return self.state.copy()

    def render(self):
        print self.state

if __name__ == "__main__":
    N = 32
    env = BatchCartPoleEnv(batch_size = N)
    env.reset()

    ds = np.zeros((100, 32), dtype=np.float32)

    for j in range(100):
        a = [env.action_space.sample() for i in range(N)]
        s,r,d,_ = env.step(a)
        d_idx = np.where(d)[0]
        ds[j, d_idx] = 1.0
        if len(d_idx) > 0:
            print d_idx
            env.reset(d_idx)
        #env.render()

    #cv2.imshow('img', ds.T)
    #cv2.waitKey(0)
