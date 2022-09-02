import jax
import jax.numpy as jnp
import haiku as hk


class BottleneckBlock(hk.Module):
    def __init__(self, k_bins, emb_width, mu):
        super(BottleneckBlock, self).__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        # self.register_buffer()