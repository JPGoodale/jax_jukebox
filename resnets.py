import math
import jax
import jax.numpy as jnp
import haiku as hk


class ResConv2DBlock(hk.Module):
    def __init__(self, n_in, n_state):
        super(ResConv2DBlock, self).__init__()
        self.block = hk.Sequential([
            jax.nn.relu,
            hk.Conv2D(
                output_channels=n_state,
                kernel_shape=3,
                stride=1,
                padding=1
            ),
            jax.nn.relu,
            hk.Conv2D(
                output_channels=n_in,
                kernel_shape=1,
                stride=1,
                padding=0
            )
        ])

    def __call__(self, x):
        return x + self.block(x)


class ResNet2D(hk.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super(ResNet2D, self).__init__()
        self.net = hk.Sequential(*[
            ResConv2DBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)
        ])

    def forward(self, x):
        return self.net(x)


class ResConv1DBlock(hk.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super(ResConv1DBlock, self).__init__()
        padding = dilation
        self.block = hk.Sequential([
            jax.nn.relu,
            hk.Conv1D(
                output_channels=n_state,
                kernel_shape=3,
                stride=1,
                rate=dilation,
                padding=padding,
            ),
            jax.nn.relu,
            hk.Conv1D(
                output_channels=n_in,
                kernel_shape=1,
                stride=1,
                padding=0,
            )
        ])
        if zero_out:
            out = self.block[-1]
            hk.get_parameter(
                name='w',
                shape=out,
                init=jnp.zeros,
            )
            hk.get_parameter(
                name='b',
                shape=out,
                init=jnp.zeros,
            )
        self.res_scale = res_scale

    def _call__(self, x):
        return x + self.res_scale * self.block(x)


class Resnet1D(hk.Module):
    def __init__(
            self,
            n_in,
            n_depth,
            m_conv=1.0,
            dilation_growth_rate=1,
            dilation_cycle=None,
            zero_out=False,
            res_scale=False,
            reverse_dilation=False,
            checkpoint_res=False):
        super(Resnet1D, self).__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return  depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in, int(m_conv*n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)
             )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.net = hk.Sequential(*blocks)

    def __call__(self, x):
            return  self.net(x)





