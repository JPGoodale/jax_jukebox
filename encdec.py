import jax
import jax.numpy as jnp
import haiku as hk
from resnets import ResNet2D, ResNet1D


class EncoderConvBlock(hk.Module):
    def __init__(
            self,
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            m_conv,
            dilation_growth_rate=1,
            dilation_cycle=None,
            zero_out=False,
            res_scale=False,
    ):
        super(EncoderConvBlock, self).__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = hk.Sequential([
                    hk.Conv1D(
                        output_channels=width,
                        kernel_shape=filter_t,
                        stride=stride_t,
                        padding=pad_t,
                    ),
                    ResNet1D(
                        n_in=width,
                        n_depth=depth,
                        m_conv=m_conv,
                        dilation_growth_rate=dilation_growth_rate,
                        dilation_cycle=dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                    ),
                ])
                blocks.append(block)
            block = hk.Conv1D(
                output_channels=output_emb_width,
                kernel_shape=3,
                stride=1,
                padding=1,
            )
            blocks.append(block)
        self.net = hk.Sequential(*[blocks])

    def __call__(self, x):
        return self.net(x)


class DecoderConvBlock(hk.Module):
    def __init__(
            self,
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            m_conv,
            dilation_growth_rate=1,
            dilation_cycle=None,
            zero_out=False,
            res_scale=False,
            reverse_decoder_dilation=False,
    ):
        super(DecoderConvBlock, self).__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = hk.Conv1D(
                output_channels=width,
                kernel_shape=3,
                stride=1,
                padding=1,
            )
            blocks.append(block)
            for i in range(down_t):
                block = hk.Sequential([
                    ResNet1D(
                        n_in=width,
                        n_depth=depth,
                        m_conv=m_conv,
                        dilation_growth_rate=dilation_growth_rate,
                        dilation_cycle=dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation,
                    ),
                    hk.Conv1DTranspose(
                        output_channels=input_emb_width if i == (down_t - 1) else width, 
                        kernel_shape=filter_t,
                        stride=stride_t,
                        padding=pad_t
                    )
                ])
                blocks.append(block)
        self.net = hk.Sequential(*[blocks])
    
    def __call__(self, x):
        return self.net(x)


class Encoder(hk.Module):
    def __init__(
            self,
            input_emb_width,
            output_emb_width,
            levels,
            downs_t,
            strides_t,
            **block_kwargs,
    ):
        super(Encoder, self).__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        level_block = lambda level, down_t, stride_t: \
            EncoderConvBlock(
                output_emb_width,
                down_t,
                stride_t,
                **block_kwargs_copy,
            )
        self.level_blocks = []
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def __call__(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        # assert_shape(x, (N, emb, T))  #custom util needs defined
        xs = []
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            # assert_shape(x, (N, emb, T))  #custom util needs defined
            xs.append(x)
        return xs


class Decoder(hk.Module):
    def __init__(
            self,
            input_emb_width,
            output_emb_width,
            levels,
            downs_t,
            strides_t,
            **block_kwargs,
    ):
        super(Decoder, self).__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: \
            DecoderConvBlock(
                output_emb_width,
                down_t,
                stride_t,
                **block_kwargs,
            )
        self.level_blocks = []
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = hk.Conv1D(
            output_channels=input_emb_width,
            kernel_shape=3,
            stride=1,
            padding=1,
        )

    def __call__(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        # assert_shape(x, (N, emb, T))   #custom util needs defined
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t**down_t)
            # assert_shape(x, (N, emb, T))   #custom util needs defined
            if level != 0 and all_levels:
                x = x + xs[level-1]

        x = self.out(x)
        return x



