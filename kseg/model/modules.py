import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple
from perceiver.model.core import (
    PerceiverEncoder,
    PerceiverDecoder,
    InputAdapter,
    OutputAdapter,
    FourierPositionEncoding,
    TrainableQueryProvider,
)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class MLP_Block(nn.Module):
    """Building block for MLP-based models."""

    def __init__(self, hidden_size: int, activation: nn.Module, depth: int) -> None:
        """Initialization of the MLP block.

        Args:
            hidden_size: Number of neurons in the linear layer.
            activation: Activation function.
            depth: Number of MLP blocks (linear layer with activation).
        """
        super(MLP_Block, self).__init__()
        layers = []
        for _ in range(depth):
            linear = nn.Linear(hidden_size, hidden_size)
            layers.append(linear)
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Propagates the input through the MLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
    ) -> None:
        """Initialization of the multi-layer perceptron.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of hidden layers. Defaults to 1.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_len, hidden_size),  # Input layer
                MLP_Block(hidden_size, nn.Tanh(), depth),
                nn.Linear(hidden_size, output_len),  # Output layer
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the network.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        # Rearrange to let the net process the input sagittally
        x = rearrange(x, 'b c v x y z -> b x (c v y z)')
        x = self.layers(x)
        return rearrange(
            x,
            'b x (c v y z) -> b c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


class SkipMLP(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth_per_block: int = 1,
        depth: int = 1,
    ) -> None:
        """Initialization of the multi-layer perceptron with skip connections.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth_per_block: Number of hidden layers per skip block.
                Defaults to 1.
            depth: Number of MLP blocks with skip conncection. Defaults to 1.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [nn.Linear(input_len, hidden_size)]  # Input layer
            + [
                Residual(
                    PreNorm(
                        hidden_size,
                        MLP_Block(
                            hidden_size,
                            nn.Tanh(),
                            depth_per_block,
                        ),
                    )
                )
                for i in range(depth)
            ]
            + [nn.Linear(hidden_size, output_len)]  # Output layer
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the network.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        # Rearrange to let the net process the input sagittally
        x = rearrange(x, 'b c v x y z -> b x (c v y z)')
        x = self.layers(x)
        return rearrange(
            x,
            'b x (c v y z) -> b c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


class ResMLP_Block(nn.Module):
    """ResMLP building block."""

    def __init__(
        self, num_patches: Tuple[int], latent_dim: int, layerscale_init: float
    ):
        """Building block for the ResMLP.

        Args:
            num_patches: Number of patches.
            latent_dim: Length of the latent dimension.
            layerscale_init: Layerscale initialization.
        """
        super().__init__()
        self.affine_1 = Affine(latent_dim)
        self.affine_2 = Affine(latent_dim)
        self.linear_patches = nn.Linear(num_patches, num_patches)
        self.mlp_channels = MLP_Block(latent_dim, nn.GELU(), 1)

        self.layerscale_1 = nn.Parameter(
            layerscale_init * torch.ones((latent_dim))
        )  # LayerScale parameters
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones((latent_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the ResMLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        res_1 = rearrange(
            self.linear_patches(rearrange(self.affine_1(x), 'b x l -> b l x')),
            'b l x -> b x l',
        )
        x = x + self.layerscale_1 * res_1
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        return x


class ResMLP(nn.Module):
    """ResMLP model: Stacking the full network.
    (See https://arxiv.org/pdf/2105.03404.pdf)"""

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
        layerscale_init: float = 0.2,
    ):
        """Initialization of the ResMLP model.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of ResMLP blocks. Defaults to 1.
            layerscale_init: Layerscale for the normalization. Defaults to 0.2.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        num_patches = input_shape[2]
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [nn.Linear(input_len, hidden_size)]
            + [
                ResMLP_Block(
                    num_patches,
                    hidden_size,
                    layerscale_init,
                )
                for i in range(depth)
            ]
            + [
                Affine(hidden_size),
                nn.Linear(hidden_size, output_len),
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # Rearrange to let the net process the input sagittally
        x = rearrange(x, 'b c v x y z -> b x (c v y z)')
        x = self.layers(x)
        return rearrange(
            x,
            'b x (c v y z) -> b c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


class Affine(nn.Module):
    """Affine Layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class DomainInputAdapter(InputAdapter):
    """An InputAdapter which can handle data in k-space domain or pixel domain.

    Transforms and position-encodes task-specific input to generic encoder
    input of shape (B, M, C) where B is the batch size, M the input sequence
    length and C the number of key/value input channels. C is determined by the
    `num_input_channels` property of the `input_adapter`. In this case,
    x-slice dimension acts as input channels.
    """

    def __init__(self, input_shape: Tuple[int], num_frequency_bands: int) -> None:
        """Initialization of the domain input adapter for the PerceiverIO.

        Args:
            input_shape: Shape of the input.
            num_frequency_bands: Number of frequency bands for the positional
                encoding.
        """
        self.input_shape = input_shape
        spatial_shape = [*input_shape[:2], *input_shape[3:]]
        num_slices = input_shape[2]

        if num_frequency_bands == 0:
            super().__init__(num_input_channels=num_slices)
            self.position_encoding = None
        else:
            position_encoding = FourierPositionEncoding(
                input_shape=spatial_shape,
                num_frequency_bands=num_frequency_bands,
            )

            super().__init__(
                num_input_channels=num_slices
                + position_encoding.num_position_encoding_channels()
            )
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the initialized positional encoding with the input.

        Args:
            x: Input.

        Raises:
            ValueError: If positional encoding does not have the same shape as
                the input.

        Returns:
            Concatenated input and positional encoding.
        """
        b, *d = x.shape

        # Check if the given input shape and positional encoding are compatible
        if tuple(d) != self.input_shape:
            raise ValueError(
                f'Input shape {tuple(d)} different from \
                required shape {self.input_shape}'
            )

        # b (batch), c (class), v (real/imag-part), x, y, z
        x = rearrange(x, 'b c v x y z -> b (c v z y) x')
        if self.position_encoding is None:
            return x

        x_enc = self.position_encoding(b)
        return torch.cat([x, x_enc], dim=-1)


class SegmentationOutputAdapter(OutputAdapter):
    """Transforms generic decoder cross-attention output to segmentation map."""

    def __init__(self, output_len: int, num_output_query_channels: int) -> None:
        """Initialization of the segmentation output adapter.

        Args:
            output_len: Desired length of the output.
            num_output_query_channels: Desired number of output query channels.
        """
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies linear layer to the input to generate desired output shape.

        Args:
            x: Input.
        Returns:
            Output in desired shape.
        """
        return self.linear(x)


class PerceiverIO(nn.Module):
    """Implementation of PerceiverIO
        (See https://github.com/krasserm/perceiver-io).

    The model uses a specified encoder and decoder.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        num_frequency_bands: int,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int,
        num_cross_attention_layers: int,
        num_self_attention_heads: int,
        num_self_attention_layers_per_block: int,
        num_self_attention_blocks: int,
        dropout: float,
    ) -> None:
        """Initialization of the PerceiverIO model.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            num_frequency_bands: Number of frequency bands used for positional
                encoding.
            num_latents: Number of latent values.
            num_latent_channels: Number of latent channels.
            num_cross_attention_heads: Number of cross-attention heads.
            num_cross_attention_layers: Number of cross-attention layers.
            num_self_attention_heads: Number of self-attention heads.
            num_self_attention_layers_per_block: Number of self-attention
                layers per block.
            num_self_attention_blocks: Number of self-attention blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        output_len = int(np.prod(output_shape))
        output_query_channels = 1
        input_adapter = DomainInputAdapter(
            self.input_shape,
            num_frequency_bands,
        )
        output_query_provider = TrainableQueryProvider(
            num_queries=1,  # Number of output channels
            num_query_channels=output_query_channels,
            init_scale=0.02,  # scale for Gaussian query initialization
        )
        output_adapter = SegmentationOutputAdapter(output_len, output_query_channels)
        modules = (
            PerceiverEncoder(
                input_adapter=input_adapter,
                num_latents=num_latents,  # N
                num_latent_channels=num_latent_channels,  # D
                num_cross_attention_heads=num_cross_attention_heads,
                num_cross_attention_layers=num_cross_attention_layers,
                first_cross_attention_layer_shared=False,
                num_self_attention_heads=num_self_attention_heads,
                num_self_attention_layers_per_block=(
                    num_self_attention_layers_per_block
                ),
                num_self_attention_blocks=num_self_attention_blocks,
                first_self_attention_block_shared=True,
                dropout=dropout,
                init_scale=0.02,  # scale for Gaussian latent initialization
                activation_checkpointing=False,
                activation_offloading=False,
            ),
            PerceiverDecoder(
                output_adapter=output_adapter,
                output_query_provider=output_query_provider,
                num_latent_channels=num_latent_channels,
                num_cross_attention_qk_channels=4,
            ),
        )
        self.linear_nonlinear = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the PerceiverIO.

        Args:
            x: Input.

        Returns:
            Predicted values of shape (b, c, v, x, y, z).
        """
        x = self.linear_nonlinear(x)
        # b (batch), q (query), c (class), v (real/imag-part), x, y, z
        return rearrange(
            x,
            'b q (c v x y z) -> (b q) c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            x=self.output_shape[2],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
        heads: int = 2,
        num_frequency_bands: int = 0,
        dropout: float = 0.2,
    ) -> None:
        """Initialization of the transformer.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of transformer encoder layers. Defaults to 1.
        """
        super().__init__()
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        hidden_size = int(input_len * hidden_factor)
        self.input_adapter = DomainInputAdapter(input_shape, num_frequency_bands)

        in_linear = nn.Linear(input_len, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size, dropout
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layers, depth)
        out_linear = nn.Linear(hidden_size, output_len)

        self.layers = nn.ModuleList([in_linear, transformer_encoder, out_linear])
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the network.

        Args:
            x: Input of shape (b c v x y z).

        Returns:
            Output of the network.
        """
        x = self.input_adapter(x)
        # Input adapter is rearranging in a d different order so correct it
        x = rearrange(x, 'b l x -> b x l')
        x = self.layers(x)
        return rearrange(
            x,
            'b x (c v y z) -> b c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


class DiceScore(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        """Initialization of the dice score module.

        Args:
            smooth: Factor to ensure differentiability. Defaults to 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Calculates the dice score.

        Args:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            Average Dice score, per class Dice score.
        """
        assert (
            y_pred.size() == y_true.size()
        ), f"y_pred.size(): {y_pred.size()}, y_true.size(): {y_true.size()}"

        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3, 4, 5))
        union = torch.sum(y_pred + y_true, dim=(0, 2, 3, 4, 5))

        # Calculate Dice score for each class
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return average Dice score and per class Dice score
        return dice_scores.mean(), dice_scores
