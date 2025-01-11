import torch
from typing import Tuple
from torch import nn, Tensor


class RotaryEmbeddingMegatron(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.1
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_base: int = 10000,
    ) -> None:
        super().__init__()

        dim = kv_channels
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
    
    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings"""
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        freqs = torch.outer(seq, self.inv_freq)  # [seq len, dim]

        return freqs

    def get_cos_sin(self, max_seq_len: int, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of RoPE embedding.
        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        return emb
    
    @staticmethod
    def apply_rotary_pos_emb_megatron(t: Tensor, freqs: Tensor, mscale: float = 1.0):
        """Apply rotary positional embedding to input tensor T.

        check https://kexue.fm/archives/8265 for detailed formulas

        Args:
            t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
            freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

        Returns:
            Tensor: The input tensor after applying RoPE
        """
        def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
            """Change sign so the last dimension becomes [-odd, +even]

            Args:
                x (Tensor): Input tensor

            Returns:
                Tensor: Tensor rotated half
            """
            if not rotary_interleaved:
                x1, x2 = torch.chunk(x, 2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            else:
                x1 = x[:, :, :, ::2]
                x2 = x[:, :, :, 1::2]
                x_new = torch.stack((-x2, x1), dim=-1)
                return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)
        rot_dim = freqs.shape[-1]
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
        t = (t * cos_) + (_rotate_half(t, False) * sin_)
        return torch.cat((t, t_pass), dim=-1)
    
    @staticmethod
    def apply_rotary_pos_emb_apex(t: Tensor, freqs: Tensor, mscale: float = 1.0):
        try:
            from apex.transformer.functional import fused_apply_rotary_pos_emb
        except:
            print("failed to import fused_apply_rotary_pos_emb from apex.transformer.functional")
            return t
        return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
    
    @staticmethod
    def apply_rotary_pos_emb_transformer_engine(t: Tensor, freqs: Tensor, mscale: float = 1.0):
        try:
            from transformer_engine.pytorch.attention import FusedRoPEFunc
        except:
            print("failed to import FusedRoPEFunc from transformer_engine")
            return t
        return FusedRoPEFunc.apply(t, freqs, "sbhd")

    @staticmethod
    def apply_rotary_pos_emb_flash(t: Tensor, cos: Tensor, sin: Tensor):
        try:
            from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
        except:
            print("failed to import apply_rotary_emb_flash from flash_attn")
            return t
        t = t.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t, cos, sin, False)
        y = y.permute(1, 0, 2, 3) 
        return y