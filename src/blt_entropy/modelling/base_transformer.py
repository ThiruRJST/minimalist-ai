import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blt_entropy.configs import BLTEntropyModelConfig
from typing import Optional, Union


def precompute_freq_cis(dim, end, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size() , 2, 2)

def reshape_freq_tensor(freq_cis, x, seq_dim):
    ndim=x.ndim
    shape = [
        d if i ==  seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ]+[2, 2]
    return freq_cis.view(*shape)

def apply_rotary_embedding(
  xq: torch.Tensor,
  xk: torch.Tensor,
  seq_dim: int,
  freq_cis: torch.Tensor  
):
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freq_cis = reshape_freq_tensor(
        freq_cis, xq_, seq_dim
    ).float()
    xq_out = (xq_ * freq_cis).sum(5).flatten(3)
    xk_out = (xk_ * freq_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    def __init__(self, theta, head_dim, max_seqlen):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        
        self.register_buffer(
            "freq_cis",
            precompute_freq_cis(
                dim=self.head_dim,
                end=self.max_seqlen,
                theta=self.theta
            ),
            persistent=False
        )

    def reset_parameters(self):
        self.freq_cis[...] = precompute_freq_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta
        )
    
    def forward(self, seqlen: Optional[int] = None, token_id: Optional[torch.Tensor] = None):
        
        check = seqlen is None or token_id is None
        assert check, "Either seqlen or token_id must be provided."
        if token_id is not None:
            return self.freq_cis[token_id]
        elif seqlen is not None:
            return self.freq_cis[:seqlen]

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _normalize(self, x:torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self._normalize(x.float())
        return (output * self.weight.float()).type_as(x)
    
    def reset_paramerters(self):
        torch.nn.init.ones_(self.weight)
        

class Attention(nn.Module):
    def __init__(self, 
                 dim:int,
                 head_dim:int,
                 n_heads:int,
                 n_kv_heads:int,
                 rope_theta:float):
        super().__init__()
        
        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor):
        batch_size, seqlen, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))
        
        output_shape = xq.shape
        
        xq = xq.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.n_heads, self.head_dim)
        
        print(xq.shape, xk.shape, xv.shape)
        
        #applying rotary embedding to query and key tensors
        xq, xk = apply_rotary_embedding(xq, xk, 1, freq_cis[0:seqlen])
        
        #applying scaled dot product attention
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        output = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
        )
        
        output = output.transpose(1, 2).contiguous()
        output = self.wo(output.reshape(output_shape))
        
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier:int, mp_size:int=1):
        super().__init__()
        
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim =  multiple_of * ((hidden_dim +  multiple_of -1) // multiple_of)
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )
    
    def forward(self, x: torch.Tensor):
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output
    
    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std
            )
            
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std
        )
        
        
class TransformerBlock(nn.Module):
    def __init__(self, model_args: BLTEntropyModelConfig):
        super().__init__()
        self.model_args = model_args
        
        #Attention Block
        self.attention = Attention(
            dim=self.model_args.dim,
            head_dim=self.model_args.dim // self.model_args.n_heads,
            n_heads=self.model_args.n_heads,
            n_kv_heads=self.model_args.n_heads,
            rope_theta=self.model_args.rope_theta
        )
        
        #Feed-forward Block
        self.feed_forward = FeedForward(
            dim=self.model_args.dim,
            hidden_dim=4*self.model_args.dim,
            multiple_of=self.model_args.multiple_of,
            ffn_dim_multiplier=self.model_args.ffn_dim_multiplier
            
            
        )
        
        self.attention_norm = RMSNorm(self.model_args.dim, eps=self.model_args.eps)
        self.ffn_norm = RMSNorm(self.model_args.dim, eps=self.model_args.eps)
        
    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, tok_idx: Optional[torch.Tensor]=None):
        attn_out = self.attention(
            self.attention_norm(x),
            freq_cis=freq_cis)
        h = x + attn_out
        h_norm = self.ffn_norm(h)
        out = h + self.feed_forward(h_norm)
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters()
        self.attention_norm.reset_paramerters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_paramerters()
    

        
class BaseTransformer(nn.Module):
    def __init__(self, model_args: BLTEntropyModelConfig):
        super().__init__()
        
        self.model_args = model_args
        self.rope_embedding = RotaryEmbedding(
            theta = self.model_args.rope_theta,
            head_dim = self.model_args.dim // self.model_args.n_heads,
            max_seqlen=self.model_args.max_seqlen
        )

        self.layers = nn.ModuleList()
        for _ in range(self.model_args.n_layers):
            self.layers.append(TransformerBlock(self.model_args))
        
    def forward(self, h, tok_idx: Optional[torch.Tensor]=None):
        
        freq_cis = self.rope_embedding(seqlen=self.model_args.max_seqlen, token_id=tok_idx)
        
        for i, layer in enumerate(self.layers):
            h = layer(
                h, freq_cis, tok_idx
            )
        return h
    
    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embedding.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                "current_depth": (2 * (depth + 1)) ** 0.5,
                "global_depth": (2 * (len(self.layers) + 1)) ** 0.5,
                "dim_ratio": self.model_args.dim / 4096,
                "disabled": 1.0,
            }[self.model_args.init_std_factor]

            layer.init_weights(self.model_args.init_base_std, factor)
