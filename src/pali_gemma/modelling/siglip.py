import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.siglip_config import SigLipVisionConfig

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            padding="valid")
        
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values:torch.Tensor):
        _, _, hieght, width = pixel_values.shape
        # convolving the image with the patch_embedding layer which has kernel and stride equal to patch_size
        # [batch_size, num_channels, height, width] => [batch_size, embedding_dim, H_num_patches, W_num_patches]
        embedding_output = self.patch_embeddings(pixel_values)
        #[batch_size, embedding_dim, H_num_patches, W_num_patches] => [batch_size, embedding_dim, num_patches]
        embedding_output = embedding_output.flatten(2)
        #[batch_size, embedding_dim, num_patches] => [batch_size, num_patches, embedding_dim]
        embedding_output = embedding_output.transpose(1, 2)
        #adding positional embeddings
        embedding_output = embedding_output + self.position_embeddings(self.position_ids)
        
        return embedding_output

class SiglipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.fc2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)
    
    def forward(self, hidden_states):
        #Fc1: [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        #GELU: [batch_size, num_patches, intermdiate_size]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        #Fc2: [batch_size, num_patches, embedding_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
        
class SiglipAttention(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.scale = self.head_dim ** -0.5 #1/sqrt(head_dim)
        self.dropout = self.config.attention_dropout
        
        self.k_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size) 
        
    def forward(self, hidden_states):
        #hidden_states: [batch_size, num_patches, embedding_dim]
        batch_size, seq_len, _ = hidden_states.size()
        #query_states: [batch_size, num_patches, embedding_dim]
        query_states = self.q_proj(hidden_states)
        #key_states: [batch_size, num_patches, embedding_dim]
        key_states = self.k_proj(hidden_states)
        #value_states: [batch_size, num_patches, embedding_dim]
        value_states = self.v_proj(hidden_states)
        
        #reshaping: [batch_size, num_patches, embedding_dim] => [batch_size, num_patches, num_heads, head_dim] => [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        
        #attention_scores: [batch_size, num_heads, num_patches, head_dim] * [batch-size, num_heads, head_dim, num_patches] => [batch-size, num_heads, num_patches, num_patches]
        attention_scores = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        
        if attention_scores.size() != (batch_size, self.config.num_attention_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention scores must be of size: {(batch_size, self.config.num_attention_heads, self.config.num_attention_heads, seq_len)}, but is" 
                f"{attention_scores.size()}")
        
        #Apply the softmax row wise
        attention_scores = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #Apply dropout
        attention_scores = F.dropout(attention_scores, p=self.dropout, training=self.training)
        #Multiply attention weights by value states: [batch_size, num_heads, num_patches, num_patches] * [batch_size, num_heads, num_patches, head_dim] => [pbatch_size, num_heads, num_patches, num_patches]
        attn_output = torch.matmul(attention_scores, value_states)
        
        if attn_output.size() != (batch_size, self.config.num_attention_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention scores must be of size: {(batch_size, self.config.num_attention_heads, seq_len, self.head_dim)}, but is" 
                f"{attn_output.size()}")
        
        #reshaping: [batch_size, num_heads, num_patches, head_dim] => [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        #reshaping: [batch_size, num_patches, num_heads, head_dim] => [batch_size, num_patches, embedding_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.hidden_size)
        #projection: [batch_size, num_patches, embedding_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output, attention_scores
        
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size, eps = self.config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        
    def forward(self, hidden_states):
        #residual connection: [batch_size, num_patches, embedding_dim]
        residual = hidden_states
        #LayerNorm: [batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm1(hidden_states)
        #Self attention: [batch_size, num_patches, embedding_dim]
        hidden_states, _ = self.self_attn(hidden_states)
        #residual connection: [batch_size, num_patches, embedding_dim]
        hidden_states = residual + hidden_states
        
        #residual connection store: [batch_size, num_patches, embedding_dim]
        residual = hidden_states
        #LayerNorm: [batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm2(hidden_states)
        #MLP: [batch_size, num_patches, embedding_dim]
        hidden_states = self.mlp(hidden_states)
        
        return residual + hidden_states
        
        
        
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(self.config.num_hidden_layers)]
        )
        
    def forward(self, input_embeds):
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            # [batch_size, num_patches, embedding_dim]
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size
        
        self.patch_embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=self.config.layer_norm_eps)
    
    def forward(self, pixel_values):
        # [batch_size, num_channels, height, width] => [batch_size, num_patches, embedding_dim]
        embedding_output = self.patch_embeddings(pixel_values)
        
        encoder_output = self.encoder(input_embeds = embedding_output)
        vision_output = self.post_layernorm(encoder_output)
        return vision_output

class SigLipVisionModel(nn.Module):
    
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
    
    def forward(self, pixel_values):
        #[batch_size, num_channels, height, width] => [batch_size, num_patches, embedding_dim]
        vision_output = self.vision_model(pixel_values)
        return vision_output

