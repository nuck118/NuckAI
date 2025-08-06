import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# --- The Vision Encoder (for images) ---
class VisionEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Use a pre-trained ResNet-18 model from torchvision
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Add a new linear layer to project the ResNet output to our desired dimension
        self.projection = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.resnet(x)
        # The output of ResNet is a 4D tensor (batch, features, 1, 1). We need to flatten it.
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x

# --- The core Transformer Block ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # We no longer register a fixed mask here.
        # It will be created dynamically in the forward pass.

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # Create a dynamic attention mask based on the current sequence length (T)
        attn_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_mask = attn_mask.view(1, 1, T, T)

        # Matmul operation
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# --- The Multimodal NuckAi ---
class MultimodalNuckAi(nn.Module):
    def __init__(self, vocab_size, block_size=256, d_model=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # New: Vision Encoder
        self.vision_encoder = VisionEncoder(out_dim=d_model)

        # Old: Text Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(block_size + 1, d_model) # +1 for the image embedding
        
        # The main transformer stack
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, block_size + 1, dropout) # +1 for image
            for _ in range(num_layers)
        ])
        
        # Final layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, text_input, image_input=None, targets=None):
        B, T = text_input.size()
        
        # Get text embeddings
        text_emb = self.token_embeddings(text_input)
        
        if image_input is not None:
            # Get image embedding with batch dimension
            image_emb = self.vision_encoder(image_input).unsqueeze(1)
            
            # Combine image embedding with text embeddings
            combined_emb = torch.cat((image_emb, text_emb), dim=1)

            # Get position embeddings for the combined sequence
            pos_indices = torch.arange(0, T + 1, dtype=torch.long, device=text_input.device)
            pos_emb = self.position_embeddings(pos_indices).unsqueeze(0)
            
            # The input to the transformer is now the combination
            x = combined_emb + pos_emb
            
        else:
            # If no image, just use text and its position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=text_input.device)
            pos_emb = self.position_embeddings(pos)
            x = text_emb + pos_emb

        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Exclude the image embedding from the loss calculation
            logits = logits[:, 1:, :].contiguous()
            logits = logits.view(-1, self.lm_head.out_features)
            targets = targets.contiguous().view(-1)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss