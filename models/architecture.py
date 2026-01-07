import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self, in_channels=1, output_channels=256):
        super(CNNBackbone, self).__init__()
        # A simple convolutional block structure
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.features(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, feature_map_size):
        super().__init__()
        # Convolutional layer to create patches and project them to embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Calculate the number of patches
        num_patches = (feature_map_size // patch_size) * (feature_map_size // patch_size)

        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # +1 for CLS token

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        x = self.patch_embeddings(x)  # (batch_size, embed_dim, num_patches_H, num_patches_W)
        x = x.flatten(2)              # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (batch_size, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4., dropout=0.):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 patch_size=2,
                 num_heads=12,
                 num_blocks=6,
                 mlp_ratio=4.,
                 dropout=0.,
                 in_channels_cnn_output=256,
                 cnn_feature_map_size=14):
        super().__init__()

        self.patch_embedding = PatchEmbedding(in_channels=in_channels_cnn_output,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            feature_map_size=cnn_feature_map_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x is assumed to be the feature map from CNN: (B, C_out_cnn, H_out_cnn, W_out_cnn)
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.patch_embedding.position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return the CLS token for classification
        return x[:, 0]  # (B, embed_dim)


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, cnn_output_channels=256, cnn_feature_map_size=14,
                 vit_embed_dim=768, vit_patch_size=2, vit_num_heads=12, vit_num_blocks=6):
        super().__init__()
        self.cnn_backbone = CNNBackbone(in_channels=1, output_channels=cnn_output_channels)
        self.vision_transformer = VisionTransformer(
            embed_dim=vit_embed_dim,
            patch_size=vit_patch_size,
            num_heads=vit_num_heads,
            num_blocks=vit_num_blocks,
            in_channels_cnn_output=cnn_output_channels,
            cnn_feature_map_size=cnn_feature_map_size
        )
        self.classifier = nn.Linear(vit_embed_dim, num_classes)

    def forward(self, x):
        # 1. Pass through CNN backbone
        cnn_features = self.cnn_backbone(x)  # (B, cnn_output_channels, H_out, W_out)

        # 2. Pass through Vision Transformer
        vit_output = self.vision_transformer(cnn_features)  # (B, vit_embed_dim)

        # 3. Classification head
        logits = self.classifier(vit_output)
        return logits


def load_model(model_path, num_classes=3, device='cpu'):
    """
    Load the trained model from checkpoint
    """
    model = HybridCNNViT(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model