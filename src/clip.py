import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import GPT2Config, GPT2Model


class ImageEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        num_channels: int = 3,
    ):
        super(ImageEmbeddings, self).__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Channels, Height, Width) -> (Batch size, Embed dim, Height, Width)
        x = self.patch_embedding(x)
        # x: (Batch size, Embed dim, Height, Width) -> (Batch size, Height * Width, Embed dim)
        x = x.flatten(2).transpose(1, 2)
        # Add position embeddings
        x = x + self.position_embedding(self.position_ids)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop_rate: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Num patches, Embed dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ImageEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super(ImageEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Num patches, Embed dim)
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x)
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ):
        super(Attention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Num patches, Embed dim)
        batch_size, n_patches, d_model = x.shape
        q = (
            self.wq(x)
            .reshape(batch_size, n_patches, self.num_heads, d_model // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.wk(x)
            .reshape(batch_size, n_patches, self.num_heads, d_model // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.wv(x)
            .reshape(batch_size, n_patches, self.num_heads, d_model // self.num_heads)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_patches, d_model)
        x = self.wo(x)
        x = self.proj_drop(x)

        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super(ImageEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                ImageEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Num patches, Embed dim)
        for layer in self.layers:
            x = layer(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        img_embed_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        max_seq_length: int = 16,
    ):
        super(CLIPModel, self).__init__()

        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.image_embeddings = ImageEmbeddings(
            embed_dim=img_embed_dim,
            patch_size=patch_size,
            image_size=image_size,
        )

        self.image_encoder = ImageEncoder(
            num_layers=num_layers,
            embed_dim=img_embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.image_ln_final = nn.LayerNorm(img_embed_dim)

        configuration = GPT2Config(
            vocab_size=50257,
            n_positions=max_seq_length,
            n_embd=embed_dim,
            n_layer=num_layers,
            n_head=num_heads,
        )
        self.text_encoder = GPT2Model(configuration)
        self.text_ln_final = nn.LayerNorm(embed_dim)

        self.image_projection = nn.Linear(img_embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

    def encode_text(
        self, text: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Text encoding
        x = self.text_encoder(
            input_ids=text, attention_mask=attention_mask, return_dict=True
        )

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x.last_hidden_state[
            torch.arange(x.last_hidden_state.shape[0]), text.argmax(dim=-1)
        ]

        x = self.text_ln_final(x)
        x = self.text_projection(x)
        return x

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: (Batch size, Channels, Height, Width)
        # text: (Batch size, Max sequence length)
        image = self.image_embeddings(image)
        image = self.image_encoder(image)
        image = self.image_ln_final(image)

        image_pooled = image[:, -1, :]
        image_pooled = self.image_projection(image_pooled)

        text = self.encode_text(text, attention_mask=attention_mask)
        return image_pooled, text

    def clip_loss(self, image_embeddings, text_embeddings):
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute logits by multiplying image and text embeddings (dot product)
        logits_per_image = image_embeddings @ text_embeddings.T
        logits_per_text = text_embeddings @ image_embeddings.T

        logits_per_image = logits_per_image / self.temperature
        logits_per_text = logits_per_text / self.temperature

        # Create targets (diagonal is positive pairs)
        num_samples = image_embeddings.shape[0]
        labels = torch.arange(num_samples, device=image_embeddings.device)

        # Compute cross-entropy loss for image-to-text and text-to-image directions
        loss_image_to_text = F.cross_entropy(logits_per_image, labels)
        loss_text_to_image = F.cross_entropy(logits_per_text, labels)

        # Final loss is the average of both directions
        loss = (loss_image_to_text + loss_text_to_image) / 2.0
        return loss
