import torch
from torch import nn

class PatchEmbedding(nn.Module):
    '''
    Class to create patch embeddings of image data.
    '''
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels = in_channels,
                                out_channels = embedding_dim,
                                kernel_size = patch_size,
                                stride = patch_size,
                                padding = 0)
        
        self.flatten = nn.Flatten(start_dim =2, 
                                    end_dim = 3)
        
    
    def forward(self, x):

        img_res = x.shape[-1]
        assert img_res % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0,2,1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


# Layer Normalisation + Multihead attention block. 
class MultiheadSeflAttentionBlock(nn.Module):

    def __init__(self,
                embedding_dim: int = 768,
                num_heads:int = 12,
                attn_dropout:float = 0):

        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

        self.mha = nn.MultiheadAttention(embed_dim = embedding_dim,
                                        num_heads = num_heads,
                                        dropout = attn_dropout,
                                        batch_first = True)
        
    
    def forward(self, x):

        x = self.layer_norm(x)

        attn_output, _ = self.mha(query = x,
                    key = x, 
                    value = x)
        

        return attn_output

class MutliLayerPerceptron(nn.Module):
    '''
    Multi Layer Perceptron Block
    '''

    def __init__(self,
                embedding_dim: int = 768,
                mlp_size:int  = 3072,
                dropout:float = 0.1):

        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
        self.mlp = nn.Sequential( 
                                nn.Linear(in_features = embedding_dim,
                                            out_features = mlp_size),
                                nn.GELU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(in_features = mlp_size, 
                                        out_features = embedding_dim),
                                nn.Dropout(p=dropout),
        )

    def forward(self, x):

        x = self.layer_norm(x)
        x = self.mlp(x)

        return x

class TransformerEncoderBlock(nn.Module):
    '''
    Enitre ViT Architecture. 
    '''

    def __init__(self,
                embedding_dim:int = 768,
                number_of_heads:int = 12,
                attn_dropout:float = 0, 
                mlp_size:int = 3072,
                dropout:float = 0.1):
        
        super().__init__()

        self.mha = MultiheadSeflAttentionBlock(embedding_dim = 768,
                                                num_heads = 12,
                                                attn_dropout = 0)
        
        self.mlp = MutliLayerPerceptron(embedding_dim = 768,
                                        mlp_size = 3072,
                                        dropout = 0.1)
        

    def forward(self, x):

        x = self.mha(x) + x # + x is the residual connection. 
        x = self.mlp(x) + x
    
        return x 