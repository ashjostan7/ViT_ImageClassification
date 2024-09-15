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
                mlp_dropout:float = 0.1):
        
        super().__init__()

        self.mha = MultiheadSeflAttentionBlock(embedding_dim = embedding_dim,
                                                num_heads = number_of_heads,
                                                attn_dropout = attn_dropout)
        
        self.mlp = MutliLayerPerceptron(embedding_dim = embedding_dim,
                                        mlp_size = mlp_size,
                                        dropout = mlp_dropout)
        

    def forward(self, x):

        x = self.mha(x) + x # + x is the residual connection. 
        x = self.mlp(x) + x
    
        return x 

class ViT(nn.Module):
    '''
    Entire ViT architecture: Patch & position embedding + Encoder + classification head
    '''

    # initialise class with all necessary paramters to use the nn blocks created in above classes. 
    # intitialise paramters for classification head. 
    def __init__(self, 
                img_size: int = 224,
                in_channels: int = 3,
                patch_size: int = 16,
                num_transformer_layers: int = 12, # Number of layers in ViT base
                embedding_dim: int = 768, 
                mlp_size: int = 3072,
                num_heads: int = 12,
                attn_dropout: float = 0,
                mlp_dropout: float = 0.1,
                embedding_dropout: float = 0.1,
                num_classes: int = 1000):

        super().__init__()

        # check if image size is divisible by patch size. 
        assert img_size % patch_size == 0, f"Image Size must be divisble by patch size; Image size: {img_size}; Patch size: {patch_size}"

        self.num_patches = (img_size * img_size) // patch_size ** 2

        self.class_embedding = nn.Parameter(data = torch.ones(1,1,embedding_dim), requires_grad = True) # torch.ones shape ---> (batch, patches, embedding_dim)

        # -------------------- * Why we use torch.randn() for the positional encoding vs absolute values. * ---------------------------
        # Some transformers use fixed sinusoidal positional encodings (e.g., in the original transformer paper), 
        # while others use learned positional embeddings. Random initialization (torch.randn) is usually a part of the 
        # earned positional encoding approach, where the model itself figures out the best way to represent positions during training. 
        # This has shown better results in some cases, especially in vision tasks.
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad = True)

        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                patch_size=patch_size,
                                                embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim = embedding_dim,
                                                                            number_of_heads = num_heads,
                                                                            attn_dropout = attn_dropout, 
                                                                            mlp_size = mlp_size,
                                                                            mlp_dropout = mlp_dropout) for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = embedding_dim), 
                                        nn.Linear(in_features = embedding_dim, out_features = num_classes))
        
    def forward(self, x):

        batch_size = x.shape[0]
        
         # expand the single learnable class token across the batch dimension, "-1" means to "infer the dimension"
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim = 1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0]) # To the classifier head we only pass the 0th patch of all the elements of the batch.
    
        return x
        

