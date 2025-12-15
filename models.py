
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch import Tensor


# Deep Autoencoder

class Autoencoder(nn.Module):

    def __init__(self, layerNums):
        super().__init__()

        enc_layers = []
        dec_layers = []

        for i in range(len(layerNums) - 1):
            enc_layers.append(nn.Linear(layerNums[i], layerNums[i + 1]))
            enc_layers.append(nn.GELU())
        s
        for i in reversed(range(1,len(layerNums))):
            dec_layers.append(nn.Linear(layerNums[i], layerNums[i - 1]))
            dec_layers.append(nn.GELU())
        
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

# Deep convolutional autoencoder

class ConvolutionalAE(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv_down1 =   nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size = 3, stride = 1, padding = 'same'),
                            nn.GELU(),
                            nn.Conv1d(8, 16, kernel_size = 3, stride = 1, padding = 'same'),
                            nn.GELU())

        self.conv_down2 =   nn.Sequential(
                            nn.Conv1d(16, 16, kernel_size = 3, stride = 1, padding = 'same'),
                            nn.GELU(),
                            nn.Conv1d(16, 32, kernel_size = 3, stride = 1, padding = 'same'),
                            nn.GELU(),
                            nn.Flatten())
        
        self.max_pool = nn.MaxPool1d(kernel_size = 2, stride = 2, return_indices=True)

        self.fnn_down = nn.Sequential(nn.Linear(32 * 26, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 16), 
                                    nn.ReLU())
        
        self.fnn_up = nn.Sequential(nn.Linear(16, 128),
                                nn.ReLU(),
                                nn.Linear(128, 32*26),
                                nn.ReLU(),
                                nn.Unflatten(1,(32,26)))
        
        self.conv_up1 = nn.Sequential(nn.ConvTranspose1d(32, 16, kernel_size = 3, stride = 1, padding = 1),
                            nn.GELU(),
                            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=1, padding = 1),
                            nn.GELU())
        
        self.max_unpool = nn.MaxUnpool1d(kernel_size=2, stride = 2)
                
        self.conv_up2 = nn.Sequential(
                            nn.Conv1d(16, 8, kernel_size = 3, stride = 1, padding = 1),
                            nn.GELU(),
                            nn.Conv1d(8, 1, kernel_size = 3, stride = 1, padding = 1))
        
        
    def forward(self, x):
        
        # x = self.conv_down1(x)
        # x, inds = self.max_pool(x)
        # x = self.conv_down2(x)
        ###################################
        # print("Input shape:", x.shape)
        x = self.conv_down1(x)
        #print("After conv1:", x.shape)
        x, inds = self.max_pool(x)
        #print("After maxpool:", x.shape)
        x = self.conv_down2(x)
        #print("After conv2:", x.shape)
        x = x.view(x.size(0), -1)
        #print("After flatten:", x.shape)  # ← this should match your Linear layer input
###################################################################
        x = self.fnn_down(x)
        
        x = self.fnn_up(x)
        x = self.conv_up1(x)
        x = self.max_unpool(x, inds)
        x = self.conv_up2(x)
        
        return x



# Transformer architecture using nn.Module

class PositionalEncodingFromTorch(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, 
                n_features = 1, 
                d_model = 512, 
                d_ff = 2048, 
                num_heads = 8, 
                n_layers = 4,
                dropout = 0.1, 
                batch_first = True):
        
        super().__init__()
        
        self.d_model = d_model
        self.encoding_embedding = nn.Linear(n_features, self.d_model)
        self.decoding_emebedding = nn.Linear(self.d_model, n_features)
        
        self.encoder_input_layer = nn.TransformerEncoderLayer(self.d_model, num_heads, d_ff, dropout, batch_first = batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_input_layer, n_layers)
        
        # self.decoder_input_layer = nn.TransformerDecoderLayer(self.d_model, num_heads, d_ff, dropout, batch_first = batch_first)
        # self.decoder = nn.TransformerDecoder(self.decoder_input_layer, n_layers)
    
        self.pos_enc = PositionalEncodingFromTorch(self.d_model, 0.1)
    
    def forward(self, x):
        
        x_out = self.encoding_embedding(x)
        x_out = self.pos_enc(x_out)
        x_out = self.encoder(x_out)
        output = self.decoding_emebedding(x_out)
        
        return output
        


# Transformer architecture

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Feed forward network folloing the multi-head attention
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

# Positional endcoding module to inject location of elements in sequence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Complete transformer module
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Linear(src_vocab_size, d_model)
        self.decoder_embedding = nn.Linear(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device = 'cuda'), diagonal=1)).bool()
        cuda0 = torch.device('cuda:0')
        nopeak_mask.to(cuda0)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    



                             

                             

        


