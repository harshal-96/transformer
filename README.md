# Transformer
Transformer explanation and code :

Transformers are the backbone of many state-of-the-art models in natural language processing (NLP) and other fields. Here’s a detailed explanation of how Transformers operate, along with code for the encoder and decoder parts.

### How Transformers Operate

Transformers use a mechanism called self-attention to process input data in parallel, making them more efficient than sequential models like RNNs. Here’s a high-level overview of the Transformer architecture:

1. **Input Embedding**: Convert the input tokens into dense vectors (embeddings).
2. **Positional Encoding**: Add positional information to the embeddings to retain the order of the sequence.
3. **Encoder**: A stack of identical layers that process the input sequence and produce context-aware representations.
4. **Decoder**: A stack of identical layers that generate the output sequence, using the encoder's output and previous decoder outputs.

### Transformer Encoder and Decoder Code

Below is a simplified implementation of the Transformer encoder and decoder using PyTorch.

#### Imports and Basic Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

#### Multi-Head Attention Mechanism

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(output)
```

#### Feed Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

#### Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        return self.norm2(x)
```

#### Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        attn_output = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        return self.norm3(x)
```

#### Transformer Model

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder_embedding(src) * math.sqrt(d_model)
        src = self.positional_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(d_model)
        tgt = self.positional_encoding(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)
        
        return self.linear(tgt)
```

### Explanation of the Code

1. **Positional Encoding**: Adds positional information to the input embeddings because transformers don’t have built-in recurrence or convolution to handle sequence order.
2. **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence simultaneously. It computes attention scores, applies a softmax to get attention weights, and then combines the input vectors weighted by these attention weights.
3. **Feed Forward Network**: A simple fully connected feed-forward network applied to each position separately and identically.
4. **Encoder Layer**: Consists of a multi-head self-attention mechanism followed by a feed-forward network. It also includes layer normalization and dropout for regularization.
5. **Decoder Layer**: Similar to the encoder layer but includes an additional cross-attention mechanism that attends to the encoder's output.
6. **Transformer Model**: Integrates the encoder and decoder layers into a full transformer model. It includes embeddings for the input and output sequences and applies the positional encoding. The final output is projected back to the vocabulary size to produce predictions.

### Running the Transformer Model

To run this transformer model, you would need to define your source and target vocabularies, create masks if necessary, and pass your input sequences through the model. Here’s a simple example of how to use the Transformer class:

```python
# Example usage of the Transformer model
src_vocab_size = 10000  # Example vocabulary size for source language
tgt_vocab_size = 10000  # Example vocabulary size for target language
d_model = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
d_ff = 2048
dropout = 0.1

# Instantiate the model
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout).to(device)

# Example input tensors
src

 = torch.randint(0, src_vocab_size, (32, 10)).to(device)  # Batch of 32 sentences, each of length 10
tgt = torch.randint(0, tgt_vocab_size, (32, 10)).to(device)  # Batch of 32 sentences, each of length 10

# Forward pass
output = model(src, tgt)
print(output.shape)  # Output shape should be (batch_size, sequence_length, tgt_vocab_size)
```

This code provides a basic but functional implementation of a transformer model, following the principles outlined in the original Transformer paper "Attention is All You Need" by Vaswani et al. (2017).
