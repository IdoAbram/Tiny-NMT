class DecoderConfig:
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
