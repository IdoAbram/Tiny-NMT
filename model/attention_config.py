class AttentionConfig:
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = attn_dim
