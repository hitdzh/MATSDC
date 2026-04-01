import torch
import torch.nn as nn
from ..layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from ..layers.SelfAttention_Family import ReformerLayer, AttentionLayer
from ..layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity via LSHSelfAttention
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ReformerLayer(
                            None, configs.d_model, configs.n_heads,
                            causal=False, bucket_size=configs.bucket_size,
                            n_hashes=configs.n_hashes
                        ),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ReformerLayer(
                            None, configs.d_model, configs.n_heads,
                            causal=True, bucket_size=configs.bucket_size,
                            n_hashes=configs.n_hashes
                        ),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ReformerLayer(
                            None, configs.d_model, configs.n_heads,
                            causal=False, bucket_size=configs.bucket_size,
                            n_hashes=configs.n_hashes
                        ),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
