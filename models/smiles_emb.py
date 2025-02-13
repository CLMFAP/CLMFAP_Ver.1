import torch
from torch import nn
import numpy as np
import random
from models.pubchem_encoder import Encoder
import pytorch_lightning as pl
from fast_transformers.masking import LengthMask as LM
from .rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder

# from fast_trans_code.builders import TransformerEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import Favor, GeneralizedRandomFeatures
import torch.nn.functional as F
from functools import partial
from apex import optimizers
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed


class SmilesModule(pl.LightningModule):

    def __init__(self, config, vocab):
        super(SmilesModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        # location of cache File
        # Special symbols

        self.debug = config.debug
        self.text_encoder = Encoder(config.max_len)
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type="linear",
            # attention_type='full',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation="gelu",
        )
        
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        # if we are starting from scratch set seeds
        if config.restart_path == "":
            seed.seed_everything(config.seed)

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)

        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def forward(self, x):
        token_embeddings = self.tok_emb(x)
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        logits = self.lang_model(x)
        return logits

    # def on_save_checkpoint(self, checkpoint):
    #     # save RNG states each time the model and states are saved
    #     out_dict = dict()
    #     out_dict["torch_state"] = torch.get_rng_state()
    #     out_dict["cuda_state"] = torch.cuda.get_rng_state()
    #     if np:
    #         out_dict["numpy_state"] = np.random.get_state()
    #     if random:
    #         out_dict["python_state"] = random.getstate()
    #     checkpoint["rng"] = out_dict
