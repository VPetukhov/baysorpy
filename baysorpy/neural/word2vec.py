import numpy as np
from scipy import sparse
import pandas as pd
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class GeneAdjDataModule(pl.LightningDataModule):
    def __init__(
            self, adj_mat: sparse._csr.csr_matrix, gene_ids: np.ndarray, n_neg_samples: int = 5,
            batch_size: int = 32, shuffle: bool = True
        ):
        super().__init__()

        self.adj_mat = adj_mat
        self.gene_ids = np.array(gene_ids)
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

        nzi,nzj = self.adj_mat.nonzero();
        adj_list = pd.Series(self.gene_ids[nzj]).groupby(nzi).apply(list)
        self.adj_list_per_gene = adj_list.groupby(self.gene_ids).apply(lambda x: np.concatenate(list(x)))

    def _generate_neg_samples(self):
        n_genes = self.n_genes()
        neg_list = []
        for targ,cont in self.adj_list_per_gene.items():
            cur_sp = self.sample_probs * (1 - self.gene_adj_probs[:, targ].A[:,0]) ** 4
            neg_list.append(list(np.random.choice(
                np.arange(n_genes),
                size=(len(cont), self.n_neg_samples),
                p=(cur_sp / cur_sp.sum())
            )))

        return neg_list

    def n_genes(self):
        return int(np.max(self.gene_ids) + 1)

    def prepare_data(self):
        n_genes = self.n_genes()
        self.gene_adj_probs = sparse.vstack([
            sparse.csc_array(pd.value_counts(aids, normalize=True).reindex(range(n_genes)).fillna(0))
            for aids in self.adj_list_per_gene.values
        ]).T

        self.sample_probs = pd.value_counts(self.gene_ids, normalize=True).reindex(range(n_genes)).fillna(0).values

    # TODO: add validation split

    def train_dataloader(self):
        train_data = pd.DataFrame({
            "target": self.adj_list_per_gene.index,
            "context": self.adj_list_per_gene.values,
            "negative": self._generate_neg_samples()
        }).explode(["context", "negative"])
        # dataset = PandasDataset(train_data, train_data.columns)
        # return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        # dataset = torch.utils.data.TensorDataset(
        #     torch.tensor(train_data.target.values),
        #     torch.tensor(train_data.context.values.astype(int)),
        #     torch.tensor(np.concatenate([np.atleast_2d(x) for x in train_data.negative.values]))
        # )

        if self.shuffle:
            train_data = train_data.iloc[np.random.choice(np.arange(train_data.shape[0]), train_data.shape[0], replace=False),:]

        return zip( # This is way faster than using a DataLoader; list is for tqdm
            np.array_split(torch.tensor(train_data.target.values), train_data.shape[0] // self.batch_size),
            np.array_split(torch.tensor(train_data.context.values.astype(int)), train_data.shape[0] // self.batch_size),
            np.array_split(torch.tensor(np.concatenate([np.atleast_2d(x) for x in train_data.negative.values])), train_data.shape[0] // self.batch_size)
        )


class Word2Vec(pl.LightningModule):
    def __init__(self, embedding_size: int, vocab_size: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

        self.example_input_array = [
            torch.randint(vocab_size, (1,)),
            torch.randint(vocab_size, (1,)),
            torch.randint(vocab_size, (1, 5))
        ]


    def forward(self, target_word, context_word, negative_example):
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)

        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)

        out = torch.sum(F.logsigmoid(emb_product))

        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))

        return -out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss

    def get_embeddings(self, numpy: bool = True, gene_names: Optional[List[str]] = None):
        embs = self.embeddings_target(torch.arange(self.hparams.vocab_size)).detach()
        if numpy:
            embs = embs.numpy()

        if gene_names is not None:
            embs = pd.DataFrame(embs, index=gene_names)

        return embs
