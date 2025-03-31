from typing import Callable
from torch import Tensor

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from deepxml.libs.model import EmbeddingModelIS as _EmbeddingModelIS


class EmbeddingModelIS(_EmbeddingModelIS):
    def get_label_representations(
            self, 
            dataset, 
            batch_size=128):
        return self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_lbl,
            batch_size=batch_size,
            feature_t=dataset.label_features._type,
            num_workers=0
            )

    @torch.no_grad()
    def _embeddings(
        self,
        data_loader: DataLoader,
        encoder: Callable = None,
        fname_out: str = None,
        _dtype='float32'
    ) -> np.ndarray:
        """Encode given data points
        * support for objects or files on disk


        Args:
            data_loader (DataLoader): DataLoader object to \
                  create batches and iterate over it
            encoder (Callable, optional): Defaults to None.
                use this function to encode given dataset
                * net.encode is used when None
            fname_out (str, optional): dump features to this file. Defaults to None.
            _dtype (str, optional): data type of output tensors. Defaults to 'float32'.

        Returns:
            np.ndarray: embeddings (as memmap or ndarray)
        """

        if encoder is None:
            self.logger.info("Using the default encoder.")
            encoder = self.net.encode
        self.net.eval()
        if fname_out is not None:  # Save to disk
            embeddings = np.memmap(
                fname_out, dtype=_dtype, mode='w+',
                shape=(len(data_loader.dataset), self.net.repr_dims))
        else:  # Keep in memory
            embeddings = np.zeros((
                len(data_loader.dataset), self.net.repr_dims),
                dtype=_dtype)
        idx = 0
        for batch in tqdm(data_loader, desc="Computing Embeddings"):
            bsz = batch['batch_size']
            out = encoder(batch['X'], batch['indices'])
            if isinstance(out, tuple):
                out, _ = out
            embeddings[idx :idx+bsz, :] = out.detach().cpu().numpy()
            idx += bsz
        torch.cuda.empty_cache()
        if fname_out is not None:  # Flush all changes to disk
            embeddings.flush()
        return embeddings

    def _setup_prototype_network(self, dataset):
        raise NotImplementedError("")

    def _setup(self, dataset):
        self._init_memory_bank(dataset)
        self._setup_prototype_network(dataset)

    def _compute_loss(self, y_hat: tuple, batch: dict) -> Tensor:
        """Compute loss

        Args:
            y_hat (tuple): predictions from the network
            batch (dict): dict containing (local) ground truth

        Returns:
            Tensor: computed loss
        """
        y_hat, y_hat_i = y_hat
        y = batch['Y'].to(y_hat.device)
        mask = batch['Y_mask']
        return self.criterion(
            y_hat,
            y,
            y_hat_i,
            mask=mask.to(y_hat.device) if mask is not None else mask)
