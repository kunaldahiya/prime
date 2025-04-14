from typing import Callable
from torch import Tensor
from numpy import ndarray
from torch.utils.data import Dataset

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from xclib.utils.dense import compute_centroid
from xclib.utils.dense import _normalize as normalize 
from .dataset_factory import DatasetFactory
from deepxml.libs.pipeline import XCPipelineIS as _XCPipelineIS
from deepxml.libs.pipeline import EmbeddingPipelineIS as _EmbeddingPipelineIS


class EmbeddingPipelineIS(_EmbeddingPipelineIS):
    def _dataset_factory(self):
        """This function allows the child method to inherit the class
        to define its own datasets. They can just redefine the class 
        to load from their local code. Otherwise more code change is required

        Returns:
            dict: A dataset factory that can return the Dataset class based 
            on the key (sampling_t)
        """
        return DatasetFactory 

    def get_label_representations(
            self, 
            dataset: torch.utils.data.Dataset, 
            batch_size: int=128) -> ndarray:
        return self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_lbl,
            batch_size=batch_size,
            feature_t=dataset.label_features._type,
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

    def _setup_prototype_network(self, dataset, batch_size, num_workers=6):
        self.logger.info("Setting up prototype network!")
        lbl_emb = self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode, # only text based embeddings
            feature_t=dataset.label_features._type,
            num_workers=num_workers
            )
        self.net.transform_lbl.setup_aux_bank(normalize(lbl_emb))

        doc_emb = self.get_embeddings(
            data=dataset.features.data,
            encoder=self.net.encode, # only text based embeddings
            feature_t=dataset.features._type,
            num_workers=num_workers
            )
        lbl_emb = compute_centroid(doc_emb, dataset.labels.data, reduction='mean')
        self.net.transform_lbl.setup_prototype_bank(normalize(lbl_emb))

    def _setup(self, dataset, batch_size, num_workers=6):
        self._init_memory_bank(dataset)
        self._setup_prototype_network(dataset, batch_size, num_workers)

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


class XCPipelineIS(_XCPipelineIS):
    """
    For models that do XC training with implicit sampling

    * XC training: classifiers and encoders (optionally) are trained    
    * Implicit sampling: negatives are not explicity sampled but
    selected from positive labels of other documents in the mini-batch

    """    
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

    def _init_classifier(
            self, 
            dataset: Dataset, 
            batch_size: int=128,
            num_workers: int=6) -> None:
        self.logger.info("Initializing the classifier!")
        lbl_emb = self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_lbl,
            batch_size=batch_size,
            feature_t=dataset.label_features._type,
            num_workers=num_workers
            )
        self.net.classifier.initialize(
            torch.from_numpy(normalize(lbl_emb)))
