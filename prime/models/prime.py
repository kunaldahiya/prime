from numpy import ndarray
from torch import Tensor

import torch
from deepxml.models import TransformerEncoderBag, EmbeddingBank


class PRIME(torch.nn.Module):
    """
    Implements the PRIME architecture. Augment the given embeddings with:
        - Auxiliary vectors 
        - Centroids or Protiotypes
    """
    def __init__(
            self, 
            embedding_dim: int, 
            num_labels: int,
            num_aux_embeddings: int,
            num_heads: int=1,
            hidden_dim: int=1024,
            dropout: float=0.1,
            activation: str='gelu',
            norm_first: bool=False,
            device="cpu"):
        """
        Args:
            embedding_dim (int): dim of input embeddings
            num_labels (int): num of labels
            num_fv_embeddings (int): number of free vectors
            num_heads (int, optional): #heads in combiner. Defaults to 1
            hidden_dim (int, optional): Defaults to 1024.
                dim_feedforward in combiner.
            dropout (float, optional): dropout in combiner. Defaults to 0.1.
            activation (str, optional): activation func in combiner. Defaults to 'gelu'.
            norm_first (bool, optional): Defaults to False.
                normalize before applying self-attention?
            device (str, optional): Device for embeddings. Defaults to "cpu".
        """
        super(PRIME, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        #TODO: Add functionality to make it optional
        self.aux_bank = EmbeddingBank(
            embedding_dim=embedding_dim, 
            num_embeddings=num_aux_embeddings,
            num_items=num_labels,
            device=device 
        )

        #TODO: Check if it stays on CPU?
        self.prototype_bank = EmbeddingBank(
            embedding_dim=embedding_dim, 
            num_embeddings=num_labels,
            num_items=num_labels,
            requires_grad=False, # prototypes are updated explicitly
            device=device 
        )

        self.combiner = TransformerEncoderBag(
            d_model=embedding_dim, 
            n_head=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )

    def encode(self, x: Tensor, ind: Tensor) -> Tensor:
        """Enrich the given embedding with auxiliary vectors and prototypes

        Args:
            x (Tensor): Input tensor (e.g., text based label representation)
            ind (Tensor): Indices used to fetch free vectors and prototypes

        Returns:
            Tensor: Enriched Tensor
        """
        v = self.aux_bank[ind].to(self.device)
        z = self.prototype_bank[ind].to(self.device)
        return self.combiner([x, v, z])
    
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        """Enrich the given embedding with auxiliary vectors and prototypes

        Args:
            x (tuple[Tensor, Tensor]):
                - Input tensor (e.g., text based label representation)
                - Indices used to fetch free vectors and prototypes

        Returns:
            Tensor: Enriched Tensor
        """
        return self.encode(*x)

    @property
    def repr_dims(self):
        return self.embedding_dim

    def setup_aux_bank(self, X: ndarray) -> None:
        raise NotImplementedError("")

    def setup_prototype_bank(self, X: ndarray) -> None:
        raise NotImplementedError("")

    def __repr__(self):
        s = '{name}('
        s += f'(Aux Bank) {str(self.aux_bank)}\n'
        s += f'(Prototype Bank) {str(self.prototype_bank)}\n'
        s += f'(Combiner) {str(self.combiner)}\n'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
