from torch import Tensor
from deepxml.models.network import SiameseNetworkIS, NetworkIS, _to_device
from .module_factory import ModuleFactory


class SiameseNetworkIS(SiameseNetworkIS):
    def _module_factory(self) -> dict:
        return ModuleFactory

    def _encode_lbl(self, x: tuple) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        return self.encoder_lbl(_to_device(x, self.device))

    def encode_lbl(self, x: tuple, ind: Tensor) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        _x = self._encode_lbl(x)
        return self.transform_lbl((_x, ind)), _x

    def forward(self, batch, *args):
        """Forward pass

        Args:
            batch (dict): A dictionary containing features or 
                tokenized representation and shared label shortlist

        Returns:
            torch.Tensor: output of the network (typically logits)
        """
        X = self.encode(_to_device(batch['X'], self.device))
        Z, _Z = self.encode_lbl(
            _to_device(batch['Z'], self.device), 
            _to_device(batch['Y_s'], self.device))
        return (self.similarity(X, Z), self.similarity(X, _Z)), X


class XCNetworkIS(NetworkIS):
    """
    Class to train extreme classifiers with shared shortlist
    """
    def _module_factory(self) -> dict:
        return ModuleFactory

    def _encode_lbl(self, x: tuple) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        return self.encoder_lbl(_to_device(x, self.device))

    def encode_lbl(self, x: tuple, ind: Tensor) -> Tensor:
        """Encode an item using the given network

        Args:
            x (tuple): #TODO

        Returns:
            torch.Tensor: Encoded item
        """
        _x = self._encode_lbl(x)
        return self.transform_lbl((_x, ind)), _x
