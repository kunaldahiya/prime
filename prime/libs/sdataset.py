from typing import Optional, Any
from argparse import Namespace

import numpy as np
from deepxml.libs.sdataset import DatasetIS
from xclib.evaluation.xc_metrics import compute_inv_propesity


class DatasetIS(DatasetIS):
    """Dataset to load and use XML-Datasets with sparse 
       classifiers or embeddings
       * Use with in-batch sampling
    """
    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 sampling_params: Optional[Namespace]=None,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_t: str='sparse',
                 label_type: str='sparse',
                 max_len: int=-1,
                 n_pos: int=1,
                 A: float=0.55,
                 B: float=1.5,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        """
        Args:
            data_dir (str): data directory
            f_features (str): file containing features
                Support for sparse, dense and sequential features
            f_labels (str): file containing labels
                Support for sparse or dense
                * sparse will return just the positives
                * dense will return all the labels as a dense array
            f_label_features (Optional[str], optional): file containing label features.
            Defaults to None. Support for sparse, dense and sequential features
            data (dict, optional): preloaded features and labels.
            Defaults to {'X': None, 'Y': None, 'Yf': None}.
            mode (str, optional): train or test. Defaults to 'train'.
            may be useful in cases where different things are applied 
            to train or test set
            sampling_params (Namespace, optional): Parameters for sampler. Defaults to None.
            n_pos (int, optional): Number of positives for each item
                * n_pos specified in sampling_params will take priority
            normalize_features (bool, optional): unit normalize? Defaults to True.
            normalize_lables (bool, optional): inf normalize? Defaults to False.
            feature_t (str, optional): feature type. Defaults to 'sparse'.
            label_type (str, optional): label type. Defaults to 'dense'.
            max_len (int, optional): max length. Defaults to -1.
        """
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         sampling_params=sampling_params,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_t=feature_t,
                         label_type=label_type,
                         mode=mode,
                         n_pos=n_pos
                        )

        self.prob = compute_inv_propesity(
            self.labels.data,
            A=sampling_params.A,
            B=sampling_params.B)

    def _sample_y_and_yf(self, ind, n=-1):
        if n == -1 or len(ind) < n or len(ind) == 0:
            sampled_ind = ind
        else:
            p = self.prob[ind]
            sampled_ind = np.random.choice(ind, p=p/sum(p), size=n), 
        Yf = None if self.label_features is None \
            else self.label_features[sampled_ind]
        return sampled_ind, Yf
