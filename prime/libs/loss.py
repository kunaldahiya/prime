import torch
from deepxml.libs.loss import _Loss


class TripletMarginLossOHNM(_Loss):
    """ Triplet Margin Loss with Online Hard Negative Mining
    * Applies loss using the hardest negative in the mini-batch
    """
    def __init__(
            self,
            margin: float = 0.3,
            margin_min: float = 0.0,
            eps: float = 1.0e-6,
            reduction: str = 'custom',
            num_negatives: int = 100,
            num_violators: bool = False,
            tau: float = 0.1):
        """
        Args:
            margin (float, optional): Margin in triplet loss. Defaults to 0.3
            margin_min (float, optional): Min Margin in triplet loss. Defaults to 0.0
                Applies triplet loss with dynamic margin if this is > 0.0
            eps (float, optional): for numerical safety. Defaults to 1.0e-6.
            reduction (str, optional): how to reduce the batch. Defaults to 'custom'
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum over the labels and mean acorss data-points
            num_negatives (int, optional): #negatives per data point. Defaults to 100
            num_violators (bool, optional): Defaults to False.
              number of labels violating the margin.
            tau (float, optional): temprature in similarity. Defaults to 0.1.
              rescale the logits as per the temprature (tau)
        """
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.mx_lim = 100
        self.mn_lim = -100
        self.tau = tau
        self._eps = eps
        self.margin = margin
        self.margin_min = margin_min
        self.reduction = reduction
        self.num_negatives = num_negatives
        self.num_violators = num_violators
        self.recale = tau != 1.0
        self.dynamic_margin = self.margin_min > 0.0

    def forward(
            self, 
            output: torch.Tensor, 
            target: torch.Tensor, *args) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): cosine similarity b/w label and document
              real number pred matrix of size: batch_size x output_size
            target (torch.Tensor): 0/1 ground truth matrix of 
              size: batch_size x output_size

        Returns:
            torch.Tensor: loss for the given data
        """
        with torch.no_grad():
            indices = torch.multinomial(target, 1, replacement=False)

        sim_p = output.gather(1, indices.view(-1, 1))

        similarities = torch.where(
            target == 0, 
            output, 
            torch.full_like(output, self.mn_lim))

        _, indices = torch.topk(
            similarities, 
            largest=True, 
            dim=1, 
            k=self.num_negatives)

        sim_n = output.gather(1, indices)

        if self.dynamic_margin:
            d_margin = torch.clamp(
                input=torch.abs(sim_p - sim_n).detach(),
                min=self.margin_min,
                max=self.margin)
        else:
            d_margin = self.margin

        loss = torch.max(
            torch.zeros_like(sim_p), 
            sim_n - sim_p + d_margin)

        if self.recale:
            sim_n[loss == 0] = self.mn_lim
            prob = torch.softmax(sim_n/self.tau, dim=1)
            loss = loss * prob
        
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss


class RegLoss(_Loss):
    """ Regularization
    * The final representation should induce less loss than intermediate
    """
    def __init__(
            self,
            margin: float = 0.1,
            reduction: str = 'custom',
            num_negatives: int = 10):
        """
        Args:
            margin (float, optional): Margin in triplet loss. Defaults to 0.3
            reduction (str, optional): how to reduce the batch. Defaults to 'custom'
                * Warning: Hardcoded for now
            num_negatives (int, optional): #negatives per data point. Defaults to 10
        """
        super(RegLoss, self).__init__(reduction=reduction)
        self.mx_lim = 100
        self.mn_lim = -100
        self.margin = margin
        self.k = num_negatives

    def forward(
            self, 
            output: torch.Tensor,  
            target: torch.Tensor, 
            output_i: torch.Tensor,
            mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): cosine similarity b/w label and document
              real number pred matrix of size: batch_size x output_size
            target (torch.Tensor): 0/1 ground truth matrix of 
              size: batch_size x output_size
            output (torch.Tensor): cosine similarity b/w label and document 
                in intermediate space. Real number pred matrix 
                of size: batch_size x output_size
            mask (torch.Tensor): boolean mask 
                * ignored for now
                            
        Returns:
            torch.Tensor: loss for the given data
        """
        with torch.no_grad():
            indices = torch.multinomial(target, 1, replacement=False)

        # get positive loss
        sim_p_i = output_i.gather(1, indices)
        sim_p_f = output.gather(1, indices)
        loss_p = torch.max(
            torch.zeros_like(sim_p_i), 
            sim_p_i - sim_p_f + self.margin).flatten()

        # get negative loss
        similarities = torch.where(
            target == 0, 
            output, 
            torch.full_like(output, self.mn_lim))

        _, indices = torch.topk(
            similarities, 
            largest=True, 
            dim=1, k=self.k)
        sim_n_f = output.gather(1, indices)
        sim_n_i = output_i.gather(1, indices)
        
        loss_n = torch.max(
            torch.zeros_like(sim_n_f), 
            sim_n_f - sim_n_i + self.margin).flatten()
        
        # ignore non zero-entries
        loss = loss_p.sum() / (loss_p > 0).sum() + loss_n.sum() / (loss_n > 0).sum()
        return loss / 2


class TripletLossWReg(torch.nn.Module):
    """ Triplet Margin Loss with Online Hard Negative Mining
    * Applies loss using the hardest negative in the mini-batch
    """
    def __init__(
            self,
            margin: float = 0.3,
            margin_min: float = 0.0,
            eps: float = 1.0e-6,
            reduction: str = 'custom',
            num_negatives: int = 100,
            tau: float = 0.1,
            reg: float = 0.1,
            dual: float = 1.0,
            inter: float = 1.0,
            *kwargs
            ) -> None:
        """
        Args:
            margin (float, optional): Margin in triplet loss. Defaults to 0.3
            margin_min (float, optional): Min Margin in triplet loss. Defaults to 0.0
                Applies triplet loss with dynamic margin if this is > 0.0
            eps (float, optional): for numerical safety. Defaults to 1.0e-6.
            reduction (str, optional): how to reduce the batch. Defaults to 'mean'
              - 'none': no reduction will be applied
              - 'mean' or 'sum': mean or sum of loss terms
              - 'custom': sum over the labels and mean acorss data-points
            num_negatives (int, optional): #negatives per data point. Defaults to 10
            num_violators (bool, optional): Defaults to False.
              number of labels violating the margin.
            tau (float, optional): temprature in similarity. Defaults to 0.1.
              rescale the logits as per the temprature (tau)
            reg (float, optional): Apply regularization term. Defaults to 0.1.
                * positive pair should be closer in final space wrt intermediate
                * negative pair should be apart in final space wrt intermediate
            dual (float, optional): Defaults to 1.0.
                Apply loss for inversed pairs too (makes sense only for triplet style loss)
            inter (float, optional): Defaults to 1.0.
                Apply loss on both intermediate and final space.
        """
        super(TripletLossWReg, self).__init__()
        self.criterion = TripletMarginLossOHNM(
            margin=margin,
            margin_min=margin_min,
            eps=eps,
            reduction=reduction,
            num_negatives=num_negatives,
            num_violators=False,
            tau=tau
        ) 
        self.regularization = RegLoss(
            margin=margin,
            reduction=None,
            num_negatives=num_negatives
        )
        self.inter = inter
        self.dual = dual
        self.reg = reg

    def forward(
            self, 
            output: torch.Tensor, 
            target: torch.Tensor, 
            output_i: torch.Tensor = None, 
            mask: torch.Tensor = None):
        """
        Args:
            output (torch.Tensor): cosine similarity b/w label and document
              real number pred matrix of size: batch_size x output_size
            target (torch.Tensor): 0/1 ground truth matrix of 
              size: batch_size x output_size
            output_i (torch.Tensor, optional): Defaults to None 
                cosine similarity b/w label and document in intermediate space. 
                Real number pred matrix of size: batch_size x output_size
            mask (torch.Tensor): boolean mask 
                * ignored for now
                            
        Returns:
            torch.Tensor: loss for the given data
        """
        loss = self.criterion(output, target) 
        if self.inter > 0:
            loss += self.inter * self.criterion(output_i, target) 
        if self.dual > 0:
            loss += self.dual * self.criterion(output_i.T, target.T)
        if self.reg > 0:
            loss += self.reg * self.regularization(output, target, output_i)
        return loss
    
