from argparse import Namespace
from torch.nn import Module
from torch.optim import Optimizer
from deepxml.libs.pipeline import PipelineBase
from torch.optim.lr_scheduler import LRScheduler
from scipy.sparse import spmatrix

from deepxml.libs.optim import construct_optimizer
from deepxml.libs.schedular import construct_schedular
from deepxml.libs.evaluater import Evaluater
from deepxml.libs.shortlist import Shortlist

import os
import json
import torch
from libs import utils 
from models.network import SiameseNetworkIS, XCNetworkIS
from libs.pipeline import EmbeddingPipelineIS, XCPipelineIS
from libs.loss import TripletLossWReg, TripletMarginLossOHNM


def construct_shortlister(args: Namespace) -> Shortlist:
    return Shortlist(
        method=args.ann_method,
        space=args.ann_space,
        M=args.M if hasattr(args, 'M') else 100,
        efC=args.efC if hasattr(args, 'efC') else 300,
        efS=args.efS if hasattr(args, 'efS') else args.top_k,
        num_neighbours=args.top_k)


def construct_network(args: Namespace) -> Module:
    if args.stage == 'siamese':
        return SiameseNetworkIS.from_config(
            config=args.arch, 
            device="cuda", 
            args=args
        )    
    elif args.stage == 'classifier':
        net = XCNetworkIS.from_config(
            config=args.arch, 
            device="cuda", 
            args=args
        )    
        if args.freeze_encoder:
            for params in net.encoder.parameters():
                params.requires_grad = False
        return net
    else:
        raise NotImplementedError("")


def initialize(args: Namespace, net: Module) -> None:
    if args.init == 'intermediate':
        print("Loading intermediate representation.")
        d = torch.load(
            os.path.join(os.path.dirname(args.model_dir), "Z.pt"),
            weights_only=True)
        net.load_state_dict(d, strict=False)
    elif args.init == 'auto':
        print("Automatic initialization.")
    elif args.init == 'checkpoint':
        print("Automatic initialization.")
    else:  # trust the random init
        print("Random initialization.")
    return net


def construct_pipeline(
        net: Module, 
        args: Namespace, 
        shortlister: Shortlist,
        criterion: Module=None, 
        optim: Optimizer=None, 
        schedular: LRScheduler=None) -> PipelineBase:
    if args.stage == 'siamese':
        return EmbeddingPipelineIS(
            net=net,
            criterion=criterion,
            optimizer=optim,
            shortlister=shortlister,
            evaluater=Evaluater(),
            schedular=schedular,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            use_amp=args.use_amp
        )    
    elif args.stage == 'classifier':
        return XCPipelineIS(
            net=net,
            criterion=criterion,
            optimizer=optim,
            shortlister=shortlister,
            evaluater=Evaluater(),
            schedular=schedular,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            use_amp=args.use_amp
        )    
    else:
        raise NotImplementedError("")


def construct_loss(args: Namespace) -> Module:
    if args.stage == 'siamese':
        return TripletLossWReg(
              margin=args.loss_margin,
              reduction=args.loss_reduction,
              num_negatives=args.loss_num_negatives,
              margin_min=args.loss_margin_min,
              reg=args.loss_reg,
              dual=args.loss_dual,
              inter=args.loss_inter,
        )    
    elif args.stage == 'classifier':
        return TripletMarginLossOHNM(
              margin=args.loss_margin,
              reduction=args.loss_reduction,
              num_negatives=args.loss_num_negatives,
        )
    else:
        raise NotImplementedError("")


def construct_opt_schedular(
        net: Module, 
        args: Namespace) -> tuple[Optimizer, LRScheduler]:
    optim = construct_optimizer(net, args)
    schedular = construct_schedular(optim, args)
    return optim, schedular


def train(pipeline: PipelineBase, args: Namespace) -> None:
    """Train the model with given data
    Arguments
    ----------
    pipeline: A wrapper object to handle training/ validation etc.
        train the given model as per given parameters (using .fit())
    args: NameSpace
        arguments like data file names, sampling, epochs etc., 
    """
    json.dump(
        args.__dict__, 
        open(os.path.join(args.result_dir, 'args.json'), 'w'),
        indent=4)
    
    trn_fname = {
        'f_features': args.trn_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.trn_label_fname}
    val_fname = {
        'f_features': args.val_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.val_label_fname,
        'f_label_filter': args.val_filter_fname}

    args.cache_doc_representations = False
    if args.freeze_encoder:
        args.cache_doc_representations = True

    args.sampling_update_steps = list(
        range(min(args.sampling_curr_steps), 
              args.num_epochs, 
              args.sampling_update_interval))
    pipeline.fit(
        data_dir=args.data_dir,
        trn_fname=trn_fname,
        val_fname=val_fname,
        validate=True,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        cache_doc_representations=args.cache_doc_representations,
        sampling_params=utils.filter_params(args, 'sampling_'),
        feature_t=args.feature_t,
        validate_interval=args.validate_interval,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        inference_t=args.inference_t,
        batch_size=args.batch_size)
    pipeline.save(fname=args.model_fname)


def predict(pipeline: PipelineBase, args: Namespace) -> spmatrix:
    """Train the model with given data
    Arguments
    ----------
    pipeline: A wrapper object to handle training/ validation etc.
        train the given model as per given parameters (using .fit())
    args: NameSpace
        arguments like data file names, sampling, epochs etc., 
    """
    tst_fname = {
        'f_features': args.tst_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.tst_label_fname,
        'f_label_filter': args.tst_filter_fname}

    output = pipeline.predict(
        data_dir=args.data_dir,
        fname=tst_fname,
        data=None,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        k=args.top_k,
        feature_t=args.feature_t)
    utils.save_predictions(
        output, 
        os.path.join(args.result_dir, args.pred_fname))
    return output


def main(args: Namespace):
    print(args)
    net = construct_network(args)
    net.to("cuda")
    print(net)
    if args.mode == 'train':
        initialize(args, net)
        criterion = construct_loss(args)
        optim, schedular = construct_opt_schedular(net, args)
        shortlister = construct_shortlister(args)
        pipeline = construct_pipeline(
            net=net,
            criterion=criterion,
            optim=optim,
            schedular=schedular,
            shortlister=shortlister,
            args=args)
        train(pipeline, args)

    elif args.mode == 'predict' or args.mode == 'inference':
        shortlister = construct_shortlister(args)
        pipeline = construct_pipeline(
            net=net,
            shortlister=shortlister,
            args=args)
        pipeline.load(fname=args.model_fname)
        predict(pipeline, args)


if __name__ == "__main__":
    pass
