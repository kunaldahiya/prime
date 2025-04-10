from deepxml.libs.optim import construct_optimizer
from deepxml.libs.schedular import construct_schedular
from deepxml.libs.evaluater import Evaluater
from deepxml.libs.shortlist import Shortlist

from libs import utils 
from models.network import SiameseNetworkIS, XCNetworkIS
from libs.pipeline import EmbeddingPipelineIS, XCPipelineIS
from libs.loss import TripletLossWReg, TripletMarginLossOHNM


def construct_shortlister(args):
    return Shortlist(
        method=args.ann_method,
        M=args.M,
        efC=args.efC,
        efS=args.efS,
        num_neighbours=args.top_k)


def construct_network(args):
    if args.stage == 'siamese':
        return SiameseNetworkIS.from_config(
            config=args.arch, 
            device="cuda", 
            args=args
        )    
    elif args.stage == 'xc':
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


def construct_pipeline(net, criterion, optim, schedular, shortlister, args):
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
    elif args.stage == 'xc':
        net = XCPipelineIS(
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


def construct_loss(args):
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
    elif args.stage == 'xc':
        return TripletMarginLossOHNM(
              margin=args.loss_margin,
              reduction=args.loss_reduction,
              num_negatives=args.loss_num_negatives,
        )
    else:
        raise NotImplementedError("")


def construct_opt_schedular(net, args):
    optim = construct_optimizer(net, args)
    schedular = construct_schedular(optim, args)
    return optim, schedular


def train(pipeline, args):
    """Train the model with given data
    Arguments
    ----------
    pipeline: A wrapper object to handle training/ validation etc.
        train the given model as per given parameters (using .fit())
    args: NameSpace
        arguments like data file names, sampling, epochs etc., 
    """
    trn_fname = {
        'f_features': args.trn_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.trn_label_fname}
    val_fname = {
        'f_features': args.val_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.val_label_fname,
        'f_label_filter': args.val_filter_fname}

    args.sampling_update_steps = list(
        range(min(args.sampling_curr_steps), 
              args.num_epochs, 
              args.sampling_update_interval))
    output = pipeline.fit(
        data_dir=args.data_dir,
        dataset=args.dataset,
        trn_fname=trn_fname,
        val_fname=val_fname,
        validate=True,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        sampling_params=utils.filter_params(args, 'sampling_'),
        feature_t=args.feature_t,
        validate_interval=args.validate_interval,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        inference_t=args.inference_t,
        batch_size=args.batch_size)
    pipeline.save(fname=args.model_fname)
    return output



def main(args):
    print(args)
    net = construct_network(args)
    net.to("cuda")
    print(net)
    if args.mode == 'train':
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


if __name__ == "__main__":
    pass