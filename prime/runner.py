import os
import sys
import json
from argparse import Namespace
from main import main


def update_args(args, dict):
    for key, value in dict.items():
        setattr(args, key, value)


def setup_dirs(args, work_dir, module='siamese'):
    args.data_dir = os.path.join(work_dir, 'data')
    
    args.result_dir = os.path.join(
        work_dir, 
        'results', 
        pipeline, 
        args.arch, 
        args.dataset, 
        f'v_{args.version}',
        module)

    args.model_dir = os.path.join(
        work_dir, 
        'models', 
        pipeline, 
        args.arch, 
        args.dataset, 
        f'v_{args.version}',
        module)

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)


def run(work_dir, pipeline, version, seed, config):

    # Directory and filenames
    # fetch arguments/parameters like dataset name, A, B etc.
    args = Namespace(**config['global'])

    # fetch parameters specific to first argument
    update_args(args, config['siamese'])
    args.seed = seed
    args.version = version
    args.stage = 'siamese'

    setup_dirs(args, work_dir, module='siamese')

    # train intermediate representation
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), 'configs', f'{args.arch}.json')

    main(args)



if __name__ == "__main__":
    pipeline = sys.argv[1]
    work_dir = sys.argv[2]
    version = sys.argv[3]
    config = sys.argv[4]
    seed = int(sys.argv[5])
    if pipeline == "PRIME" or pipeline == "PRIME++":
        run(
            pipeline=pipeline,
            work_dir=work_dir,
            version=f"{version}_{seed}",
            seed=seed,
            config=json.load(open(config)))
    else:
        raise NotImplementedError("")