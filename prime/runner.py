import os
import sys
import json
from argparse import Namespace
from main import main


def update_args(args, dict):
    for key, value in dict.items():
        setattr(args, key, value)


def setup_dirs(args, work_dir, pipeline='PRIME', module='siamese'):
    args.data_dir = os.path.join(work_dir, 'data')
    arch = args._arch if hasattr(args, '_arch') else args.arch

    args.result_dir = os.path.join(
        work_dir, 
        'results', 
        pipeline, 
        arch, 
        args.dataset, 
        f'v_{args.version}',
        module)

    args.model_dir = os.path.join(
        work_dir, 
        'models', 
        pipeline, 
        arch, 
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

    setup_dirs(args, work_dir, pipeline=pipeline, module='siamese')

    # train intermediate representation
    args.mode = 'train'
    args._arch = args.arch
    args.arch = os.path.join(os.getcwd(), 'configs', f'{args.arch}.json')

    # main(args)
    try:
        os.symlink(
            os.path.join(args.model_dir, f'{args.model_fname}.network.pt'), 
            os.path.join(os.path.dirname(args.model_dir), "Z.pt"))
    except FileExistsError:
        pass

    update_args(args, config['classifier'])
    args.stage = 'classifier'

    setup_dirs(args, work_dir, pipeline=pipeline, module='classifier')
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