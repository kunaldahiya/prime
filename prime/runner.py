import os
import sys
import json
import torch
from argparse import Namespace
import numpy as np
from main import main
from deepxml.libs import utils
from deepxml.libs.evaluater import Evaluater
from xclib.data.data_utils import read_gen_sparse


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize_text(
        data_dir: str, 
        tokenizer_name: str='bert-base-uncased', 
        max_len: int | tuple[int]=32, 
        num_threads: int=1, 
        do_lower_case: bool=True) -> None:
    print("Tokenizing text..")
    #This is to support different lengths for inputs and outputs
    _inputs = ['trn', 'tst']
    _outputs = ['lbl']
    prefixes = _inputs + _outputs
    if isinstance(max_len, int):
        max_lens = [max_len] * len(prefixes)
    else:
        max_lens = [max_len[0]]*len(_inputs) + [max_len[1]] * len(_outputs)

    for f, l in zip(prefixes, max_lens):
        utils.tokenize_corpus(
            corpus=os.path.join(data_dir, f'{f}.raw.txt'), 
            tokenizer_type=tokenizer_name,
            tokenization_dir=data_dir,
            max_len=l, 
            prefix=f,
            do_lower_case=do_lower_case,
            num_threads=num_threads, 
            batch_size=100000)


def prepare_data(data_dir):
    print("Extracting text and labels from json.gz files..")
    if 'title' in os.path.basename(data_dir).lower():
        fields = ['title']
    else:
        fields = ["title", 'description']

    num_labels = utils.count_num_labels(os.path.join(data_dir, 'lbl.json.gz'))

    utils.extract_text_labels(
        in_fname=os.path.join(data_dir, 'lbl.json.gz'), 
        op_tfname=os.path.join(data_dir, 'lbl.raw.txt'), 
        op_lfname=None, 
        fields=fields, 
        num_labels=num_labels
    )

    utils.extract_text_labels(
        in_fname=os.path.join(data_dir, 'tst.json.gz'), 
        op_tfname=os.path.join(data_dir, 'tst.raw.txt'), 
        op_lfname=os.path.join(data_dir, 'tst_X_Y.txt'), 
        fields=fields, 
        num_labels=num_labels
    )

    utils.extract_text_labels(
        in_fname=os.path.join(data_dir, 'trn.json.gz'), 
        op_tfname=os.path.join(data_dir, 'trn.raw.txt'), 
        op_lfname=os.path.join(data_dir, 'trn_X_Y.txt'), 
        fields=fields, 
        num_labels=num_labels
    )


def evaluate(args):
    ev = Evaluater(
        A=args.A,
        B=args.B,
        labels=read_gen_sparse(os.path.join(args.data_dir, args.trn_label_fname)),
        filter_map=os.path.join(args.data_dir, args.tst_filter_fname) if hasattr(args, 'tst_filter_fname') else None
    )
    out = ev(
        read_gen_sparse(os.path.join(args.data_dir, args.tst_label_fname)),
        read_gen_sparse(os.path.join(args.result_dir, args.pred_fname))
    )
    print(out)
    return str(out)


def update_args(args, dict):
    for key, value in dict.items():
        setattr(args, key, value)


def setup_dirs(args, work_dir, method='PRIME', module='siamese'):
    args.data_dir = os.path.join(work_dir, 'data', args.dataset)
    arch = args._arch if hasattr(args, '_arch') else args.arch

    args.result_dir = os.path.join(
        work_dir, 
        'results', 
        method, 
        arch, 
        args.dataset, 
        f'v_{args.version}',
        module)

    args.model_dir = os.path.join(
        work_dir, 
        'models', 
        method, 
        arch, 
        args.dataset, 
        f'v_{args.version}',
        module)

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)


def run(work_dir: str, method: str, version: str, seed: str, config: dict):

    # Directory and filenames
    # fetch arguments/parameters like dataset name, A, B etc.
    args = Namespace(**config['global'])

    # fetch parameters specific to first argument
    update_args(args, config['siamese'])
    args.seed = seed
    args.version = version
    args.stage = 'siamese'

    setup_dirs(args, work_dir, method=method, module='siamese')

    # train intermediate representation
    args.mode = 'train'
    args._arch = args.arch
    args.arch = os.path.join(os.getcwd(), 'configs', f'{args.arch}.json')

    main(args)
    try:
        os.symlink(
            os.path.join(args.model_dir, f'{args.model_fname}.network.pt'), 
            os.path.join(os.path.dirname(args.model_dir), "Z.pt"))
    except FileExistsError:
        pass

    update_args(args, config['classifier'])
    args.stage = 'classifier'

    args.arch = os.path.join(os.getcwd(), 'configs', f'{args._arch}.clf.json')
    setup_dirs(args, work_dir, method=method, module='classifier')
    main(args)

    args.mode = 'predict'
    main(args)

    result = evaluate(args)
    # Dump in the file inside result dir
    with open(os.path.join(os.path.dirname(args.result_dir), 'log_eval.txt'), 'w') as fp:
        fp.write(result)


if __name__ == "__main__":
    method = sys.argv[1]
    work_dir = sys.argv[2]
    dataset = sys.argv[3]
    version = sys.argv[4]
    seed = int(sys.argv[5])

    set_seed(seed)

    config = json.load(
        open(os.path.join(os.getcwd(), 'configs', method, f'{dataset}.json')))

    g_config = config["global"]
    data_dir = os.path.join(work_dir, 'data', g_config['dataset'])

    if not os.path.isfile(os.path.join(data_dir, "trn.raw.txt")):
        print("Preparing data; Will overwrite!")
        prepare_data(data_dir)
        tokenize_text(data_dir, g_config['tokenizer_name'], g_config['max_length'])  

    if method == "PRIME" or method == "PRIME++":
        run(
            method=method,
            work_dir=work_dir,
            version=f"{version}_{seed}",
            seed=seed,
            config=config)
    else:
        raise NotImplementedError("")
