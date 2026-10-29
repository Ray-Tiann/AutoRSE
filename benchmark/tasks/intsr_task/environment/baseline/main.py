import argparse
import datetime
import logging
import os

from models import *
from utils import *
from utils import const, utils


def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_seed', type=int, default=int(os.environ.get("RESEARCH_SEED", "2024")))
    parser.add_argument('--time', type=str, default='none')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--metadata_path', type=str, default='')

    parser.add_argument('--data', type=str, default='KuaiSAR')
    parser.add_argument('--model', type=str, default='UniSAR')
    return parser


if __name__ == '__main__':
    global_start_time = datetime.datetime.now()

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser = parse_global_args(preliminary_parser)
    preliminary_args, _ = preliminary_parser.parse_known_args()
    model_registry = {
        'UniSAR': UniSAR,
        'IntSR': IntSR,
    }
    if preliminary_args.model not in model_registry:
        raise ValueError(
            'Unknown model {}. Available models: {}'.format(
                preliminary_args.model, ', '.join(model_registry)))
    model_class = model_registry[preliminary_args.model]

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_class.parse_model_args(parser)
    parser = SarRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()

    if args.gpu == 'cpu':
        args.device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.gpu != '' and torch.cuda.is_available():
            args.device = torch.device('cuda')

    if args.data == 'KuaiSAR':
        const.init_setting_KuaiSAR(args.data_root)
    elif args.data == 'Amazon':
        const.init_setting_Amazon(args.data_root)
    else:
        raise ValueError('Dataset Error')


    if args.time == 'none':
        cur_time = datetime.datetime.now()
        args.time = cur_time.strftime(r"%Y%m%d-%H%M%S")

    args.model_path = "output/{}/{}/checkpoints/{}".format(
        args.data, args.model, args.time)

    utils.load_hyperparam(args)
    # YAML provides the experiment defaults, while explicitly supplied CLI
    # flags retain the final say (useful for short smoke runs such as
    # ``--epoch 2``). The original code loaded YAML after parsing and silently
    # discarded command-line overrides.
    args = parser.parse_args(namespace=args)
    utils.setup_seed(args.random_seed)
    utils.set_logging(args)
    print(args)
    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {}'.format(flag, value))

    model: BaseModel = model_class(args).to(args.device)
    runner: BaseRunner = SarRunner(args)

    num_parameters = model.count_variables()
    logging.info("num model parameters:{}".format(num_parameters))

    if args.train == 0:
        raise ValueError(
            'Labeled test evaluation is isolated in score.sh. Use predict.py '
            'to score test_data_unlabeled.pkl.')

    training_summary = runner.train(model)

    global_end_time = datetime.datetime.now()
    logging.info("running used time:{}".format(global_end_time -
                                               global_start_time))
    utils.write_run_metadata(
        args.metadata_path, args, global_start_time, global_end_time,
        training_summary)
