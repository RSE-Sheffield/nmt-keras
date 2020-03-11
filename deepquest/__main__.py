import argparse
import sys


def train(args):

    if args.gpuid:
        set_gpu_id(args.gpuid)

    from  . import train
    train(args.config, args.changes)


def predict(args):
    from . import predict
    predict(args.model, args.dataset, save_path=args.save_path, evalset=args.evalset, changes=args.changes)


def score(args):
    from . import score
    score(args.files)

def set_gpu_id(gpuid):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpuid = gpuid.split(',') if ',' in gpuid else gpuid.split()
    gpustr = ''
    for g in gpuid:
        gpustr += str(g).strip() + ','
    gpustr = gpustr[0:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpustr

def main():
    parser = argparse.ArgumentParser(prog='{deepquest, dq}',
                                    description='A framework for neural-based quality estimation.')
    subparsers = parser.add_subparsers(help='mode of operation')

    # parser for help text
    help_parser = subparsers.add_parser('help', help='Show the help text')
    help_parser.set_defaults(func=help)

    # parser for training
    train_parser = subparsers.add_parser('train', help='Train QE models')
    train_parser.set_defaults(func=train)
    train_parser.add_argument("-c", "--config",   required=False,
                              help="Config YAML or pkl for loading the model configuration. ")
    train_parser.add_argument("--changes", nargs="*", help="Changes to config. "
                              "Following the syntax Key=Value",
                              default="")
    train_parser.add_argument("help", nargs='?', help="Show the help information.")
    train_parser.add_argument("--gpuid", type=str, required=False,
                            help="One or more integers specifying GPU device IDs (default 0)")

    # parser for prediction
    predict_parser = subparsers.add_parser('predict', help='Sample using trained QE models')
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("--model", required=False,
                                help="model file (.h5) to use")
    predict_parser.add_argument("--dataset", required=False,
                                help="dataset file (.pkl) to use")
    predict_parser.add_argument("--save_path", required=False, help="Directory path to save predictions to. "
                                "If not specified, defaults to STORE_PATH")
    predict_parser.add_argument("--evalset", required=False, help="Set to evaluate on. "
                                "Defaults to 'test' if not specified. ")
    predict_parser.add_argument("--changes", nargs="*", help="Changes to config. "
                                "Following the syntax Key=Value",
                                default="")
    predict_parser.add_argument("help", nargs='?', help="Show the help information.")

    # parser for scoring
    score_parser = subparsers.add_parser(
        'score', help='Evaluate a set of predictions with regard to a reference set. ')
    score_parser.add_argument("help", nargs='?', help="Show the help information.")
    score_parser.set_defaults(func=score)
    score_parser.add_argument("files", nargs="*", help="Two text files containing predictions and references. ", default="")

    args = parser.parse_args()

    if (not hasattr(args,'func')) or (len(sys.argv) == 1):
        parser.print_help()
    elif hasattr(args, 'func') and (len(sys.argv) == 2):
        parser.print_help()
    elif hasattr(args, 'func') and 'help' in sys.argv:
        parsers = {'train': train_parser,
                   'predict': predict_parser,
                   'score': score_parser}
        parsers[str(args.func.__name__)].print_help()
    else:
        args.func(args)
    sys.exit(0)

if __name__ == "__main__":
    main()
