import argparse
import sys


def train(args):
    from  . import train
    train(args.config, args.changes)


def predict(args):
    from . import predict
    # predict(model=args.model, dataset=args.dataset, directory=args.dir, filename=args.file, save_path=args.save_path, evalset=args.evalset, changes=changes2dict(args))
    predict(args.config, args.changes)

def score(args):
    from . import score
    score(args.files)


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

    # parser for prediction
    predict_parser = subparsers.add_parser('predict', help='Sample using trained QE models')
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("help", nargs='?', help="Show the help information.")
    predict_parser.add_argument("-c", "--config",   required=False,
                              help="Config YAML for loading the prediction configuration. ")
    # predict_parser.add_argument("--model", required=False,
    #                             help="model file (.h5) to use")
    # predict_parser.add_argument("--dataset", required=False,
    #                             help="dataset file (.pkl) to use")
    # predict_parser.add_argument("--dir", required=False,
    #                             help="Path to directory containing files to predict on. Default={DATA_ROOT_PATH}{evalset}.{SRC_LAN | TRG_LAN}")
    # predict_parser.add_argument("--file", required=False,
    #                             help="Common name of source and target language files to be predicted on. Default={evalset}")
    # predict_parser.add_argument("--save_path", required=False, help="Directory path to save predictions to. "
    #                             "If not specified, defaults to STORE_PATH")
    # predict_parser.add_argument("--evalset", required=False, help="Set to evaluate on. "
    #                             "Defaults to 'test' if not specified. ")
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
