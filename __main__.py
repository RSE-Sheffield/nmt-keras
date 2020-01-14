import argparse
import sys

def train(args):
    import train
    train.main(args)

def predict(args):
    import predict
    predict.main(args)

def score(args):
    import score
    score.main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A framework for neural-based quality estimation for machine translation. ")
    subparsers = parser.add_subparsers(help='train '
                                            'predict '
                                            'score ')

    # parser for training
    train_parser = subparsers.add_parser('train', help='Train QE models')
    train_parser.set_defaults(func=train)
    train_parser.add_argument("-c", "--config",   required=False, help="Config YAML or pkl for loading the model configuration. ")
    train_parser.add_argument("-ds", "--dataset", required=False, help="Dataset instance with data")
    train_parser.add_argument("changes", nargs="*", help="Changes to config. "
                                                   "Following the syntax Key=Value",
                                                    default="")

    # parser for prediction
    predict_parser = subparsers.add_parser('predict', help='Sample using trained QE models')
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("--dataset", required=True,
        help="dataset file (.pkl) to use")
    predict_parser.add_argument("--model", required=True,
        help="model file (.h5) to use")
    predict_parser.add_argument("--save_path", required=False, help="Directory path to save predictions to. "
                                                                "If not specified, defaults to STORE_PATH")
    predict_parser.add_argument("--evalset", required=False, help="Set to evaluate on. "
                                                                "Defaults to 'test' if not specified. ")
    predict_parser.add_argument("changes", nargs="*", help="Changes to config. "
                                                   "Following the syntax Key=Value",
                                                   default="")

    # parser for scoring
    score_parser = subparsers.add_parser('score', help='Compute a score for a set of predictions vs reference set. ')
    score_parser.set_defaults(func=score)
    score_parser.add_argument("files", nargs=2, help="Two text files containing predictions and references. ")

    args = parser.parse_args()
    args.func(args)
