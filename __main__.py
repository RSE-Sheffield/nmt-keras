import argparse


def train(args):
    import yaml
    if args.config.endswith('.yml'):
        # FIXME make this a user option (maybe depend on model type and level?)
        with open('configs/default-config-BiRNN.yml') as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        with open(args.config) as file:
            user_parameters = yaml.load(file, Loader=yaml.FullLoader)
        parameters.update(user_parameters)
        del user_parameters
    elif args.config.endswith('.pkl'):
        parameters = update_parameters(parameters, pkl2dict(args.config))
    parameters.update(changes2dict(args))
    if parameters.get('SEED') is not None:
        print('Setting deepQuest seed to', parameters['SEED'])
        import numpy.random
        numpy.random.seed(parameters['SEED'])
        import random
        random.seed(parameters['SEED'])
    import train
    train.main(parameters, args.dataset)


def predict(args):
    import predict
    predict.main(args.model, args.dataset, args.save_path, args.evalset, changes2dict(args))


def score(args):
    import score
    score.main(args.files)


def changes2dict(args):
    import ast
    if args.changes:
        changes = {}
        try:
            for arg in args.changes:
                try:
                    k, v = arg.split('=')
                except ValueError:
                    print('Overwriting arguments must have the form key=value.\n This one is: %s' % str(changes))
                    exit(1)
                if '_' in v:
                    changes[k] = v
                else:
                    try:
                        changes[k] = ast.literal_eval(v)
                    except ValueError:
                        changes[k] = v
        except ValueError:
            print("Error processing arguments: {!r}".format(arg))
            exit(2)
        return changes
    else:
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "A framework for neural-based quality estimation for machine translation. ")
    subparsers = parser.add_subparsers(help='train '
                                            'predict '
                                            'score ')

    # parser for training
    train_parser = subparsers.add_parser('train', help='Train QE models')
    train_parser.set_defaults(func=train)
    train_parser.add_argument("-c", "--config",   required=False,
                              help="Config YAML or pkl for loading the model configuration. ")
    train_parser.add_argument("-ds", "--dataset", required=False,
                              help="Optional dataset instance to be trained on. ")
    train_parser.add_argument("changes", nargs="*", help="Changes to config. "
                              "Following the syntax Key=Value",
                              default="")

    # parser for prediction
    predict_parser = subparsers.add_parser('predict', help='Sample using trained QE models')
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("--model", required=True,
                                help="model file (.h5) to use")
    predict_parser.add_argument("--dataset", required=True,
                                help="dataset file (.pkl) to use")
    predict_parser.add_argument("--save_path", required=False, help="Directory path to save predictions to. "
                                "If not specified, defaults to STORE_PATH")
    predict_parser.add_argument("--evalset", required=False, help="Set to evaluate on. "
                                "Defaults to 'test' if not specified. ")
    predict_parser.add_argument("changes", nargs="*", help="Changes to config. "
                                "Following the syntax Key=Value",
                                default="")

    # parser for scoring
    score_parser = subparsers.add_parser(
        'score', help='Evaluate a set of predictions with regard to a reference set. ')
    score_parser.set_defaults(func=score)
    score_parser.add_argument(
        "files", nargs=2, help="Two text files containing predictions and references. ")

    args = parser.parse_args()
    args.func(args)
