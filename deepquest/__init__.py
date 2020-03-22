from deepquest.utils.utils import setparameters, changes2dict

def train(config, changes={}):
    """
    Handles QE model training.
    :param config: Either a path to a YAML or pkl config file or a dictionary of parameters.
    :param dataset: Optional path to a previously built pkl dataset.
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    parameters = setparameters(user_config_path=config)

    if changes:
        parameters.update(changes2dict(changes))

    if parameters.get('SEED') is not None:
        print('Setting deepQuest seed to', parameters['SEED'])
        import numpy.random
        numpy.random.seed(parameters['SEED'])
        import random
        random.seed(parameters['SEED'])

    import deepquest.train
    train.main(parameters)


def predict(config, changes={}):
    """
    Predicts QE scores on a dataset using a pre-trained model.
    :param model: Model file (.h5) to use.
    :param dataset: Dataset file (.pkl) to use.
    :param save_path: Optinal directory path to save predictions to. Default = STORE_PATH
    :param evalset: Optional set to evaluate on. Default = 'test'
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    parameters = setparameters(user_config_path=config)

    # TODO: determine whether we want to allow a user to update parameters as
    # this could result in incompatibility (e.g. different toknization for the data)
    # if changes:
    #     parameters.update(changes2dict(changes))

    if parameters.get('SEED') is not None:
        print('Setting deepQuest seed to', parameters['SEED'])
        import numpy.random
        numpy.random.seed(parameters['SEED'])
        import random
        random.seed(parameters['SEED'])

    import deepquest.predict
    predict.main(parameters)


def score(files):
    """
    Evaluate a set of predictions with regard to a reference set.
    :param files: List of paths to two text files containing predictions and references.
    """
    import deepquest.score
    score.main(files)
