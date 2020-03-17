def train(config=None, changes={}):
    """
    Handles QE model training.
    :param config: Either a path to a YAML or pkl config file or a dictionary of parameters.
    :param dataset: Optional path to a previously built pkl dataset.
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    import deepquest.train
    train.main(config, changes)

def predict(model, dataset, save_path=None, evalset=None, changes={}):
    """
    Predicts QE scores on a dataset using a pre-trained model.
    :param model: Model file (.h5) to use.
    :param dataset: Dataset file (.pkl) to use.
    :param save_path: Optinal directory path to save predictions to. Default = STORE_PATH
    :param evalset: Optional set to evaluate on. Default = 'test'
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    import deepquest.predict
    predict.main(model=model, dataset=dataset, save_path=save_path, evalset=evalset, changes=changes)

def score(files):
    """
    Evaluate a set of predictions with regard to a reference set.
    :param files: List of paths to two text files containing predictions and references.
    """
    import deepquest.score
    score.main(files)
