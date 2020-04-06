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
    
    if parameters.get('GPU_ID') is not None:
        n_gpus = set_gpu_id(str(parameters.get('GPU_ID')))
        parameters.update({'N_GPUS': n_gpus, 'GPU_ID': parameters.get('GPU_ID')})

    import deepquest.train
    train.main(parameters)

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
    predict.main(model, dataset, save_path, evalset, changes2dict(changes))

def score(files):
    """
    Evaluate a set of predictions with regard to a reference set.
    :param files: List of paths to two text files containing predictions and references.
    """
    import deepquest.score
    score.main(files)

def set_gpu_id(gpuid):
    """
    Sets the environment variable CUDA_VISIBLE_DEVICES to control which GPUs are used for training.
    Must only be called from deepquest/__main__.py, ie before import of deepquest module
    :param gpuid: String of comma-separated integers refering to GPU devices.
    """
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpuid = gpuid.split(',') if ',' in gpuid else gpuid.split()
    gpustr = ''
    gpucount = 0
    for g in gpuid:
        gpustr += str(g).strip() + ','
        gpucount += 1
    gpustr = gpustr[0:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpustr

    return gpucount