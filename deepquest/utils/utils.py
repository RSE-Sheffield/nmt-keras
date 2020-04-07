import logging

from six import iteritems

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def update_parameters(params, updates, restrict=False):
    """
    Updates the parameters from params with the ones specified in updates
    :param params: Parameters dictionary to update
    :param updates: Updater dictionary
    :param restrict: If True, parameters from the original dict are not overwritten.
    :return:
    """
    for new_param_key, new_param_value in iteritems(updates):
        if restrict and params.get(new_param_key) is not None:
            params[new_param_key] = new_param_value
        else:
            params[new_param_key] = new_param_value

    return params

def compare_params(params_new, params_old, ignore=None):
    """
    Checks a new params dictionary against one from a previous model for differences.
    :param params_new: new params dictionary
    :param params_old: previous params dictionary
    :param ignore: list of keys to ignore in comparison
    """
    stop_flag = False
    for key in params_old:
        if key not in ignore:
            if key not in params_new:
                logger.info(
                    'New config does not contain ' + key)
                stop_flag = True
            elif params_new[key] != params_old[key]:
                logger.info('New model has ' + key + ': ' +
                            str(params_new[key]) + ' but previous model has ' + key + ': ' + str(params_old[key]))
                stop_flag = True
    for key in params_new:
        if (key not in params_old) and (key not in ignore):
            logger.info('Previous config does not contain ' + key)
            stop_flag = True
    if stop_flag == True:
        raise Exception('Model parameters not equal, can not resume training. ')
    else:
        logger.info(
            'Previously trained config and new config are compatible. ')
        return