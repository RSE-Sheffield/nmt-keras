import codecs
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


def changes2dict(changes_list):
    import ast
    if changes_list:
        changes_dict = {}
        try:
            for arg in changes_list:
                try:
                    k, v = arg.split('=')
                except ValueError:
                    print('Overwriting arguments must have the form key=value.\n This one is: %s' % str(changes_dict))
                    exit(1)
                if '_' in v:
                    changes_dict[k] = v
                else:
                    try:
                        changes_dict[k] = ast.literal_eval(v)
                    except ValueError:
                        changes_dict[k] = v
        except ValueError:
            print("Error processing arguments: {!r}".format(arg))
            exit(2)
        return changes_dict
    else:
        return {}


def setparameters(user_config_path, default_config_path='configs/default-config-BiRNN.yml'):
    """
    This function sets the overall parameters to used by the current running instance of dq.
    """
    try:
        import yaml

        with codecs.open(default_config_path, 'r', encoding='utf-8') as fh_default:
            parameters = yaml.load(fh_default, Loader=yaml.FullLoader)

        if user_config_path.endswith('.yml'):
            with codecs.open(user_config_path, 'r', encoding='utf-8') as fh_user:
                user_parameters = yaml.load(fh_user, Loader=yaml.FullLoader)
            parameters.update(user_parameters)
            del user_parameters

        elif user_config_path.endswith('.pkl'):
            from keras_wrapper.extra.read_write import pkl2dict
            parameters = update_parameters(parameters, pkl2dict(user_config_path))

    except Exception as exception:
        logger.exception("Exception occured: {}".format(exception))

    return parameters

