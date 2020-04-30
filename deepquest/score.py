# -*- coding: utf-8 -*-
import logging
import sys
import scipy.stats
import sklearn.metrics
from math import sqrt
from deepquest.utils.logs import logger_setup

logger, logging = logger_setup('score')


def read_file_to_list(file_in, logger):
    logger.info('Reading %s ', file_in)
    data_list = []
    with open(file_in) as inp:
        data_list = list(map(float, inp))
    return data_list


def main(files):
    """
    Evaluate a set of predictions with regard to a reference set.
    :param files: List of paths to two text files containing predictions and references.
    """
    assert len(files) == 2, "Number of files specified must equal 2. "

    data = []
    for path in files:
        a = read_file_to_list(path, logger)
        data.append(a)

    assert len(data[0]) == len(data[1]), "Files must be of equal length. "

    pcc, _ = scipy.stats.pearsonr(data[0], data[1])
    mae = sklearn.metrics.mean_absolute_error(data[0], data[1])
    rmse = sqrt(sklearn.metrics.mean_squared_error(data[0], data[1]))

    print('Mean absolute error: %.3f' % mae)
    print('Pearsons correlation: %.3f' % pcc)
    print('RMSE: %.3f' % rmse)
