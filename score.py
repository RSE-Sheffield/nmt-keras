# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import scipy.stats
import sklearn.metrics
from math import sqrt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Compute a score for a set of predictions vs reference set. ")
    parser.add_argument("files", nargs=2, help="Two text files containing predictions and references. ")
    return parser.parse_args()

def read_file_to_list(file_in,logger):
    logger.info('Reading %s ', file_in)
    data_list = []
    with open(file_in) as file:
        for line in file.readlines():
            data_list.append(float(line))
    return data_list

def main():
    args = parse_args()
    assert len(args.files) == 2, "Number of files specified must equal 2. "

    data = []
    for i,path in enumerate(args.files,1):
        a = read_file_to_list(path,logger)
        data.append(a)

    assert len(data[0]) == len(data[1]), "Files must be of equal length. "

    pcc, _ = scipy.stats.pearsonr(data[0],data[1])
    mae = sklearn.metrics.mean_absolute_error(data[0],data[1])
    rmse = sqrt(sklearn.metrics.mean_squared_error(data[0],data[1]))

    print('Mean absolute error: %.3f' % mae)
    print('Pearsons correlation: %.3f' % pcc)
    print('RMSE: %.3f' % rmse)

if __name__ == "__main__":
    sys.exit(main())
