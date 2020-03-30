import keras
import numpy.random
import pytest
import random

import deepquest as dq
from tests.utils import *

def test_BiRNN_document():

	backend = keras.backend.backend()

	level = 'document'
	task_name = 'testData-doc'
	metric = 'pearson'
	model_type = 'EncSent'  # FIXME after merge with EncDoc branch
	getTestData_BiRNN_doc()


	# TRAINING TEST

	testVal = getTestVal(backend, level, 'train', task_name, metric)

	# set the seed
	numpy.random.seed(1)
	random.seed(1)

	model_name = task_name + '_srcmt_' + model_type

	# this is a problem, keras_wrapper's evaluate callback exits by using exit(1) which raises SystemExit
	with pytest.raises(SystemExit):
		dq.train('tests/config-' + level + '-BiRNN.yml')
	result, epoch = getOutputVal('trained_models/' + model_name + '/val.qe_metrics', metric)

	assert (result - testVal)**2 < 1E-10 # check that output is within 1E-5 of the expected result



	# PREDICTION TEST

	model = 'trained_models/' + model_name +'/epoch_' + epoch + '.h5'
	dataset = 'trained_models/' + model_name + '/Dataset_' + task_name + '_srcmt.pkl'
	save_path = 'saved_predictions/prediction_' + task_name + '/'
	evalset = 'test'
	dq.predict(model=model, dataset=dataset, save_path=save_path, evalset=evalset)
	result = getPredictedVal(save_path + 'test.qe_metrics', metric)
	testVal = getTestVal(backend, level, 'predict', task_name, metric)

	assert (result - testVal)**2 < 1E-10 # check that output is within 1E-5 of the expected result

if __name__ == '__main__':
	test_BiRNN_document()