import csv
import os

import keras
import numpy.random
import pytest
import random

import deepquest as dq
import tests.getTestData as getTestData


def getTestVal(backend, level, mode, task_name, metric):
	"""Gets the relevant value from the testVal.csv file to run a test against

	Parameters:
	backend (str): Keras backend eg tensorflow, or , theano
	level (str): QE level under test, eg 'word', 'sentence', 'document'
	task_name (str): Name of dataset, eg 'testData-word'
	metric (str): test metric, eg 'pearson' or 'f1_prod'

	Returns:
	float: Value to run test against
	"""

	with open('tests/testVals.csv') as testValsFile:
		reader = csv.reader(testValsFile)
		next(reader)  # skip header line
		for row in reader:
			if (row[0] == backend) and (row[1] == level) and (row[2] == mode) and (row[3] == task_name) and (row[4] == metric):
				value = float(row[5])
	return value


def getOutputVal(filepath, metric):
	"""Gets the relevant value from the trained model output

	Parameters:
	filepath (str): path to the model output file
	metric (str): test metric, eg 'pearson' or 'f1_prod'

	Returns:
	float: Value to run test against
	"""
	with open(filepath) as file:
		reader = csv.reader(file)
		value = 0.0
		row = next(reader)
		col = row.index(metric)  # in header find column corresponding to metric
		for row in reader:
			# grab the highest value in the column
			if float(row[col]) > value:
				value = float(row[col])
				epoch = row[0]
	return value, epoch


def getPredictedVal(filepath, metric):
	"""Gets the relevant value from the trained model output

	Parameters:
	filepath (str): path to the model output file
	metric (str): test metric, eg 'pearson' or 'f1_prod'

	Returns:
	float: Value to run test against
	"""

	with open(filepath) as file:
		reader = csv.reader(file)
		rownum = 0
		row = next(reader)
		col = row.index(metric)  # in header find column corresponding to metric
		for row in reader:
			# grab the latest value in the column
			if reader.line_num > rownum:
				value = row[col]
				rownum = reader.line_num
	return float(value)


def test_BiRNN():

	# check for required environment variables
	assert os.environ['TEST_LEVEL'] is not None

	backend = keras.backend.backend()

	if os.environ['TEST_LEVEL'] == 'word':
		level = os.environ["TEST_LEVEL"]
		task_name = 'testData-word'
		metric = 'f1_prod'
		model_type = 'EncWord'
		getTestData.BiRNN_word()
	elif os.environ["TEST_LEVEL"] == 'sentence':
		level = os.environ["TEST_LEVEL"]
		task_name = 'testData-sent'
		metric = 'pearson'
		model_type = 'EncSent'
		getTestData.BiRNN_sent()
	elif os.environ["TEST_LEVEL"] == 'document':
		level = os.environ["TEST_LEVEL"]
		task_name = 'testData-doc'
		metric = 'pearson'
		model_type = 'EncSent'  # FIXME after merge with EncDoc branch
		getTestData.BiRNN_doc()



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

	assert (result - testVal)**2 < 1E-10 # check that output is within 1E-6 of the expected result



	# PREDICTION TEST

	model = 'trained_models/' + model_name +'/epoch_' + epoch + '.h5'
	dataset = 'trained_models/' + model_name + '/Dataset_' + task_name + '_srcmt.pkl'
	save_path = 'saved_predictions/prediction_' + task_name + '/'
	evalset = 'test'
	# with pytest.raises(SystemExit):
	dq.predict(model=model, dataset=dataset, save_path=save_path, evalset=evalset)
	result = getPredictedVal(save_path + 'test.qe_metrics', metric)
	testVal = getTestVal(backend, level, 'predict', task_name, metric)

	assert (result - testVal)**2 < 1E-10 # check that output is within 1E-6 of the expected result
