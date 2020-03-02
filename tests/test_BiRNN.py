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

    import csv
    with open('tests/testVals.csv') as testValsFile:
        reader = csv.reader(testValsFile)
        next(reader) # skip header line
        for row in reader:
            if (row[0] == backend) and (row[1] == level) and (row[2] == mode) and (row[3] == task_name) and (row[4] == metric):
                value = float(row[5])
    return value

def getOutputVal(filepath, metric):

    import csv
    with open(filepath) as file: # trained_models/${task_name}_srcmt_${model_type}/val.qe_metrics
        reader = csv.reader(file)
        value = 0.
        row = next(reader)
        col = row.index(metric) # in header find column corresponding to metric
        for row in reader:
            # grab the highest value in the column
            if float(row[col]) > value:
                value = float(row[col])
    return value

def test_BiRNN_word_train():
    import os
    import keras
    import pytest
    import tests.getTestData as getTestData
    import deepquest as dq

    # check for required environment variables
    assert os.environ['TEST_LEVEL'] is not None

    if os.environ['TEST_LEVEL'] == 'word':
        backend = keras.backend.backend()
        task_name = 'testData-word'
        metric = 'f1_prod'
        model_type = 'EncWord'

        getTestData.BiRNN_word()
        testVal = getTestVal(backend, 'word', 'train', task_name, metric)

        # set the seed
        import numpy.random
        numpy.random.seed(1)
        import random
        random.seed(1)

        with pytest.raises(SystemExit):
            dq.train('tests/config-word-BiRNN.yml')
        result = getOutputVal('trained_models/' + task_name + '_srcmt_' + model_type + '/val.qe_metrics', metric)

    assert (result - testVal)**2 < 1E-12

    # elif os.environ["TEST_LEVEL"] == 'sentence':
    #     getTestData.BiRNN_sent()
    # elif os.environ["TEST_LEVEL"] == 'document':
    #     getTestData.BiRNN_doc()
