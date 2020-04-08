import csv
import os
import tarfile
import urllib.request

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


def getTestData_BiRNN_sent():
    # Download and extract the files
    def downloadAndExtractFiles(cachePath,*args):
        for url in args:
            fileName = os.path.basename(os.path.normpath(url))
            checkPath = cachePath +  fileName
            if os.path.exists(checkPath):
                print(checkPath + ' already downloaded')
            else :
                print('\n Downloading ' + fileName + ' from ' + 'url')
                download_to = os.path.join(cachePath, fileName)
                with urllib.request.urlopen(url) as f:
                    with open(download_to, "wb") as out:
                        out.write(f.read())
        print('\n')

    baseCacheDir = 'cache/'
    task = 'testData-sent/'
    rawtask = 'raw-'+task

    train = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_training.tar.gz'
    dev = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_dev.tar.gz'
    test = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_test.tar.gz'
    label = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/gold/Task1_gold.tar.gz'

    os.makedirs(baseCacheDir, exist_ok=True)
    cachePath = baseCacheDir + rawtask
    os.makedirs(cachePath, exist_ok=True)

    downloadAndExtractFiles(cachePath,train,dev,test,label)

    for file in os.listdir(cachePath):
        if file.endswith(".tar.gz"):
            tar = tarfile.open(cachePath+file, "r:gz")
            print('Extracting: ' + file + ' to ' + cachePath)
            tar.extractall(path=cachePath)
            tar.close()

    # make the test data
    dataDir = 'data/'+task
    os.makedirs(dataDir, exist_ok=True)

    totalLines = 500 # total number of lines to take from example data
    for f in os.listdir( cachePath ):
        if f.endswith(".hter") or f.endswith(".pe") or f.endswith(".source") or f.endswith(".target"):
            file_in = cachePath+f
            file_out = dataDir+f
            print('Copying first ' + str(totalLines) + ' lines of ' + file_in + ' to ' + file_out)
            with open(file_in) as file:
                lines = file.readlines()
                lines = [l for i, l in enumerate(lines) if i < totalLines]
                with open(file_out, "w") as f1:
                    f1.writelines(lines)

        if f.endswith(".source"):
            filename, file_extension = os.path.splitext(dataDir + f)
            print('Renaming ' + f + ' to ' + filename + '.src')
            os.rename(dataDir + f, filename + '.src')
        elif f.endswith(".target"):
            filename, file_extension = os.path.splitext(dataDir + f)
            print('Renaming ' + f + ' to ' + filename + '.target')
            os.rename(dataDir + f, filename + '.mt')

def getTestData_BiRNN_word():
    # Download and extract the files
    def downloadAndExtractFiles(cachePath,*args):
        for url in args:
            fileName = os.path.basename(os.path.normpath(url))
            checkPath = cachePath +  fileName
            if os.path.exists(checkPath):
                print(checkPath + ' already downloaded')
            else :
                print('\n Downloading ' + fileName + ' from ' + 'url')
                download_to = os.path.join(cachePath, fileName)
                with urllib.request.urlopen(url) as f:
                    with open(download_to, "wb") as out:
                        out.write(f.read())
        print('\n')

    baseCacheDir = 'cache/'
    task = 'testData-word/'
    rawtask = 'raw-'+task

    train = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task2_en-es_training.tar.gz'
    dev = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task2_en-es_dev.tar.gz'
    test = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task2_en-es_test.tar.gz'
    label = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/gold/Task2_gold.tar.gz'

    os.makedirs(baseCacheDir, exist_ok=True)
    cachePath = baseCacheDir + rawtask
    os.makedirs(cachePath, exist_ok=True)

    downloadAndExtractFiles(cachePath,train,dev,test,label)

    for file in os.listdir(cachePath):
        if file.endswith(".tar.gz"):
            tar = tarfile.open(cachePath+file, "r:gz")
            print('Extracting: ' + file + ' to ' + cachePath)
            tar.extractall(path=cachePath)
            tar.close()

    ## make the test data
    dataDir = 'data/'+task
    os.makedirs(dataDir, exist_ok=True)

    totalLines = 500 # total number of lines to take from example data
    for f in os.listdir( cachePath ):
        if f.endswith(".tags") or f.endswith(".pe") or f.endswith(".source") or f.endswith(".target"):
            file_in = cachePath+f
            file_out = dataDir+f
            print('Copying first ' + str(totalLines) + ' lines of ' + file_in + ' to ' + file_out)
            with open(file_in) as file:
                lines = file.readlines()
                lines = [l for i, l in enumerate(lines) if i < totalLines]
                with open(file_out, "w") as f1:
                    f1.writelines(lines)

        if f.endswith(".source"):
            filename, file_extension = os.path.splitext(dataDir + f)
            print('Renaming ' + f + ' to ' + filename + '.src')
            os.rename(dataDir + f, filename + '.src')
        elif f.endswith(".target"):
            filename, file_extension = os.path.splitext(dataDir + f)
            print('Renaming ' + f + ' to ' + filename + '.target')
            os.rename(dataDir + f, filename + '.mt')

def getTestData_BiRNN_doc():
    # Download and extract the files
    def downloadAndExtractFiles(cachePath,*args):
        for url in args:
            fileName = os.path.basename(os.path.normpath(url))
            checkPath = cachePath +  fileName
            if os.path.exists(checkPath):
                print(checkPath + ' already downloaded')
            else :
                print('\n Downloading ' + fileName + ' from ' + 'url')
                download_to = os.path.join(cachePath, fileName)
                with urllib.request.urlopen(url) as f:
                    with open(download_to, "wb") as out:
                        out.write(f.read())

    def processDocs(split,numDocs,numDocsStart):
        src_file_out = dataDir + split + '.src'
        with open(src_file_out, "w") as writefile:
            writefile.write("")
        trg_file_out = dataDir + split + '.mt'
        with open(trg_file_out, "w") as writefile:
            writefile.write("")

        file_in = baseCacheDir + rawtask + 'source_doc-level.training'
        with open(file_in) as file:
            lines = file.readlines()
            for i, l in enumerate(lines):
                if numDocsStart <= i < (numDocs + numDocsStart):
                    l = baseCacheDir + rawtask + l.rstrip()
                    with open(l,"r") as doc:
                        docText = doc.read()
                    with open(src_file_out, "a") as writefile:
                        writefile.write(docText.rstrip())
                        writefile.write("\n#doc#\n")

        file_in = baseCacheDir + rawtask + 'target_doc-level.training'
        with open(file_in) as file:
            lines = file.readlines()
            for i, l in enumerate(lines):
                if numDocsStart <= i < (numDocs + numDocsStart):
                    l = baseCacheDir + rawtask + l.rstrip()
                    with open(l,"r") as doc:
                        docText = doc.read()
                    with open(trg_file_out, "a") as writefile:
                        writefile.write(docText.rstrip())
                        writefile.write("\n#doc#\n")

    baseCacheDir = 'cache/'
    task = 'testData-doc/'
    rawtask = 'raw-'+task

    traindev = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_qe/task3_en-es_training.tar.gz'
    test = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_qe/task3_en-es_test.tar.gz'
    labels = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_qe/wmt16_task3_gold.tar.gz'

    os.makedirs(baseCacheDir, exist_ok=True)
    cachePath = baseCacheDir + rawtask
    os.makedirs(cachePath, exist_ok=True)

    downloadAndExtractFiles(cachePath,traindev,test,labels)

    for file in os.listdir(cachePath):
        if file.endswith(".tar.gz"):
            tar = tarfile.open(cachePath+file, "r:gz")
            print('Extracting: ' + file + ' to ' + cachePath)
            tar.extractall(path=cachePath)
            tar.close()
            # os.remove(file)
            # print('Deleting: ' + file)

    dataDir = 'data/' + task
    os.makedirs(dataDir, exist_ok=True)

    trainSize = 100
    devSize = 46
    testSize = 62

    # split the training labels into a training split and dev split (just the labels)
    file_in = baseCacheDir + rawtask + '/' + 'labels_doc-level.training'
    file_out = dataDir + 'train.hter'
    with open(file_in) as file:
        lines = file.readlines()
        lines = [l for i, l in enumerate(lines) if i < trainSize]
        with open(file_out, "w") as f1:
            f1.writelines(lines)
    file_out = dataDir + 'dev.hter'
    with open(file_in) as file:
        lines = file.readlines()
        lines = [l for i, l in enumerate(lines) if trainSize <= i < (trainSize + devSize)]
        with open(file_out, "w") as f1:
            f1.writelines(lines)

    # rename the test labels file
    file_in = baseCacheDir + rawtask + 'labels_doc-level.test'
    file_out = dataDir + 'test.hter'
    with open(file_in) as file:
        lines = file.readlines()
        lines = [l for i, l in enumerate(lines) if i < testSize]
        with open(file_out, "w") as f1:
            f1.writelines(lines)

    processDocs('train',trainSize,0)
    processDocs('dev',devSize,trainSize)
    processDocs('test',testSize,0)
