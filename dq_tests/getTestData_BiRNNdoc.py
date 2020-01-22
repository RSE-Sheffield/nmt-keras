import os
import tarfile
import urllib.request

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
