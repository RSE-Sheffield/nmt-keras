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

def processDocData(cachePath,extractDir,dataDir,numDocs,split):
    src_file_out = dataDir + split + '.src'
    with open(src_file_out, "w") as writefile:
        writefile.write("")
    trg_file_out = dataDir + split + '.mt'
    with open(trg_file_out, "w") as writefile:
        writefile.write("")
    if split != 'test':
        score_file_out = dataDir + split + '.mqm'
        with open(score_file_out, "w") as writefile:
            writefile.write("")

    i = 0
    while i < numDocs:
        docDir = cachePath + extractDir + '/doc' + '{:0>4}'.format(i)

        file_in = docDir + '/' + 'source.segments'
        with open(file_in) as file:
            lines = file.read()
        with open(src_file_out, "a") as writefile:
            print('Writing from ' + file_in + ' to ' + src_file_out)
            writefile.write(lines)
            writefile.write("#doc#\n")

        file_in = docDir + '/' + 'mt.segments'
        with open(file_in) as file:
            lines = file.read()
        with open(trg_file_out, "a") as writefile:
            print('Writing from ' + file_in + ' to ' + trg_file_out)
            writefile.write(lines)
            writefile.write("#doc#\n")

        if split != 'test':
            file_in = docDir + '/' + 'document_mqm'
            with open(file_in) as file:
                lines = file.read()
            with open(score_file_out, "a") as writefile:
                print('Writing from ' + file_in + ' to ' + score_file_out)
                writefile.write(lines)

        i += 1

baseCacheDir = 'cache/'
task = 'testData-doc/'
rawtask = 'raw-'+task

traindev = 'https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/doc_level_training.tar.gz'
test = 'https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/doc_level_test.tar.gz'

os.makedirs(baseCacheDir, exist_ok=True)
cachePath = baseCacheDir + rawtask
os.makedirs(cachePath, exist_ok=True)

downloadAndExtractFiles(cachePath,traindev,test)

for file in os.listdir(cachePath):
    if file.endswith(".tar.gz"):
        tar = tarfile.open(cachePath+file, "r:gz")
        print('Extracting: ' + file + ' to ' + cachePath)
        tar.extractall(path=cachePath)
        tar.close()
        # os.remove(file)
        # print('Deleting: ' + file)

dataDir = 'examples/' + task
os.makedirs(dataDir, exist_ok=True)
processDocData(cachePath,'task4_en-fr_training',dataDir,5,'train')
processDocData(cachePath,'task4_en-fr_dev',dataDir,5,'dev')
processDocData(cachePath,'doc_level_test/task4_en-fr_test',dataDir,5,'test')
