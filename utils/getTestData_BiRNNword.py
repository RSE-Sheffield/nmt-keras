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
dataDir = 'examples/'+task
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
