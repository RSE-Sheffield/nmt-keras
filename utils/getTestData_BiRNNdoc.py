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
task = 'testData-doc/'
rawtask = 'raw-'+task

train = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task3_en-de_training.tar.gz'
dev = ''
test = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task3_en-de_test.tar.gz'
label = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/gold/Task3_gold.tar.gz'

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
exampleDir = 'examples/'+task
os.makedirs(exampleDir, exist_ok=True)

totalLines = 800 # total number of lines to take from example data
for f in os.listdir( cachePath ):
    if f.endswith(".training") or f.endswith(".test") or (f.endswith(".meteor") and not f.startswith("de-en")):
        file_in = cachePath+f
        file_out = exampleDir+f
        print('Copying first ' + str(totalLines) + ' lines of ' + file_in + ' to ' + file_out)
        with open(file_in) as file:
            lines = file.readlines()
            lines = [l for i, l in enumerate(lines) if i < totalLines]
            with open(file_out, "w") as f1:
                f1.writelines(lines)

    if "source" in f and f.endswith(".training"):
        print('Renaming ' + f + ' to train.src')
        os.rename(exampleDir + f, exampleDir + 'train.src')
    elif "source" in f and f.endswith(".test"):
        print('Renaming ' + f + ' to test.src')
        os.rename(exampleDir + f, exampleDir + 'test.src')
    elif "target" in f and f.endswith(".training"):
        print('Renaming ' + f + ' to training.mt')
        os.rename(exampleDir + f, exampleDir + 'training.mt')
    elif "target" in f and f.endswith(".test"):
        print('Renaming ' + f + ' to test.mt')
        os.rename(exampleDir + f, exampleDir + 'test.mt')
    elif "training" in f and f.endswith(".meteor") and not "de-en" in f:
        print('Renaming ' + f + ' to training.meteor')
        os.rename(exampleDir + f, exampleDir + 'training.meteor')
    elif "test" in f and f.endswith(".meteor") and not "de-en" in f:
        print('Renaming ' + f + ' to test.meteor')
        os.rename(exampleDir + f, exampleDir + 'test.meteor')
