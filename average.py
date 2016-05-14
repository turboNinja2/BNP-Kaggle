import pandas as pd
from os import listdir
from os.path import isfile, join
import sys
import time

mypath = './staged_submissions/'
arg_parser_index = 0

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-folder':
        mypath = sys.argv[arg_parser_index+1]
    arg_parser_index+=1

onlyfiles = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]

file = onlyfiles.pop(0)
print(file)
sum = pd.read_csv(file)

for currentFile in onlyfiles:
    data = pd.read_csv(currentFile)
    sum += data

submission = sum / (len(onlyfiles)+1)

print((len(onlyfiles)+1))

time.sleep(0.5)
    
submission[["ID"]] = data[["ID"]]
submission.to_csv(join(mypath,'avg.csv'),index=False)

time.sleep(0.5)
    