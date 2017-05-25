import numpy as np
from sklearn import svm
import operator
import time

new_data = "/home/paruru/graduate/new_data"
#new_data = "."

theY = None
X_labeled = None #the x is the value which will be computed by kernels
YX_labeled = list()
X_target = list()

def file_output(filename, listname) :
    fileout = open(new_data+filename,'w')
    for line in listname :
        for num in line :
            fileout.write(str(num) + " ")
        fileout.write("\n")
    fileout.close()

labeded_data_file = open(new_data + "/wine_labeled",'r')
for line in labeded_data_file :
    li = line.split()
    if li[len(li)-1] == '\n':
        del li[len(li)-1]
    arrr = map(float,li)
    YX_labeled.append(arrr)

YX_labeled.sort(key = operator.itemgetter(0))
YX_labeled = np.array(YX_labeled,dtype=float)
X_labeled = YX_labeled[:,1:]

n_a = len(X_labeled)

theY = YX_labeled[:,0]
theY.shape = (n_a,)

target_data_file = open(new_data + "/wine_test",'r')
for line in target_data_file :
    li = line.split()
    if len(li) >= 1 and li[len(li)-1] == '\n':
        del li[len(li)-1]
    arrr = np.array(li,dtype='float64')
    X_target.append(arrr[1:])
X_target = np.array(X_target)

labeded_data_file.close()
target_data_file.close()

n_t = len(X_target)
print("read file OK")
X_all = np.concatenate((X_labeled,X_target),axis=0)

clf = svm.SVC()
clf.fit(X_labeled.tolist(),theY.tolist())
yy = clf.predict(X_target.tolist())
file_out = open(new_data+"/svmY_"+time.strftime("%H:%M:%S"),'w')
for y in yy :
    file_out.write(str(y))
    file_out.write("\n")
