import numpy as np
from sklearn import svm
import operator

deltaForGaussian = [0, 0.5, 1, 1.5]
deltaForOthers = [3,3.5,4,4.5]
new_data = "/home/paruru/graduate/new_data"
#new_data = "."
M = 16
ETA = 1.0e-2
SETA = 1.0e-4
EPSILON = 0.001
almostZero = 1.0e-4
I = np.ones(M*M)
I = I.reshape(M,M)
dec_func = list()
'''
def get_x(prob):
    result = []
        for sparse_sv in prob.x[:prob.l]:
			row = dict()
			i = 0
			while True:
				row[sparse_sv[i].index] = sparse_sv[i].value
				if sparse_sv[i].index == -1:
					break
				i += 1
			result.append(row)
	return result
'''
def file_output(filename, listname) :
    fileout = open(new_data+filename,'w')
    for line in listname :
        for num in line :
            fileout.write(str(num) + " ")
        fileout.write("\n")
    fileout.close()


def Gaussian_kernel(dis,gamma,deltaIndex) :
    gamma = (1.2**deltaForGaussian[deltaIndex])*gamma
    return np.exp(-1.0*gamma*dis*dis)

def Laplacian_kernel(dis,gamma,deltaIndex) :
    gamma = (1.2**deltaForOthers[deltaIndex])*gamma
    return np.exp(-1.0*np.sqrt(gamma)*dis)
    
def Inverse_square_distance_kernel(dis,gamma,deltaIndex) :
    gamma = (1.2**deltaForOthers[deltaIndex])*gamma
    divisor = gamma*dis*dis+1
    return 1.0/divisor

def Inverse_distance_kernel(dis,gamma,deltaIndex) : 
    gamma = 1.2**deltaForOthers[deltaIndex]*gamma
    divisor = np.sqrt(gamma)*dis+1
    return 1.0/divisor

theD = None
Dlist = list()

def Kernel(dis,m) :
    if theD[m] <= 0 :
        theD[m] = almostZero
    if m<M/4 and m >=0  :
        return Gaussian_kernel(dis,1/theD[m],m)
    if m>=M/4 and m <M/2 :
        return Laplacian_kernel(dis,1/theD[m],m-M/4)
    if m>=M/2 and m<M/4*3 :
        return Inverse_distance_kernel(dis,1/theD[m],m-M/2)
    if m>=M/4*3 and m<M :
        return Inverse_square_distance_kernel(dis,1/theD[m],m-M/4*3)
    return 0

def Mytolist(x) :
    return x.tolist()

def GetKernelMat(X,m):
    n = X.shape[0]
    l = X.shape[1]
    X_mat = np.zeros(n*n*l)
    X_mat.shape = n,n,l
    for i in range(n):
        X_mat[i,:] = X

    dis_mat = np.zeros(n*n)
    dis_mat.shape = n,n
    for i in range(l) :
        X_mat[:,:,i] = X_mat[:,:,i] - X_mat[:,:,i].transpose()
        dis_mat = dis_mat + X_mat[:,:,i] * X_mat[:,:,i] 

    dis_mat = np.sqrt(dis_mat)
    kernel_ufunc = np.frompyfunc(Kernel, 2, 1)
    return kernel_ufunc(dis_mat,m)

def GetKernelMat2(tes,tra,m):
    n = tes.shape[0]
    t = tra.shape[0]
    l = tes.shape[1]
    if tes.shape[1] != tra.shape[1] :
        print("EORROE GETK2")
        return 0
    
    X_mat = np.zeros(n*t*l)
    X_mat.shape = n,t,l
    Y_mat = np.zeros(n*t*l)
    Y_mat.shape = n,t,l
    for i in range(n):
        X_mat[i,:] = tra
    for i in range(t):
        Y_mat[:,i,:] = tes

    dis_mat = np.zeros(n*t)
    dis_mat.shape = n,t
    for i in range(l) :
        X_mat[:,:,i] = X_mat[:,:,i] - Y_mat[:,:,i]
        dis_mat = dis_mat + X_mat[:,:,i] * X_mat[:,:,i] 
    
    dis_mat = np.sqrt(dis_mat)
    kernel_ufunc = np.frompyfunc(Kernel, 2, 1)
    result = kernel_ufunc(dis_mat,m)
    float_ufunc = np.frompyfunc(float,1,1)
    return float_ufunc(result)

'''
def GetInv(x) :
    if x.shape[0] != x.shape[1] :
        print("ERROR")
        return None
    xinv = np.zeros(x.shape)
    for i in range (xinv.shape[0]) :
        for j in range (xinv.shape[1]) :
            sign = 1
            if (i+j)%2 == 0 :
                sign = 1
            else :
                sign = -1
            b = x[:,:]
            b = np.delete(b,i,axis=0)
            b = np.delete(b,j,axis=1)
            bbb = np.linalg.det(b)
            xinv[j,i] = sign*bbb
    ii = np.linalg.det(x)
    xii = xinv / ii
    return xii
'''
def block_matrix(tes,tra,m) :
    n = tes.shape[0]
    t = tra.shape[0]
    l = tes.shape[1]
    if n < 30 and t < 30 :
        return GetKernelMat2(tes,tra,m)
    x1 = np.concatenate((block_matrix(tes[:n/2],tra[:t/2],m),block_matrix(tes[n/2:],tra[:t/2],m)),axis=0)
    x2 = np.concatenate((block_matrix(tes[:n/2],tra[t/2:],m),block_matrix(tes[n/2:],tra[t/2:],m)),axis=0)
    return (np.concatenate((x1,x2),axis=1))


theY = None
X_labeled = None #the x is the value which will be computed by kernels
YX_labeled = list()
X_target = list()

labeded_data_file = open(new_data + "/iris_labeled",'r')
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

target_data_file = open(new_data + "/iris_labeled_test",'r')
for line in target_data_file :
    li = line.split()
    if li[len(li)-1] == '\n':
        del li[len(li)-1]
    arrr = np.array(li,dtype='float64')
    X_target.append(arrr[1:])
X_target = np.array(X_target)

labeded_data_file.close()
target_data_file.close()

n_t = len(X_target)
print("read file OK")
X_all = np.concatenate((X_labeled,X_target),axis=0)

s_vec = np.zeros(n_a +n_t)
s_vec.shape = 1,n_a+n_t
for i in range(0, n_a) :
    s_vec[0,i] = 1.0/n_a
for i in range(n_a, n_a+n_t) :
    s_vec[0,i] = -1.0/n_t

S = np.dot(s_vec.transpose(),s_vec)
print("S ok")
class_num = 1
begline = 0
endline = 0
comXtar = np.zeros(n_t*n_a)
comXtar.shape = n_t,n_a

while(endline < n_a) :
    while(theY[endline] == class_num) :
        endline = endline + 1
        if(endline >= n_a) :
            break
    #beginning of a classifer
    newY = -1 * np.ones(n_a)
    if(endline <= n_a) :
        newY[begline:endline] *= -1
    #fengp theD
    theD = np.ones(M)
    theD = 1.0/M * theD
    Dlist.append(theD)
    #begin t times for circle
    clf = None
    for t in range(0,2) :
        P = np.zeros(M)
        computed_X_labeled = np.zeros(n_a*n_a)
        computed_X_labeled.shape = n_a,n_a
        for m in range(0,M) :
            Km = block_matrix(X_all,X_all,m)
            computed_X_labeled = computed_X_labeled + theD[m]*Km[:n_a,:n_a]
            P[m] = np.trace(np.dot(Km,S))
        P.shape = 1,M
        PPt = np.dot(P.transpose(),P)
        print("PPT ok comlab OK")
        tempMat = (PPt + EPSILON*I)
        Dlist[len(Dlist)-1] = theD
        clf = svm.SVC(kernel='precomputed')
        clf.fit(computed_X_labeled.tolist(),newY.tolist())
        alphasY = clf.dual_coef_[0]
        gradJ = np.zeros(M)
        nSv = len(clf.support_)
        indices = clf.support_
        SV = np.zeros(nSv*X_labeled.shape[1])
        SV.shape = nSv,X_labeled.shape[1]
        for i in range(nSv) :
            SV[i] = X_labeled[indices[i]]
        SVKM = None
        for m in range(0,M) :
            SVKM = block_matrix(SV,SV,m)
            for i in range(0,nSv) :
                for j in range(0,nSv) :
                    gradJ[m] = gradJ[m] + alphasY[i]*alphasY[j]*SVKM[i,j]
        gradJ = -0.5 * gradJ
        gradJ.shape = 1,M
        print("gradJ ok")
        g = np.zeros(M)
        theD.shape = M,1
        g.shape = M,1
        if tempMat.shape[0] == tempMat.shape[1] :
            if np.linalg.det(tempMat) != 0 :
                print("DET !=0")
                tempMat = np.linalg.pinv(tempMat)
                aaa = SETA*np.dot(tempMat,gradJ.transpose())
                g = theD + aaa
            else :
                print("DET == 0")
                break
        else:
            print("ERROR")
        theD = theD - ETA * g
        theD.shape = (M,)
        for i in range(len(theD)) :
            if theD[i] < 0 :
                theD[i] = almostZero
        su = np.sum(theD)
        theD = theD / su

    dec_func.append(clf.decision_function)
    class_num = class_num+1
    begline = endline

###predict


prediclist = np.zeros(n_t*len(dec_func))
prediclist.shape = n_t, len(dec_func)
for i in range(0,len(dec_func)) :
    comXtar = np.zeros(n_t*n_a)
    comXtar.shape = n_t,n_a
    dd = Dlist[i]
    for m in range(M) :
        cotes = block_matrix(X_target,X_labeled,m)
        comXtar = comXtar + dd[m] * cotes
    dec = dec_func[i](comXtar.tolist())
    prediclist[:,i] = np.array(dec)
print("predict ok")
class Yitem :
    def __init__(self) :
        self.y = 0
        self.ratio = 0

fileo = open(new_data+"/yyyyy",'w')
for i in range(prediclist.shape[0]) :
    for j in range(prediclist.shape[1]) :
        if(prediclist[i,j] > 0) :
            fileo.write(str(j+1)+":"+str(prediclist[i,j])+" ")
    fileo.write("\n")
fileo.close()
