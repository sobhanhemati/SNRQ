import numpy as np
from numpy.linalg import svd, norm
from scipy.stats import ortho_group
from scipy.optimize import minimize
from scipy.optimize import Bounds
import warnings
warnings.filterwarnings("ignore")
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy.linalg import eig
import scipy.io as sio
from ITQ import ITQ
from Datasets import load_dataset
from evaluate import precision_recall, precision_radius, mAP, Macro_AP, return_all_metrics
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler

method_name='SNRQ'
path = 'mnist_gist512' # folder containing dataset
dataset_name = 'labelme_vggfc7' #options: mnist_gist512, cifar10_vggfc7, labelme_vggfc7, nuswide_vgg, colorectal_efficientnet 
K = 16 # number of bits
alpha=3 # Control quantization power 
beta=.01 # Control the trade-off between corrupting neighbourhood structure of data and of minimizing quantization loss.n_iter=70 #number of iterations
analytic_derivatve=True  #if False, the derivative is calculated numerically uing  which is slower in L-BFGS-B optimizer. 
# if you need further training speed, implement the derivative using Tensorflow and use GPU
n_iter=70 #number of iterations

# pathfor saving results
result_path = os.path.join('results', dataset_name, method_name 
                           + '_' + str(K)+ '.npz')
if not os.path.exists(os.path.join('results', dataset_name)):
    os.makedirs(os.path.join('results', dataset_name))
    
# load data
train_data, train_labels, test_data, test_labels = load_dataset(
    dataset_name, path=path, one_hot=False)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
n_train = train_data.shape[0]

# normalization
scaler = StandardScaler(with_mean=True, with_std=False)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

if dataset_name=='labelme_vggfc7' or  dataset_name=='colorectal_efficientnet':
	# dimensionality reduction
	from sklearn.decomposition import PCA
	pca = PCA(n_components=512)
	pca.fit(train_data)
	train_data=pca.transform(train_data)
	test_data=pca.transform(test_data)
X=train_data.copy()  #X d*n

#calculate Cx
Cx=X.T@X

#initialization
Q=np.zeros((Cx.shape))
u=np.zeros((1,Cx.shape[0]))
Q_loss = np.zeros((n_iter,)) # quantization loss

#Initialize W
eig_values, W = np.linalg.eigh(X.T@X)
idx = np.argsort(-eig_values)
eig_values, W = eig_values[idx], W[:,idx]
W = W[:,:K] #W  d*k

# projected data
V=X@W   #n*k

#Initialize R
#R0 = ortho_group.rvs(K)
R0, itq_Q_loss = ITQ(V.copy(), n_iter=70)

a={}
def fun(x):
    tmp=x.T@Q@x  -beta*(x.T@x)**2 +2*alpha*u@x    #q 1*d    -beta*(x.T@x)**2
    return -1*tmp

def jac(x):
    tmp=2*Q@x-beta*4*x* x.T@x  +2*alpha*u.T
#     tmp=np.squeeze(tmp)
    return -1*tmp
  
for it in range(n_iter):
    if it==0:
        VR=V@R0  

    # compute quantization loss
    if it>0:
        VR = V@R
        # note that the following is Q_loss of previous iteration
        Q_loss[it-1] = (norm(B-VR))**2

    # fix R,W and update B
    B = np.sign(VR)

    # fix B,W and update R
    S, omega, Shat_transpose = svd(B.T@V)
    Shat = Shat_transpose.T
    R = Shat@S.T
    
    #calculate Cy
    Cy=R@B.T@X
    
    # fix B,R and update W
    t_max=1
    for iteration in range(t_max):
        for i in range(K):
            idxx=np.ones(K, dtype=bool)
            idxx[i]=False
            L=W[:,idxx].copy()
            Q=(1-alpha)*Cx-(2*beta)*L@L.T+2*beta*np.eye(W.shape[0])
            u=Cy[i,:]
            if analytic_derivatve==True:
                a[i]=minimize(fun,W[:,i],method='L-BFGS-B',jac=jac)
            else:
                a[i]=minimize(fun,W[:,i],method='L-BFGS-B')
            W[:,i]=a[i].x
            
    #updated projected data
    V=X@W   #n*k 
    
    if it>0:
        print('Iteration:{}....normalized quantization loss:{}'.format(it,Q_loss[it-1]/Q_loss[0]))
    # compute the last quantization loss
    if it==(n_iter-1):
        VR = V@R
        Q_loss[it] = (norm(B-VR))**2


B = np.sign(VR)
S, omega, Shat_transpose = svd(B.T@V)
Shat = Shat_transpose.T
R = Shat@S.T


# obtining binary codes using our hash function: sign(XWR)
pca_test_data=(test_data)@W
Test_SNRQ=np.sign(pca_test_data@R)
Test_SNRQ[Test_SNRQ<0]=0


pca_train_data=(train_data)@W
Train_SNRQ=np.sign(pca_train_data@R)
Train_SNRQ[Train_SNRQ<0]=0


if dataset_name=='labelme_vggfc7' :
     MAP = Macro_AP(Train_SNRQ>0, train_labels, Test_SNRQ>0, test_labels, 
                         num_return_NN=None)
     print(f"Macro average precision for K={K} is {MAP}")
     np.savez(result_path, MAP=MAP)
else:
    ## precision and recall are computed @ M_set points
    M_set = np.arange(250, n_train, 250) 
    MAP, precision, recall, precision_Radius = return_all_metrics(
        Train_SNRQ>0, train_labels, Test_SNRQ>0, test_labels, M_set, Radius=2)
    print(f"MAP={MAP}")
    
    if dataset_name=='nuswide_vgg':
        print(f"precision_5000={precision[19]}")
    else:
        print(f"precision_1000={precision[3]}")
        np.savez(result_path, MAP=MAP, precision=precision, recall=recall,
          precision_Radius=precision_Radius, M_set=M_set) 
    print(f"precision_Radius={precision_Radius}")   

