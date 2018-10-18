import os
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path        = os.path.expanduser('~')+'/closure'
outpath     = path+'/data/output/'
ml_path  	= path+'/data/train/'
inpath      = path+'/data/tested/'
files  = os.listdir(outpath)

hard    = [1,3,4]
core    = [8,9,10,11,15]
overlap = [12,13,14]
soft    = [2,6]
tot     = [1,2,3,4,6,8,9,10,11,12,13,14,15]

used = tot


for i in range(len(files)):
# for i in range(5):    
    n = np.floor(int(re.findall('\d+', files[i])[0])/1000).astype(int)
    j = 0
    if np.any(used==n):
        output_fp   = np.loadtxt(outpath+files[i])
        r           = output_fp[:,0]
        g           = output_fp[:,2]
        sigma_g     = output_fp[:,3]
        c           = output_fp[:,4]
        sigma_c     = output_fp[:,5]

        dg          = np.gradient(g, r[0])
        dc          = np.gradient(c, r[0])

        try:
            trunc_zero = np.max(np.where(g==0.0))+1
        except ValueError:
            trunc_zero = 0
        if n == 4:
            trunc_zero = np.min(np.where(g>1.0))+1
        trunc_small = 5
        trunc = max(trunc_small, trunc_zero)

        bridge = np.log(g[trunc:])+c[trunc:]+1.-g[trunc:]+phi[trunc:]

        bridge_min = np.argmin(bridge)
        bridge = bridge[bridge_min:]
        trunc += bridge_min

        sparse = np.max(np.where(r<10))+1

        form = n * np.ones_like(g[trunc:sparse])

        if 'X' in dir() :
            X_temp  = np.array((bridge[:sparse-trunc], g[trunc:sparse] - 1., c[trunc:sparse], dg[trunc:sparse], dc[trunc:sparse], 
                                sigma_g[trunc:sparse], sigma_c[trunc:sparse], form))
            X       = np.hstack((X,X_temp))
        else:
            X       = np.array((bridge[:sparse-trunc], g[trunc:sparse] - 1., c[trunc:sparse], dg[trunc:sparse], dc[trunc:sparse],
                                sigma_g[trunc:sparse], sigma_c[trunc:sparse], form))
        
            

X = np.transpose(X)


X_train, X_test, y_train, y_test = train_test_split(X[:,2:], X[:,0:2], test_size=0.3,  random_state=43)

train_set 		= np.column_stack((y_train, X_train))
test_set  		= np.column_stack((y_test, X_test))

np.savetxt(ml_path+'whole.txt', np.random.permutation(X))
np.savetxt(ml_path+'train.dat', train_set)
np.savetxt(ml_path+'test.dat' , test_set)
