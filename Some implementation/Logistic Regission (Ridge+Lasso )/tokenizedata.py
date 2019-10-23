import subprocess
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
import scipy.io as sio
import numpy as np

def tokenizedata(directory='data/data_train',output='data_train'):


        
#         %% tokenization (X)
        
#         % Uncomment these lines if you want to do tokenization yourself
        
        print('Running python tokenization script ...\n')
        
        if(directory=='data/data_test'):
            subprocess(['python tokenize_spam.py data/data_test 10 > spamdata.csv'],shell=True);
        else:
            subprocess(['python tokenize_spam.py data/data_train 10 > spamdata.csv'],shell=True);
        print('Loading data ...\n');
        M = np.loadtxt(open("spamdata.csv","rb"),delimiter=",",skiprows=0)  
        
#         % normalize the colums to sum to 1
        col = np.array(M[:,0])
        row = np.array(M[:,1])
        data = np.ones((len(M)))

        X = csr_matrix((data, (row, col)), shape=(1024, 5000)).toarray()
        
        X=X*(spdiags(1./sum(X).T,0,X.shape[1],X.shape[1]))
        
#         % Load in labels (do not change this part of the code)
        if(directory=='data/data_test'):
            b=np.loadtxt('data/data_test/index',dtype=str)
        else:
            b=np.loadtxt('data/data_train/index',dtype=str)
        label = b[:,0]
        

        Y=np.zeros((1,len(label)))
        for i in range(len(label)):
            if(label[i]=='spam'):
                Y[:,i]=1
            else:
                Y[:,i]=-1
 
        if(output=='data_test'):  
            sio.savemat('data/data_test.mat', {'X': X,'Y': Y})
        else:
            sio.savemat('data/data_train.mat', {'X': X,'Y': Y})
        




