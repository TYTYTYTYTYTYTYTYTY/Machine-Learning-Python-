import numpy as np
def spamupdate(w,email,truth):

    # Input:
    # w     weight vector
    # email instance vector
    # truth label
    #
    # Output:
    #
    # updated weight vector
    #
    # INSERT CODE HERE:
    updatedir=2*np.squeeze(email*np.matmul(np.transpose(email),w)-truth).reshape(w.shape[0],1)+ 2*0.004* w
    w=w-0.014*updatedir

    return w
