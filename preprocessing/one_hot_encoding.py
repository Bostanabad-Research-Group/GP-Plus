def one_hot_encoding(X,qual_index):
    import numpy as np
    import torch
    import torch.nn.functional as F
    X_numerical= torch.tensor(np.delete(X.numpy(), list(qual_index.keys()), 1))
    X_CAT_one=torch.zeros(X_numerical.shape[0],sum(qual_index.values()))
    # Xtrain_CAT_one=qual_index
    Xtrain_CAT_one=[]
    k=0
    for i in qual_index.keys():
        x_one_hoted=F.one_hot(X[:,i].to(torch.int64), num_classes=qual_index[i])
        # Xtrain_CAT_one[i]=x_one_hoted
        if k==0:
            X_CAT_one_all=x_one_hoted
            k=+1
        else:
            X_one_hot_all=torch.cat((X_CAT_one_all,x_one_hoted),1)


    X_final=torch.cat((X_one_hot_all.to(torch.int64),X_numerical),1)
    return X_final