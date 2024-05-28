def AF_LF(samples,best_f,model,xmean,xstd,cost_fun, maximize = False, si = 0.0):
    import numpy as np
    import torch
    from torch.distributions import Normal
    samples= np.concatenate([((samples[0:-1]-xmean)/xstd).reshape(1,-1), samples[-1].reshape(-1,1)], axis = -1)
    samples=torch.tensor(samples.reshape(1,-1))
    with torch.no_grad():
        mean, std = model.predict(samples, return_std=True, include_noise = True)

    cost = torch.tensor(list(map(cost_fun, samples[:,-1].clone().detach())))

    mean=mean.reshape(-1,1)
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f - np.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * updf
    return -1 * (ei/cost)
        
        
def AF_HF(samples,best_f, model,xmean,xstd,cost_fun, maximize = False, si = 0.0,data_gen_func=None):
    import numpy as np
    import torch
    from torch.distributions import Normal
    
    samples= np.concatenate([((samples[0:-1]-xmean)/xstd).reshape(1,-1), samples[-1].reshape(-1,1)], axis = -1)
    samples=torch.tensor(samples.reshape(1,-1))
    with torch.no_grad():
        mean, std = model.predict(samples, return_std=True, include_noise = True)

    mean=mean.reshape(-1,1)

    cost = torch.tensor(list(map(cost_fun, samples[:,-1].clone().detach())))

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f - np.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u

    ei = sigma * u

    return -1* (ei/cost)
        
        
def AF_EI(samples,best_f, model,xmean,xstd,cost_fun, maximize = False, si = 0.0):
    import numpy as np
    import torch
    from torch.distributions import Normal
    samples= np.concatenate([((samples[0:-1]-xmean)/xstd).reshape(1,-1), samples[-1].reshape(-1,1)], axis = -1)
    samples=torch.tensor(samples.reshape(1,-1))
    with torch.no_grad():
        mean, std = model.predict(samples, return_std=True, include_noise = True)

    mean=mean.reshape(-1,1)

    cost = torch.tensor(list(map(cost_fun, torch.tensor(samples[:,-1].clone().detach(), dtype = torch.int64))))

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f - np.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei= sigma * (updf + u * ucdf)
    return -1* (ei/cost)
        
        
def AF_LF_Engineering(best_f, mean, std,x_val,cost_fun, maximize = True, si = 0.0, cost = None):
    import numpy as np
    import torch
    from torch.distributions import Normal
    
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    
    cost = torch.tensor(list(map(cost_fun, x_val[:,-1].clone().detach())))


    
    u = (mean - best_f - np.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        #si = -si
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * updf
    return (ei/cost)


def AF_HF_Engineering(best_f, mean, std,x_val, cost_fun,maximize = True, si = 0.0):
    import numpy as np
    import torch
    from torch.distributions import Normal
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)



    cost = torch.tensor(list(map(cost_fun, x_val[:,-1].clone().detach())))

    # deal with batch evaluation and broadcasting
    
    u = (mean - best_f - np.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        #si = -si
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ei = sigma * u
    return (ei/cost)