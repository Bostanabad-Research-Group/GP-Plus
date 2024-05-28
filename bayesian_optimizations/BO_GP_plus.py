import numpy as np
import torch
from gpplus.models import GP_Plus
from gpplus.optim import fit_model_scipy
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from gpplus.bayesian_optimizations.AFs import AF_HF,AF_LF, AF_HF_Engineering,AF_LF_Engineering
import matplotlib.pyplot as plt

    
def BO(Xtrain=None,
        ytrain=None,
        costs=None,
        l_bound=None,
        u_bound=None,
        xmean=None,
        xstd=None,
        qual_index=None,
        data_gen_func=None,
        n_train=None,
        maximize_flag=False,
        one_iter=False,
        max_cost=40000,
        MF=True,
        AF_hf=AF_HF,
        AF_lf=AF_LF,
        max_iter=2,
        IS=True):  
      
    ymin_list = []
    xmin_list = []   
    cumulative_cost = []
    bestf = []
    Fidelity=[]
    num_fidelity=list(qual_index.values())[-1]
    def cost_fun(x):
        return costs[str(int(x))]
    
    def bestf_calculator(MF,num_fidelity,ytrain,Xtrain):
        if MF:
            best_values = []
            if maximize_flag:
                for i in range(num_fidelity):
                    best_values.append((ytrain[Xtrain[:,-1] == i].max().reshape(-1,).item()))
            else:
                for i in range(num_fidelity):
                    best_values.append((ytrain[Xtrain[:,-1] == i].min().reshape(-1,).item()))
        else:
            best_values = []
            if maximize_flag:
                best_values.append((ytrain.max().reshape(-1,)).item())
            else:
                best_values.append((ytrain.min().reshape(-1,)).item())
        
        return best_values
    
    
    def run_scipy(EI, best_f, bound, model,xmean,xstd,cost_fun, fidelity,l_bound,u_bound):
        import torch
        import numpy as np
        from scipy.optimize import minimize
        random_seed = np.random.choice(range(0,1000), size=12, replace=False)
        def run(EI, best_f, bound, model,xmean,xstd,cost_fun,k, fidelity,l_bound_h,u_bound_h):
            l_bound_h=l_bound+[fidelity]
            u_bound_h=u_bound+[fidelity]
            np.random.seed(random_seed[k])
            samples_h = torch.tensor((np.random.uniform(l_bound_h,u_bound_h)).reshape(1,-1))
            samples_h[:,-1] = torch.round(samples_h[:,-1])
            result_h = minimize(EI,samples_h.reshape(-1,), args=(best_f, model,xmean,xstd,cost_fun),bounds=bound)
            temp = result_h.fun
            tempx = result_h.x
            return temp, tempx
        
        set_loky_pickler("dill") 
        out = Parallel(n_jobs=-1,verbose=0)(delayed(run)(EI, best_f, bound, model, xmean,xstd,cost_fun,k,fidelity,l_bound,u_bound) for k in range(12))
        set_loky_pickler("pickle")
        temp = [out[i][0] for i in range(len(out))]
        tempx = [out[i][1] for i in range(len(out))]
        min_index = np.argmin(temp)
        Y_h = temp[min_index]
        X_h = tempx[min_index]
        return Y_h, X_h
    

    
    if callable(data_gen_func):
        initial_cost = np.sum(list(map(cost_fun,Xtrain[:,-1])))
        cumulative_cost.append(initial_cost)
        
        
        if type(ytrain)!=torch.tensor:
            ytrain=torch.tensor(ytrain)
        ytrain=ytrain.reshape(-1)
        problem = lambda x: data_gen_func(False,x)
        while cumulative_cost[-1] < max_cost:
            
            best_values=bestf_calculator(MF,num_fidelity,ytrain,Xtrain)
            bestf.append(best_values[0])
            if len(bestf)> max_iter:
                if np.var(bestf[-max_iter:])<1e-6:
                    break
                
            model = GP_Plus(Xtrain, ytrain, qual_index,IS=IS)

            _ = fit_model_scipy(model, bounds=True)
            
                
            X_list=[]   
            y_list=[]
            
            for i in range(num_fidelity):
                if i==0:
                    bound=tuple(list(zip(l_bound, u_bound))+[(i,i)])
                    Y_h, X_h = run_scipy(AF_hf, best_values[0], bound, model,np.array(xmean),np.array(xstd),cost_fun, i,l_bound,u_bound)
                    X_list.append(X_h)
                    y_list.append(Y_h)
                else:
                    bound=tuple(list(zip(l_bound, u_bound))+[(i,i)])
                    Y_l, X_l = run_scipy(AF_lf, best_values[0], bound, model,np.array(xmean),np.array(xstd),cost_fun, i,l_bound,u_bound)
                    X_list.append(X_l)
                    y_list.append(Y_l)
            
                
                    
            min_index = np.argmin(y_list)
            temp = torch.tensor(X_list[min_index])
            ynew = problem(temp.unsqueeze(0))
            
            
            if MF:
                Xnew= np.concatenate([((temp[0:-1]-xmean)/xstd).reshape(1,-1), temp[-1].reshape(-1,1)],axis=-1)
            else:
                Xnew= ((temp-xmean)/xstd).reshape(1,-1)
                

                
            
            Xtrain = torch.cat([Xtrain,torch.tensor(Xnew.reshape(1,-1))])
            ytrain = torch.cat([ytrain, ynew.reshape(-1,)])
            ymin_list.append(ynew.reshape(-1,))
            xmin_list.append(Xnew)
            ##################
            cumulative_cost.append(initial_cost + cost_fun(Xnew[0][-1]))
            initial_cost = cumulative_cost[-1]
            Fidelity.append(Xnew[0][-1])
            if one_iter:
                best_values=bestf_calculator(MF,num_fidelity,ytrain,Xtrain)
                bestf.append(best_values[0])
                break

        
    else:
        Xtrain = np.empty((0, data_gen_func.shape[1]-1))
        ytrain = np.empty((0,))
        for i in range(num_fidelity):
            random_index=np.random.randint(0, len(data_gen_func[data_gen_func[:,-2]==i]), n_train[i])
            Xtrain = np.append(Xtrain, data_gen_func[data_gen_func[:,-2]==i][random_index][:,0:-1], axis=0)
            ytrain = np.append(ytrain, data_gen_func[random_index][:,-1], axis=0)
            
            
        Xtrain=torch.tensor(Xtrain)
        ytrain=torch.tensor(ytrain)
        initial_cost = np.sum(list(map(cost_fun,Xtrain[:,-1])))
        cumulative_cost.append(initial_cost)
        
        
        while cumulative_cost[-1] < max_cost:
            best_values=bestf_calculator(MF,num_fidelity,ytrain,Xtrain)
                    
            bestf.append(best_values[0])
            if len(bestf)> max_iter:
                if np.var(bestf[-max_iter:])<1e-6:
                    break
                
            model = GP_Plus(Xtrain, ytrain, qual_index,IS=IS)

            _ = fit_model_scipy(model, bounds=True)
            

            X_list=[]   
            y_list=[]
            
            for i in range(num_fidelity):
                scores=[]
                with torch.no_grad():
                    ytest, ystd = model.predict(torch.tensor(data_gen_func[data_gen_func[:,-2]==i][:,0:-1]), return_std=True,include_noise = False)
                if i==0:
                    # scores.append(AF_hf(best_values[i], ytest.reshape(-1,1), ystd.reshape(-1,1), maximize = maximize_flag)) 
                    scores.append(AF_HF_Engineering(best_values[i], ytest.reshape(-1,1), ystd.reshape(-1,1),torch.tensor(data_gen_func[data_gen_func[:,-2]==i][:,0:-1]),cost_fun,maximize = maximize_flag)) 
                else:
                    scores.append(AF_LF_Engineering(best_values[i], ytest.reshape(-1,1), ystd.reshape(-1,1),torch.tensor(data_gen_func[data_gen_func[:,-2]==i][:,0:-1]),cost_fun,maximize = maximize_flag))
            
            scores_final=torch.cat(scores,dim=0)
            index = torch.argmax(scores_final)
            
            Xnew = data_gen_func[index][0:-1]
            Xnew=torch.tensor(Xnew)
            ynew = data_gen_func[index][-1]
            
            Xtrain = torch.cat([Xtrain,Xnew.reshape(1,-1)])
            ytrain = torch.cat([ytrain, torch.tensor(ynew).reshape(-1,)], dim=0)
            ymin_list.append(ynew.reshape(-1,))
            xmin_list.append(Xnew)

            cumulative_cost.append(initial_cost + cost_fun(Xnew[-1]))
            initial_cost = cumulative_cost[-1]
            Fidelity.append(Xnew[-1])
            if one_iter:
                best_values=bestf_calculator(MF,num_fidelity,ytrain,Xtrain)
                bestf.append(best_values[0])
                break
    

    

    return np.array(bestf), np.array(cumulative_cost)



def Visualize_BO(bestf,cost):
    plt.scatter(cost,bestf)
    plt.ylabel('y^*')
    plt.xlabel('cost')
    plt.show()
    



