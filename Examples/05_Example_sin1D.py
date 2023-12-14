from gpplus.models import GP_Plus
from gpplus.test_functions.analytical import sine_1D
from gpplus.preprocessing import train_test_split_normalizeX
from gpplus.utils import set_seed
import matplotlib.pyplot as plt

random_state = 1245
set_seed(random_state)
############################ Generate Data #########################################
X, y = sine_1D(n = 10000, random_state= random_state, frequency=1.0, noise_std=0.0)
############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99)
############################### Model ##############################################
model = GP_Plus(Xtrain, ytrain, fixed_length_scale=True,quant_correlation_class ='Matern32Kernel')
############################### Fit Model ##########################################
model.fit(n_jobs = -1,num_restarts = 16)
############################### evaluation ##############################################
model.evaluation(Xtest, ytest)  
############################### plot results ##############################################
model.plot_xy_print_params(Xtest, ytest, Xtrain, ytrain, model)