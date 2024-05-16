import pandas as pd
import numpy as np
import seaborn
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
# huge images can't be displayed on a screen, so directly use Agg
import matplotlib
matplotlib.use("Agg")
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
import statsmodels.api as sm
#import statsmodels.formula.api as smf
from scipy import stats
from DataSciency import DataSciency
import os, sys

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

d = DataSciency()

train_sets = []
files = []
max_score = float('-inf') # r2
max_score_params = []
best_name ="no model assessed yet"

for filename in os.listdir("./heavy_tailed_data/"):
    if filename.endswith(".csv"):
        files.append(filename)
        train_sets.append(pd.read_csv('./heavy_tailed_data/'+filename, header= 0))#, names = ['x','y']))
        
train = train_sets[0]                    
col_names = train.columns.values
print col_names
train.head(5)

# data visualization and normality tests
for data, filename in zip(train_sets, files):
    x=data.iloc[:,0]
    y=data.iloc[:,1]
    print len(y), " values for Y train -unique: ", len(set(list(y)))
    print len(x), " values for X test -unique: ", len(set(list(x)))  
    d.test_normality(y)
    d.visualize_normality_for_sample(y)
    d.plot_histogram(y, filename)

for filename, data in zip(files,train_sets):
    d.visualize_feature_correlations(data, 'spearman') #'pearson') # assumes Normal distrib. (not in this case)
    d.visualize_feature_correlations(data, 'kendall')

def update_best_model_score_and_params(name, model, params, current_score, max_score):
    if current_score > max_score:
        max_score = current_score
        max_score_params = params
        best_model = model
        best_name = name

# Data models 
best_model = None
alpha = 0.5 #[0.5, 1.0]
ridge = Ridge(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)

lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
l1_ratios = [0.5, 0.7]
EN = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

# Distribution Families
# Family(link, variance)	The parent class for one-parameter exponential families.
# Binomial([link])	Binomial exponential family distribution.
# Gamma([link])	Gamma exponential family distribution.
# Gaussian([link])	Gaussian exponential family distribution.
# InverseGaussian([link])	InverseGaussian exponential family.
# NegativeBinomial([link, alpha])	Negative Binomial exponential family.
# Poisson([link])	Poisson exponential family.

exp_family_distribs = [sm.families.Gaussian(), sm.families.Gamma()]#, sm.families.InverseGaussian()]#, sm.families.NegativeBinomial(), sm.families.Poisson()] # sm.families.Binomial(),
linear_models = [ridge, lasso, EN]
filenames_log=[]; a_params=[]; b_params = []; r2_logs=[]; model_logs=[];
# Link functions per family (Not all link functions are available for each distribution family). 
# The list of available link functions can be obtained by   sm.families.family.<familyname>.links
for data, filename in zip(train_sets, files):
    max_score = float('-inf') # r2
    max_score_params = []
    best_name ="no model assessed yet"
    
    x, y = d.get_xy_reshaped_for_numpy(data)
    
    # Linear models
    ###### Ordinary Least Squares (OLS) ###############################
    # An intercept is not included by default and should be added by the user. See statsmodels.tools.add_constant.
    X = sm.add_constant(x, prepend=False)
    OLS = sm.OLS(y,X)
    model = OLS.fit()
    params = model.params
    r2 = model.rsquared
    print filename,' OLS Parameters: ', params, ' R2: ', r2
    print(model.summary())
    
    update_best_model_score_and_params("OLS", model, params, r2, max_score)
    filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model); 
    
    #### 
    d.OLS_nonlinear_curve_but_linear_in_params(x, y)
    
    #### Regularizer regressors ######################################
    for regressor, name in zip([ridge, lasso, EN], ['Ridge','Lasso','ElasticNet']):
        model = d.fit_model(regressor, x, y, file)
        coef = model.coef_.tolist() 
        if isinstance(coef[0], list):
            coef = coef[0][0]
        else:
            coef = coef[0]
        intercept = model.intercept_.tolist()[0] #np.asmatrix(model.intercept_)[0]
        params = [coef, intercept]
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred) #regressor.rsquared
        print filename,' Regularizer Parameters: ', params, ' R2: ', r2, ' coef and intercept: ',coef, intercept
        update_best_model_score_and_params("Regressor"+name, model, params, r2, max_score)
        filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model); 
    
    #### GLM #########################################################
    for family in exp_family_distribs:
        glm = sm.GLM(y, x, family=family)
        model = glm.fit()  
        params = model.params
        if len(params)==1:
            params = [params.tolist()[0], 0]
        # LLF: float Value of the loglikelihood function evalued at params.
        print filename," GLM family parameters: ",family,':\n', params, "\nLLF and Pearson Chi2: ",model.llf,' ', model.pearson_chi2,'\n', model.summary()
        update_best_model_score_and_params("GLM_Family_"+str(family), model, params, r2, max_score)
        #filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model); 
    
    print filename+" Best r2 and params: ",max_score, max_score_params,"\n",best_model, best_name    
    # PLOT MIC
    d.plot_covariance_based_mutual_info_for_categorical_correlations(data, filename)

df = pd.DataFrame({'filename':filenames_log,'a': a_params, 'b':b_params, 'r2': r2_logs,
              'model_name':model_logs})
df = df[['filename','a','b','r2','model_name']]
df.to_csv("./output/regression_results.csv")    

def Huber_regression(x,y, filename):
    """
    Huber regression with scikit-learn
    Since ridge is strongly influenced by the outliers present in the dataset, Huber regressor 
    is less influenced by the outliers since the model uses the linear loss for these. 
    The HuberRegressor is different to Ridge because it applies a linear loss to samples that are classified 
    as outliers. A sample is classified as an inlier if the absolute error of that sample is lesser than a certain 
    threshold epsilon. As the parameter epsilon is increased for the Huber regressor, 
    the decision function approaches that of the ridge.
    -Uses Huber loss and returns the fitted model, the coefficients and the r2 coef of determination, best epsilon and alpha
    """
    plt.plot(x, y, 'b.')

    # Fit the huber regressor over a series of epsilon values.
    colors = ['r-', 'b-', 'y-', 'm-']
    best_r2 = float('-inf') # the closer to 1, the better fit the model provides
    alpha = best_alpha = 0.0 # default #for alpha in [0.0001, 0.1, 1.0, 10.0, 0.001, 0.01, 0.0]:
    best_coef = [-1, -1] 
    epsilon_values = [1.35, 1.5, 1.75, 1.9]
    for k, epsilon in enumerate(epsilon_values):
        huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                               epsilon=epsilon)
        fitted = huber.fit(x, y)
        coef_ = huber.coef_ * x + huber.intercept_
        plt.plot(x, coef_, colors[k], label=("Huber loss, epsilon %s, alpha %s" % (epsilon, best_alpha)))
        intercept = huber.intercept_  #params.append(intercept)
        r2 = fitted.score(x, y)  
        if r2 > best_r2:
            best_coef = params
            best_intercept = huber.intercept_
            best_r2 = r2
            best_fitted = fitted
            best_epsilon = epsilon
            best_alpha = alpha
    
    # Fit a ridge regressor to compare it to huber regressor.
    ridge = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
    ridge.fit(x, y)
    coef_ridge = ridge.coef_
    coef_ = ridge.coef_ * x + ridge.intercept_
    plt.plot(x, coef_, 'g-', label="ridge regression")

    plt.savefig('./output/HuberRegressorVSRidge_'+filename.replace('.csv','.png'))
    plt.title("Comparison of Huber Regressor vs Ridge: "+filename)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc=1, scatterpoints=1)#numpoints = 1) bbox_to_anchor=(1, 0.5))#loc=1) loc= 'upper left'#plt.legend(loc=0)
    plt.show()
    
    print " Best Huber Regr. fitted model had ", best_coef," coefficients, intercept: ",best_intercept
    print " R2: ", best_r2," alpha: ", best_alpha, " and epsilon= ", best_epsilon
    return best_fitted, best_coef, best_r2, best_epsilon, best_alpha

filenames_log=[]; a_params=[]; b_params = []; r2_logs=[]; model_logs=[]; epsilons = []; alphas = []
for filename, data in zip(files,train_sets):
    x, y = d.get_xy_reshaped_for_numpy(data)
    
    # Huber regression A (no intercepts, and no r2)
    #model, params, r2 = robust_linear_model_Huber_loss_funct(x,y)
    #filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model); epsilons.append(-1)     
    
    # Huber regression B (my solution)
    model, params, r2, epsilon, alpha = Huber_regression(x,y, filename)
    filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model); epsilons.append(epsilon); alphas.append(alpha)
    
df = pd.DataFrame({'filename':filenames_log,'a': a_params, 'b':b_params, 'r2': r2_logs,
              'epsilon':epsilons,'alpha': alphas, 'model_name':model_logs})
df = df[['filename','a','b','r2','epsilon','alpha','model_name']]
df.to_csv("./output/robust_regression_results.csv")

