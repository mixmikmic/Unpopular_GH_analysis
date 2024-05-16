kmf.fit(timevar,event_observed = eventvar,label = "All groups")
plt.figure()
plt.plot(kmf.survival_function_.index.values,-np.log(kmf.survival_function_['All groups']),linewidth = 2.0)
plt.ylabel('-ln(S(t))')
plt.xlabel('Time')
plt.title('Exponential')
y = -np.log(kmf.survival_function_['All groups'])
X = np.asarray(kmf.survival_function_.index.values)
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

#Define the likelihood function.
#Although we are not using any covariates at this time, we use a vector of
#ones as the 'X' argument for the Generallikelihood function
def _ll_exponential(y,X,scale):
    ll = eventvar * np.log(scale) - (scale * y)
    return ll 
#The Exponential class uses the _ll_exponential function to iteratively update the 
#parameter values to maximiez the likelihood function
class Exponential(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Exponential, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        ll = _ll_exponential(self.endog,self.exog,params)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if start_params == None:
            start_params = start_params_var 
        return super(Exponential, self).fit(start_params = start_params,maxiter = maxiter, maxfun = maxfun,**kwds)

kmf.fit(timevar,event_observed = eventvar,label = "All groups")
#Exponential
#Specify starting values for the parameters
start_params_var = np.repeat(0.5,1)
exp_data = np.repeat(1,len(timevar))
mod_exponential = Exponential(timevar,exp_data)
res_exp = mod_exponential.fit()
print(res_exp.summary())

#Plot the exponential prediction against the empirical survival curve
#Prediction intervals are calculated by complementary log - log transform
#of the survival prediction. The variance of the log - log of the survival
#term is approximated by the delta method, explained on the state website:
#www.stata.com/support/faqs/statistics/delta-method
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_exponential.predict_survival_exponential(res_exp.params, t))
plt.plot(t,mod_exponential.predict_survival_exponential_cis(res_exp.params, res_exp.bse, t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_exponential.predict_survival_exponential_cis(res_exp.params, res_exp.bse, t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Exponential')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Exponential.png',dpi = 300)

plt.figure()
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['All groups'])),linewidth = 2.0)
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Weibull')
y = np.log(-np.log(kmf.survival_function_['All groups'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

def _ll_weibull(y,X,scale,gamma):
    ll = eventvar * np.log(scale * gamma * np.power(y, (gamma - 1))) - (scale * np.power(y,gamma))
    return ll

class Weibull(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Weibull, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        scale = params[0]
        gamma = params[1]
        ll = _ll_weibull(self.endog,self.exog,scale,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(Weibull, self).fit(start_params = start_params, maxiter = maxiter, maxfun = maxfun,**kwds)

#Weibull
start_params_var = np.repeat(0.5,2)
weibull_data = np.repeat(1,len(timevar))
mod_weibull = Weibull(timevar,weibull_data)
res_wei = mod_weibull.fit()
print(res_wei.summary())

#Plot the weibull prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_weibull.predict_survival_weibull(res_wei.params, t))
plt.plot(t,mod_weibull.predict_survival_weibull_cis(res_wei.params, res_wei.cov_params(), t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_weibull.predict_survival_weibull_cis(res_wei.params, res_wei.cov_params(), t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Weibull')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Weibull.png',dpi = 300)

plt.figure()
t_1 = kmf.survival_function_.iloc[0:len(kmf.survival_function_) - 1]
t_2 = kmf.survival_function_.iloc[1:len(kmf.survival_function_)]
gompertz_hazard = []
gompertz_hazard.append(0)
for index in range(len(t_1)):
    #Because we are summing the log of the hazard, check if the 
    #hazard is zero between t_2 and t_1. 
    if (-np.log(t_2.iloc[index,0])+np.log(t_1.iloc[index,0])) > 0:
        gompertz_hazard.append(gompertz_hazard[index] - np.log(-np.log(t_2.iloc[index,0])+np.log(t_1.iloc[index,0])))
    else:
    #If it is, append the latest value of the cumulative hazard at time t_2
        gompertz_hazard.append(gompertz_hazard[index])
plt.plot(t_1.index.values,gompertz_hazard[:-1],linewidth = 2.0)
plt.title('Gompertz')
plt.ylabel('-Integral(ln(h(t)))')
plt.xlabel('Time')
y = gompertz_hazard[:-1]
X = t_1.index.values
X = st.add_constant(X,prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,0],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

def _ll_gompertz(y,X,scale,gamma):
    ll = eventvar * (scale + gamma * y) - ((np.exp(scale) * (np.exp(gamma * y) - 1)) / gamma)    
    return ll

class Gompertz(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Gompertz, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        scale = params[0]
        gamma = params[1]
        ll = _ll_gompertz(self.endog,self.exog,scale,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(Gompertz, self).fit(start_params = start_params, maxiter = maxiter, maxfun = maxfun,**kwds)

#Gompertz
start_params_var = [0,0]
gompertz_data = np.repeat(1,len(timevar))
mod_gompertz = Gompertz(timevar,gompertz_data)
res_gomp = mod_gompertz.fit()
print(res_gomp.summary())

#Plot the gompertz prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_gompertz.predict_survival_gompertz(res_gomp.params, t))
plt.plot(t,mod_gompertz.predict_survival_gompertz_cis(res_gomp.params, res_gomp.cov_params(), t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_gompertz.predict_survival_gompertz_cis(res_gomp.params, res_gomp.cov_params(), t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Gompertz')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Gompertz.png',dpi = 300)

plt.figure()
plt.plot(np.log(kmf.survival_function_.index.values),-np.log(kmf.survival_function_['All groups']/(1-kmf.survival_function_['All groups'])),linewidth = 2.0)
plt.ylabel('Probability of ending')
plt.xlabel('Years')
plt.title('Log-Logistic')
plt.ylabel('-(ln(S(t)/(1-S(t))))')
plt.xlabel('ln(Time)')
y = -np.log(kmf.survival_function_['All groups'].iloc[1:len(kmf.survival_function_)]/(1-kmf.survival_function_['All groups'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

def _ll_loglogistic(y,X,scale,gamma):
    ll = eventvar * np.log((scale * gamma * np.power(y,gamma - 1))/(1 + scale * np.power(y,gamma))) + np.log(1/(1 + scale * np.power(y,gamma)))    
    return ll

class Loglogistic(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Loglogistic, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        scale = params[0]
        gamma = params[1]
        ll = _ll_loglogistic(self.endog,self.exog,scale,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(Loglogistic, self).fit(start_params = start_params, maxiter = maxiter, maxfun = maxfun,**kwds)

#Log-Logistic
start_params_var = [0,0]
loglogistic_data = np.repeat(1,len(timevar))
mod_loglogistic = Loglogistic(timevar,loglogistic_data)
res_loglog = mod_loglogistic.fit()
print(res_loglog.summary())

#Plot the loglogistic prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_loglogistic.predict_survival_loglogistic(res_loglog.params, t))
plt.plot(t,mod_loglogistic.predict_survival_loglogistic_cis(res_loglog.params, res_loglog.cov_params(), t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_loglogistic.predict_survival_loglogistic_cis(res_loglog.params, res_loglog.cov_params(), t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Loglogistic')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Loglogistic.png',dpi = 300)

plt.figure()
plt.plot(np.log(kmf.survival_function_.index.values),norm.ppf(1 - kmf.survival_function_['All groups']),linewidth = 2.0)
plt.xlabel('Years')
plt.title('Log-Normal')
plt.ylabel('norm.ppf(1 - S(t))')
plt.xlabel('ln(Time)')
y = norm.ppf(1 - kmf.survival_function_['All groups'].iloc[1:len(kmf.survival_function_)])
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

def _ll_lognormal(y,X,scale,gamma):
    ll = eventvar * (
    np.log(
    norm.pdf(((np.log(y) - scale) * gamma))/
    (y * (1/gamma) * (1 - norm.cdf((np.log(y) - scale) * gamma))))
    ) + np.log(1 - norm.cdf((np.log(y) - scale) * gamma))
    return ll

class Lognormal(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Lognormal, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        scale = params[0]
        gamma = params[1]
        ll = _ll_lognormal(self.endog,self.exog,scale,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(Lognormal, self).fit(start_params = start_params, maxiter = maxiter, maxfun = maxfun,**kwds)

#Log-Normal
start_params_var = [0,0]
lognormal_data = np.repeat(1,len(timevar))
mod_lognormal = Lognormal(timevar,lognormal_data)
res_lognorm = mod_lognormal.fit()
print(res_lognorm.summary())

#Plot the lognormal prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_lognormal.predict_survival_lognormal(res_lognorm.params, t))
plt.plot(t,mod_lognormal.predict_survival_lognormal_cis(res_lognorm.params, res_lognorm.cov_params(), t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_lognormal.predict_survival_lognormal_cis(res_lognorm.params, res_lognorm.cov_params(), t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Lognormal')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Lognormal.png',dpi = 300)

def _ll_generalizedgamma(y,X,scale,kappa,sigma):
    gammavalue = np.power(np.abs(kappa),-2)
    zeta = np.sign(kappa) * (np.log(y) - scale) / sigma
    upsilon = gammavalue * np.exp(np.abs(kappa)*zeta)        
    if kappa > 0:
        density = np.power(gammavalue,gammavalue) * np.exp(zeta * np.power(gammavalue,0.5) - upsilon) / (sigma * y * np.power(gammavalue,0.5) * gammafunction(gammavalue))
        survival = 1 - gammainc(gammavalue,upsilon)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    elif kappa == 0: 
        density = np.exp(- np.power(zeta,2) / 2) / (sigma * y * np.power(2 * np.pi,0.5))
        survival = 1 - norm.cdf(zeta)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    elif kappa < 0:
        density = np.power(gammavalue,gammavalue) * np.exp(zeta * np.power(gammavalue,0.5) - upsilon) / (sigma * y * np.power(gammavalue,0.5) * gammafunction(gammavalue))
        survival = gammainc(gammavalue,upsilon)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    return ll
    
class Generalizedgamma(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(Generalizedgamma, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        scale = params[0]
        kappa = params[1]
        sigma = params[2]
        ll = _ll_generalizedgamma(self.endog,self.exog,scale,kappa,sigma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('kappa')
            self.exog_names.append('sigma')
        if start_params == None:
            start_params = start_params_var 
        return super(Generalizedgamma, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#Generalized gamma
start_params_var = [1,1,1]
generalizedgamma_data = np.repeat(1,len(timevar))
mod_generalizedgamma = Generalizedgamma(timevar,generalizedgamma_data)
res_generalizedgamma = mod_generalizedgamma.fit()
print(res_generalizedgamma.summary())


#Plot the generalized gamma prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(1,150,151)
plt.plot(t,mod_generalizedgamma.predict_survival_generalizedgamma(res_generalizedgamma.params, t))
plt.plot(t,mod_generalizedgamma.predict_survival_generalizedgamma_cis(res_generalizedgamma.params, res_generalizedgamma.cov_params(), t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_generalizedgamma.predict_survival_generalizedgamma_cis(res_generalizedgamma.params, res_generalizedgamma.cov_params(), t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Generalized gamma')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('Generalizedgamma.png',dpi = 300)

def _ll_exponentialph(y,X,beta):
    scale = np.exp(np.dot(X,beta))
    ll = eventvar * np.log(scale) - (scale * y)    
    return ll 

class ExponentialPH(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(ExponentialPH, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        ll = _ll_exponentialph(self.endog,self.exog,params)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if start_params == None:
            start_params = start_params_var 
        return super(ExponentialPH, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

kmf.fit(timevar,event_observed = eventvar,label = "All groups")
#ExponentialPH
start_params_var = np.repeat(0,len(survivaldata.columns))
mod_exponentialph = ExponentialPH(timevar,survivaldata)
res_expPH = mod_exponentialph.fit()
print(res_expPH.summary())

#Plot the exponential PH prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_exponentialph.predict_survival_exponential_ph(res_expPH.params,survivaldata,t))
plt.plot(t,mod_exponentialph.predict_survival_exponential_ph_cis(res_expPH.params,res_expPH.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_exponentialph.predict_survival_exponential_ph_cis(res_expPH.params,res_expPH.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Exponential PH')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('ExponentialPH.png',dpi = 300)

def _ll_weibullph(y,X,beta,gamma):
    scale = np.exp(np.dot(X,beta))
    ll = eventvar * np.log(scale * gamma * np.power(y, (gamma - 1))) - (scale * np.power(y,gamma))
    return ll 
   
class WeibullPH(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(WeibullPH, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_weibullph(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(WeibullPH, self).fit(start_params = start_params,method = 'basinhopping',maxiter = maxiter, maxfun = maxfun,**kwds)

#WeibullPH
#Set the initia gamma value to 1
start_params_var = np.repeat(1,len(survivaldata.columns) + 1)
mod_weibullph = WeibullPH(timevar,survivaldata)
res_weiPH = mod_weibullph.fit()
print(res_weiPH.summary())

#Plot the weibull PH prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_weibullph.predict_survival_weibull_ph(res_weiPH.params,survivaldata,t))
plt.plot(t,mod_weibullph.predict_survival_weibull_ph_cis(res_weiPH.params,res_weiPH.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_weibullph.predict_survival_weibull_ph_cis(res_weiPH.params,res_weiPH.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Weibull PH')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('WeibullPH.png',dpi = 300)

def _ll_gompertzph(y,X,beta,gamma):
    scale = np.dot(X,beta)
    ll = eventvar * (scale + gamma * y) - ((np.exp(scale) * (np.exp(gamma * y) - 1)) / gamma)    
    return ll
    
class GompertzPH(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(GompertzPH, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_gompertzph(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(GompertzPH, self).fit(start_params = start_params,method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#GompertzPH
start_params_var = np.repeat(0,len(survivaldata.columns))
start_params_var = np.append(start_params_var,1)
mod_gompertzph = GompertzPH(timevar,survivaldata)
res_gompPH = mod_gompertzph.fit()
print(res_gompPH.summary())

#Plot the Gompertz PH prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_gompertzph.predict_survival_gompertz_ph(res_gompPH.params,survivaldata,t))
plt.plot(t,mod_gompertzph.predict_survival_gompertz_ph_cis(res_gompPH.params,res_gompPH.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_gompertzph.predict_survival_gompertz_ph_cis(res_gompPH.params,res_gompPH.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Gompertz PH')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('GompertzPH.png',dpi = 300)

def _ll_loglogisticpo(y,X,beta,gamma):
    scale = np.exp(np.dot(X,beta))
    ll = eventvar * np.log((scale * gamma * np.power(y,gamma - 1))/(1 + scale * np.power(y,gamma))) + np.log(1/(1 + scale * np.power(y,gamma)))    
    return ll

class LoglogisticPO(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(LoglogisticPO, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_loglogisticpo(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(LoglogisticPO, self).fit(start_params = start_params, method = 'powell',maxiter = maxiter, maxfun = maxfun,**kwds)

#LoglogisticPO
#Set the initial gamma value to 1
start_params_var = np.repeat(1,len(survivaldata.columns) + 1)
mod_loglogisticpo = LoglogisticPO(timevar,survivaldata)
res_loglogPO = mod_loglogisticpo.fit()
print(res_loglogPO.summary())

#Plot the loglogistic PO prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_loglogisticpo.predict_survival_loglogistic_po(res_loglogPO.params,survivaldata,t))
plt.plot(t,mod_loglogisticpo.predict_survival_loglogistic_po_cis(res_loglogPO.params,res_loglogPO.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_loglogisticpo.predict_survival_loglogistic_po_cis(res_loglogPO.params,res_loglogPO.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Loglogistic PO')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('LoglogisticPH.png',dpi = 300)

def kmqqplot(array1,array2):
    t1 = []
    t2 = []
    foundt1 = False
    foundt2 = False
    current1index = 0
    current2index = 0
    minpercentile = max(float(array1[[0]].min(axis = 0)),float(array2[[0]].min(axis = 0)))
    percentiles = np.linspace(1,minpercentile,11)
    for x in percentiles:
        foundt1 = False
        foundt2 = False
        for y in xrange(current1index,len(array1)):
            if array1.iloc[y,0] <= x and foundt1 == False:            
                current1index = y
                t1.append(array1.index.values[y])
                foundt1 = True
        for g in xrange(current2index,len(array2)):
            if array2.iloc[g,0] <= x and foundt2 == False:
                t2.append(array2.index.values[g])
                current1index = g
                foundt2 = True
    plt.figure()
    minlength = min(len(t1),len(t2))
    plt.scatter(t1[:minlength],t2[:minlength])
    t1_cons = st.add_constant(t1[:minlength],prepend = False)
    linearmodel = st.OLS(t2[:minlength],t1_cons)
    linearmodelresults = linearmodel.fit()
    plt.plot(t1[:minlength],linearmodelresults.fittedvalues,'g')
    plt.xticks(np.arange(0, max(t1[:minlength]) + 5, 4.0))
    plt.yticks(np.arange(0, max(t2[:minlength]) + 5, 4.0))
    plt.xlim(0,max(t1[:minlength]) + 4)
    plt.ylim(0,max(t2[:minlength]) + 4)
    plt.xlabel(array1.columns.values[0])
    plt.ylabel(array2.columns.values[0])

#Fit Kaplan Meier curves for each group, then plot times from different groups against each other
kmf_regime = KaplanMeierFitter()
kmf_regime.fit(timevar[regime],event_observed = eventvar[regime],label = "Regime change")
kmf_territorial = KaplanMeierFitter()
kmf_territorial.fit(timevar[territorial],event_observed = eventvar[territorial],label = "Territorial")
kmf_policy = KaplanMeierFitter()
kmf_policy.fit(timevar[policy],event_observed = eventvar[policy],label = "Policy")
kmf_empire = KaplanMeierFitter()
kmf_empire.fit(timevar[empire],event_observed = eventvar[empire],label = "Empire")
kmf_social = KaplanMeierFitter()
kmf_social.fit(timevar[social],event_observed = eventvar[social],label = "Social")
kmf_status = KaplanMeierFitter()
kmf_status.fit(timevar[status],event_observed = eventvar[status],label = "Status")

kmqqplot(kmf_regime.survival_function_,kmf_territorial.survival_function_)
kmqqplot(kmf_regime.survival_function_,kmf_policy.survival_function_)
kmqqplot(kmf_regime.survival_function_,kmf_empire.survival_function_)
kmqqplot(kmf_regime.survival_function_,kmf_social.survival_function_)
kmqqplot(kmf_regime.survival_function_,kmf_status.survival_function_)

kmf_tenthousands = KaplanMeierFitter()
kmf_tenthousands.fit(timevar[tenthousands],event_observed = eventvar[tenthousands],label = "Ten thousands")
kmf_tens = KaplanMeierFitter()
kmf_tens.fit(timevar[tens],event_observed = eventvar[tens],label = "Tens")
kmf_hundreds = KaplanMeierFitter()
kmf_hundreds.fit(timevar[hundreds],event_observed = eventvar[hundreds],label = "Hundreds")
kmf_thousands = KaplanMeierFitter()
kmf_thousands.fit(timevar[thousands],event_observed = eventvar[thousands],label = "Thousands")

kmqqplot(kmf_tenthousands.survival_function_,kmf_tens.survival_function_)
kmqqplot(kmf_tenthousands.survival_function_,kmf_hundreds.survival_function_)
kmqqplot(kmf_tenthousands.survival_function_,kmf_thousands.survival_function_)

kmf_low = KaplanMeierFitter()
kmf_low.fit(timevar[low],event_observed = eventvar[low],label = "Low")
kmf_lowermiddle = KaplanMeierFitter()
kmf_lowermiddle.fit(timevar[lowermiddle],event_observed = eventvar[lowermiddle],label = "Lower middle")
kmf_uppermiddle = KaplanMeierFitter()
kmf_uppermiddle.fit(timevar[uppermiddle],event_observed = eventvar[uppermiddle],label = "Upper middle")
kmf_high = KaplanMeierFitter()
kmf_high.fit(timevar[high],event_observed = eventvar[high],label = "High")

kmqqplot(kmf_high.survival_function_,kmf_low.survival_function_)
kmqqplot(kmf_high.survival_function_,kmf_lowermiddle.survival_function_)
kmqqplot(kmf_high.survival_function_,kmf_uppermiddle.survival_function_)

kmf_free = KaplanMeierFitter()
kmf_free.fit(timevar[free],event_observed = eventvar[free],label = "Free")
kmf_partlyfree = KaplanMeierFitter()
kmf_partlyfree.fit(timevar[partlyfree],event_observed = eventvar[partlyfree],label = "Partly free")
kmf_notfree = KaplanMeierFitter()
kmf_notfree.fit(timevar[notfree],event_observed = eventvar[notfree],label = "Not free")

kmqqplot(kmf_free.survival_function_,kmf_partlyfree.survival_function_)
kmqqplot(kmf_free.survival_function_,kmf_notfree.survival_function_)

kmf_left_wing = KaplanMeierFitter()
kmf_left_wing.fit(timevar[left_wing],event_observed = eventvar[left_wing],label = "Left wing")
kmf_nationalist = KaplanMeierFitter()
kmf_nationalist.fit(timevar[nationalist],event_observed = eventvar[nationalist],label = "Nationalist")
kmf_right_wing = KaplanMeierFitter()
kmf_right_wing.fit(timevar[right_wing],event_observed = eventvar[right_wing],label = "Right wing")
kmf_religious = KaplanMeierFitter()
kmf_religious.fit(timevar[religious],event_observed = eventvar[religious],label = "Religious")

kmqqplot(kmf_left_wing.survival_function_,kmf_nationalist.survival_function_)
kmqqplot(kmf_left_wing.survival_function_,kmf_right_wing.survival_function_)
kmqqplot(kmf_left_wing.survival_function_,kmf_religious.survival_function_)   

def _ll_exponentialaft(y,X,beta):
    scale = np.exp(-np.dot(X,beta))
    ll = eventvar * np.log(scale) - (scale * y)    
    return ll 

class ExponentialAFT(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(ExponentialAFT, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        ll = _ll_exponentialaft(self.endog,self.exog,params)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if start_params == None:
            start_params = start_params_var 
        return super(ExponentialAFT, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#ExponentialAFT
start_params_var = np.repeat(0,len(survivaldata.columns))
mod_exponentialaft = ExponentialAFT(timevar,survivaldata)
res_expAFT = mod_exponentialaft.fit()
print(res_expAFT.summary())

#Plot the exponential AFT prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_exponentialaft.predict_survival_exponential_aft(res_expAFT.params,survivaldata,t))
plt.plot(t,mod_exponentialaft.predict_survival_exponential_aft_cis(res_expAFT.params,res_expAFT.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_exponentialaft.predict_survival_exponential_aft_cis(res_expAFT.params,res_expAFT.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Exponential AFT')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('ExponentialAFT.png',dpi = 300)

def _ll_weibullaft(y,X,beta,gamma):
    scale = np.exp(-1 * np.dot(X,beta) * gamma)
    ll = eventvar * np.log(scale * gamma * np.power(y, (gamma - 1))) - (scale * np.power(y,gamma))
    return ll 
   
class WeibullAFT(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(WeibullAFT, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_weibullaft(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(WeibullAFT, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#WeibullAFT
#Set the initial gamma value to 1
start_params_var = np.repeat(1,len(survivaldata.columns) + 1)
mod_weibullaft = WeibullAFT(timevar,survivaldata)
res_weiAFT = mod_weibullaft.fit()
print(res_weiAFT.summary())

#Plot the weibull AFT prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_weibullaft.predict_survival_weibull_aft(res_weiAFT.params,survivaldata,t))
plt.plot(t,mod_weibullaft.predict_survival_weibull_aft_cis(res_weiAFT.params,res_weiAFT.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_weibullaft.predict_survival_weibull_aft_cis(res_weiAFT.params,res_weiAFT.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Weibull AFT')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('WeibullAFT.png',dpi = 300)

def _ll_loglogisticaft(y,X,beta,gamma):
    scale = np.exp(- np.dot(X,beta) * gamma)
    ll = eventvar * np.log((scale * gamma * np.power(y,gamma - 1))/(1 + scale * np.power(y,gamma))) + np.log(1/(1 + scale * np.power(y,gamma)))    
    return ll

class LoglogisticAFT(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(LoglogisticAFT, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_loglogisticaft(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(LoglogisticAFT, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#LoglogisticAFT
#Set the initia gamma value to 1
start_params_var = np.repeat(1,len(survivaldata.columns) + 1)
mod_loglogisticaft = LoglogisticAFT(timevar,survivaldata)
res_loglogAFT = mod_loglogisticaft.fit()
print(res_loglogAFT.summary())

#Plot the loglogistic AFT prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_loglogisticaft.predict_survival_loglogistic_aft(res_loglogAFT.params,survivaldata,t))
plt.plot(t,mod_loglogisticaft.predict_survival_loglogistic_aft_cis(res_loglogAFT.params,res_loglogAFT.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_loglogisticaft.predict_survival_loglogistic_aft_cis(res_loglogAFT.params,res_loglogAFT.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Loglogistic AFT')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('LoglogisticAFT.png',dpi = 300)

def _ll_lognormalaft(y,X,beta,gamma):
    scale = np.dot(X,beta)
    ll = eventvar * (
    np.log(
    norm.pdf(((np.log(y) - scale) * gamma))/
    (y * (1/gamma) * (1 - norm.cdf((np.log(y) - scale) * gamma))))
    ) + np.log(1 - norm.cdf((np.log(y) - scale) * gamma))
    return ll

class LognormalAFT(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(LognormalAFT, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        gamma = params[-1]
        beta = params[:-1]
        ll = _ll_lognormalaft(self.endog,self.exog,beta,gamma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('gamma')        
        if start_params == None:
            start_params = start_params_var 
        return super(LognormalAFT, self).fit(start_params = start_params,method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#LognormalAFT
start_params_var = np.repeat(1,len(survivaldata.columns) + 1)
mod_lognormalaft = LognormalAFT(timevar,survivaldata)
res_lognormAFT = mod_lognormalaft.fit()
print(res_lognormAFT.summary())


#Plot the lognormal AFT prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(0,150,151)
plt.plot(t,mod_lognormalaft.predict_survival_lognormal_aft(res_lognormAFT.params,survivaldata,t))
plt.plot(t,mod_lognormalaft.predict_survival_lognormal_aft_cis(res_lognormAFT.params,res_lognormAFT.cov_params(),survivaldata,t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_lognormalaft.predict_survival_lognormal_aft_cis(res_lognormAFT.params,res_lognormAFT.cov_params(),survivaldata,t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Lognormal AFT')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('LognormalAFT.png',dpi = 300)

def _ll_generalizedgammaaft(y,X,beta,kappa,sigma):
    scale = np.dot(X,beta)
    gammavalue = np.power(np.abs(kappa),-2)
    zeta = np.sign(kappa) * (np.log(y) - scale) / sigma
    upsilon = gammavalue * np.exp(np.abs(kappa)*zeta)        
    if kappa > 0:
        density = np.power(gammavalue,gammavalue) * np.exp(zeta * np.power(gammavalue,0.5) - upsilon) / (sigma * y * np.power(gammavalue,0.5) * gammafunction(gammavalue))
        survival = 1 - gammainc(gammavalue,upsilon)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    elif kappa == 0: 
        density = np.exp(- np.power(zeta,2) / 2) / (sigma * y * np.power(2 * np.pi,0.5))
        survival = 1 - norm.cdf(zeta)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    elif kappa < 0:
        density = np.power(gammavalue,gammavalue) * np.exp(zeta * np.power(gammavalue,0.5) - upsilon) / (sigma * y * np.power(gammavalue,0.5) * gammafunction(gammavalue))
        survival = gammainc(gammavalue,upsilon)
        ll = eventvar * np.log(density / survival) + np.log(survival)
    return ll
    
class GeneralizedgammaAFT(GenericLikelihoodModel):
    def _init_(self,endog,exog,**kwds):
        super(GeneralizedgammaAFT, self).__init__(endog,exog,**kwds)
    def nloglikeobs(self,params):
        beta = params[:-2]
        kappa = params[-2]
        sigma = params[-1]
        ll = _ll_generalizedgammaaft(self.endog,self.exog,beta,kappa,sigma)
        return -ll
    def fit(self, start_params = None, maxiter = 10000,maxfun = 5000,**kwds):
        if len(self.exog_names) == len(self.exog[1]):
            self.exog_names.append('kappa')
            self.exog_names.append('sigma')
        if start_params == None:
            start_params = start_params_var 
        return super(GeneralizedgammaAFT, self).fit(start_params = start_params, method = 'ncg',maxiter = maxiter, maxfun = maxfun,**kwds)

#Generalized gamma AFT
start_params_var = np.repeat(1,len(survivaldata.columns) + 2)
mod_generalizedgammaaft = GeneralizedgammaAFT(timevar,survivaldata)
res_generalizedgammaAFT = mod_generalizedgammaaft.fit()
print(res_generalizedgammaAFT.summary())
#Plot the generalized gamma aft prediction against the empirical survival curve
plt.figure()
ax = plt.subplot(1,1,1)
t = np.linspace(1,150,150)
plt.plot(t,mod_generalizedgammaaft.predict_survival_generalizedgamma_aft(res_generalizedgammaAFT.params, survivaldata, t))
plt.plot(t,mod_generalizedgammaaft.predict_survival_generalizedgamma_aft_cis(res_generalizedgammaAFT.params, res_generalizedgammaAFT.cov_params(), survivaldata, t)[[1]],'r--',linewidth = 1.0)
plt.plot(t,mod_generalizedgammaaft.predict_survival_generalizedgamma_aft_cis(res_generalizedgammaAFT.params, res_generalizedgammaAFT.cov_params(), survivaldata, t)[[2]],'r--',linewidth = 1.0)
kmf.plot(ax = ax)
plt.title('Generalized gamma AFT')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.savefig('GeneralizedgammaAFT.png',dpi = 300)

