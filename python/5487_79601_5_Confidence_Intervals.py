#The following is added to the prediction class in Model.py(object):    
def predict_survival_exponential_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    XB = np.dot(X,params)
    for time in timerange:
        predicted_survival_units.append(np.exp(-np.exp(-XB) * time))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_exponential_aft_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for unit in range(len(X)):
            covariates = list(X.iloc[unit])
            XB = np.dot(covariates,params)
            sqrt = np.sqrt(np.dot(np.dot(covariates,covar),covariates))
            lower = np.power(np.exp(-np.exp(-XB) * time),np.exp( 1.96 * sqrt))
            lower_bound.append(lower)
            upper = np.power(np.exp(-np.exp(-XB) * time),np.exp(-1.96 * sqrt))         
            upper_bound.append(upper)
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame    

def predict_survival_weibull_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    XBg = np.dot(X,params[:-1]) * params[-1]
    for time in timerange:
        predicted_survival_units.append(np.exp(-np.exp(-XBg) * np.power(time,params[-1])))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_weibull_aft_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    gamma = params[-1]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            XB = np.dot(covariates,params[0:(len(params) - 1)])                
            Xgamma = np.dot(covariates,gamma)                
            derivativevector = np.append(Xgamma,np.log(np.maximum(1,time)) - XB)
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(np.exp(-np.exp(-XB * gamma) * np.power(time,gamma)),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(np.exp(-np.exp(-XB * gamma) * np.power(time,gamma)),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame


def predict_survival_loglogistic_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    XBg = np.dot(X,beta) * gamma
    for time in timerange:
        predicted_survival_units.append(1/(1 + (np.exp(-XBg) * np.power(time,gamma))))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_loglogistic_aft_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            Xgamma = np.dot(covariates,gamma)
            XB = np.dot(covariates,beta)
            XBg = XB * gamma
            tg = np.power(time,gamma)
            multiplierbetas = (
            (tg*np.exp(-XBg))/
            (np.log(1+np.exp(-XBg)*tg)*(1+np.exp(-XBg)*tg))
            )
            Xmultipliers = np.dot(-Xgamma,multiplierbetas)
            multipliergamma = (
            (tg*np.exp(-XBg)) * (np.log(time)-XB) /                
            (np.log(1+np.exp(-XBg)*tg)*(1+np.exp(-XBg)*tg))                
            )
            derivativevector = np.append(Xmultipliers,multipliergamma)
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(1/(1 + (np.exp(-XBg) * np.power(time,gamma))),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(1/(1 + (np.exp(-XBg) * np.power(time,gamma))),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame

def predict_survival_lognormal_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    XB = np.dot(X,beta)
    for time in timerange:
        predicted_survival_units.append(1 - norm.cdf((np.log(time) - XB) * gamma))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_lognormal_aft_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            Xgamma = np.dot(covariates,gamma)
            XB = np.dot(covariates,beta)
            normalargument = (np.log(time) - XB) * gamma
            multiplier = (
            norm.pdf(normalargument)/                
            (np.log(1 - norm.cdf(normalargument))*(1-norm.cdf(normalargument)))
            )
            multiplierbetas = np.dot(-Xgamma, multiplier) 
            multipliergamma = np.dot((np.log(time) - XB), multiplier) 
            derivativevector = np.append(multiplierbetas,multipliergamma)
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(1 - norm.cdf(normalargument),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(1 - norm.cdf(normalargument),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame

def predict_survival_generalizedgamma_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    scale = np.dot(X,params[:-2])
    kappa = params[-2]
    sigma = params[-1]
    gamma = np.power(np.abs(kappa),-2)
    if kappa > 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival_units.append(1 - gammainc(gamma,upsilon))
    if kappa == 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            predicted_survival_units.append(1 - norm.cdf(zeta))
    if kappa < 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival_units.append(gammainc(gamma,upsilon))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival    


def predict_survival_generalizedgamma_aft(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    scale = np.dot(X,params[:-2])
    kappa = params[-2]
    sigma = params[-1]
    gamma = np.power(np.abs(kappa),-2)
    if kappa > 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival_units.append(1 - gammainc(gamma,upsilon))
    if kappa == 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            predicted_survival_units.append(1 - norm.cdf(zeta))
    if kappa < 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival_units.append(gammainc(gamma,upsilon))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival    
   
def predict_survival_generalizedgamma_aft_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    beta = params[:-2]
    kappa = params[-2]
    sigma = params[-1]
    gamma = np.power(np.abs(kappa),-2)        
    lower_bound = []
    upper_bound = []
    if kappa > 0:
        for time in timerange:
            lower_bound = []
            upper_bound = []
            for patient in range(len(X)):
                covariates = list(X.iloc[patient])
                scale = np.dot(covariates,beta)
                zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
                upsilon = gamma * np.exp(np.abs(kappa)*zeta)
                incompletegamma = gammafunction(gamma) - (gammainc(gamma,upsilon) * gammafunction(gamma))
                #d(log-(log(S(t))))/dx:
                numerator = np.abs(kappa) * np.sign(kappa) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) / sigma) - upsilon)
                denominator =  np.power(kappa,2) * sigma * incompletegamma * np.log(incompletegamma/gammafunction(gamma))    
                dsurvdscale = numerator / denominator
                dsurvdscale = np.dot(dsurvdscale,covariates)
                #d(log-(log(S(t))))/dkappa:
                term11 = (-1/np.power(kappa,3)) * 2 * (mpmath.meijerg([[],[1,1]],[[0,0,gamma],[]],upsilon) + np.log(upsilon) * incompletegamma)
                term12 = np.exp(-upsilon) * np.power(upsilon,gamma-1) * (upsilon * ((2 * np.abs(kappa) * DiracDelta(kappa) * ((np.log(time) - scale)/sigma)) + (kappa * np.sign(kappa) * (np.log(time) - scale)/(sigma * np.abs(kappa)))) - (2 * np.exp(np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) / np.power(kappa,3)))
                term1 = (-1/gammafunction(gamma)) * (term11 - term12)
                term2 = 2 * psi(gamma) * incompletegamma / (np.power(kappa,3) * gammafunction(gamma))                             
                numerator = gammafunction(gamma) * (term1 + term2)                 
                denominator = incompletegamma * np.log(incompletegamma/gammafunction(gamma))
                dsurvdkappa = numerator / denominator
                #d(log-(log(S(t))))/dsigma:
                numerator = np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) - upsilon)
                denominator = np.power(kappa,2) * np.power(sigma,2) * incompletegamma * np.log(incompletegamma/gammafunction(gamma))               
                dsurvdsigma = numerator / denominator
                #vector of derivatives of d(log-(log(S(t)))) 
                derivativevector = np.append(dsurvdscale,float(dsurvdkappa))
                derivativevector = np.append(derivativevector,dsurvdsigma)          
                sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector)) 
                lower_bound.append(np.power(1 - gammainc(gamma,upsilon),np.exp( 1.96 * sqrt)))
                upper_bound.append(np.power(1 - gammainc(gamma,upsilon),np.exp(-1.96 * sqrt)))
            lower_all_units.append(np.mean(lower_bound)) 
            upper_all_units.append(np.mean(upper_bound))     
    if kappa == 0:
         for time in timerange:
            lower_bound = []
            upper_bound = []
            for patient in range(len(X)):
                covariates = list(X.iloc[patient])
                scale = np.dot(covariates,beta)
                zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
                term1 = -norm.cdf(zeta) / (np.log(1 - norm.cdf(zeta)) * (1 - norm.cdf(zeta)))
                #d(log-(log(S(t))))/dx:
                dsurvdscale = term1 * (- np.sign(kappa)/sigma)
                dsurvdscale = np.dot(dsurvdscale,covariates)
                #d(log-(log(S(t))))/dkappa:
                dsurvdkappa = term1 * ((2 * DiracDelta(kappa) * np.log(time) - scale) / sigma) 
                #d(log-(log(S(t))))/dsigma:
                dsurvdsigma = term1 * (np.sign(kappa) * (scale - np.log * (time))/np.power(sigma,2))
                derivativevector = np.append(dsurvdscale,float(dsurvdkappa))
                derivativevector = np.append(derivativevector,dsurvdsigma)
                sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))    
                lower_bound.append(np.power(1 - norm.cdf(zeta)),np.exp( 1.96 * sqrt))
                upper_bound.append(np.power(1 - norm.cdf(zeta)),np.exp(-1.96 * sqrt))
            lower_all_units.append(np.mean(lower_bound)) 
            upper_all_units.append(np.mean(upper_bound))    
    if kappa < 0:
        for time in timerange:
            lower_bound = []
            upper_bound = []
            for patient in range(len(X)):
                covariates = list(X.iloc[patient])
                scale = np.dot(covariates,beta)
                zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
                upsilon = gamma * np.exp(np.abs(kappa)*zeta)
                incompletegamma = gammafunction(gamma) - (gammainc(gamma,upsilon) * gammafunction(gamma))
                #d(log-(log(S(t))))/dx:
                numerator = -np.abs(kappa) * np.sign(kappa) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) / sigma) - upsilon)
                denominator =  np.power(kappa,2) * sigma * gammafunction(gamma) * (1 - incompletegamma/gammafunction(gamma)) * np.log(1 - incompletegamma/gammafunction(gamma))    
                dsurvdscale = numerator / denominator
                dsurvdscale = np.dot(dsurvdscale,covariates)
                #d(log-(log(S(t))))/dkappa:
                term11 = (-1/np.power(kappa,3)) * 2 * (mpmath.meijerg([[],[1,1]],[[0,0,gamma],[]],upsilon) + np.log(upsilon) * incompletegamma)
                term12 = np.exp(-upsilon) * np.power(upsilon,gamma-1) * (upsilon * ((2 * np.abs(kappa) * DiracDelta(kappa) * ((np.log(time) - scale)/sigma)) + (kappa * np.sign(kappa) * (np.log(time) - scale)/(sigma * np.abs(kappa)))) - (2 * np.exp(np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) / np.power(kappa,3)))
                term1 = (-1/gammafunction(gamma)) * (term11 - term12)
                term2 = 2 * psi(gamma) * incompletegamma / (np.power(kappa,3) * gammafunction(gamma))                             
                numerator = term1 - term2                 
                denominator = (1 - (incompletegamma/gammafunction(gamma))) * np.log(1 - (incompletegamma/gammafunction(gamma)))
                dsurvdkappa = numerator / denominator
                #d(log-(log(S(t))))/dsigma:
                numerator = -np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) - upsilon)
                denominator = np.power(kappa,2) * np.power(sigma,2) * gammafunction(gamma) * (1 - incompletegamma/gammafunction(gamma)) * (np.log(1 - incompletegamma/gammafunction(gamma)))               
                dsurvdsigma = numerator / denominator
                derivativevector = np.append(dsurvdscale,float(dsurvdkappa))
                derivativevector = np.append(derivativevector,dsurvdsigma)
                sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
                lower_bound.append(np.power(gammainc(gamma,upsilon),np.exp( 1.96 * sqrt)))
                upper_bound.append(np.power(gammainc(gamma,upsilon),np.exp(-1.96 * sqrt)))
            lower_all_units.append(np.mean(lower_bound)) 
            upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame    
 
def predict_survival_exponential(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    for time in timerange:
        predicted_survival.append(np.exp(-params[0] * time))
    return predicted_survival

def predict_survival_exponential_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_bound = []
    upper_bound = []
    #Standard deviation of log log transformation of exponential survival
    #does not depend on time. We calcualte it only once
    sqrt = (covar / params)
    for time in timerange:
        lower_bound.append(np.power(np.exp(-params[0] * time),np.exp( 1.96 * sqrt)))#np.exp(-np.exp(np.log(params) + np.log(time) + 1.96 * (covar / params)))) 
        upper_bound.append(np.power(np.exp(-params[0] * time),np.exp(-1.96 * sqrt)))#np.exp(-np.exp(np.log(params) + np.log(time) - 1.96 * (covar / params))))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame    


def predict_survival_weibull(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    for time in timerange:
        predicted_survival.append(np.exp(-params[0] * np.power(time, params[1])))
    return predicted_survival

def predict_survival_weibull_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_bound = []
    upper_bound = []
    for time in timerange:
        sqrt = np.sqrt((covar.item(0,0)/np.power(params[0],2))+np.power(np.log(time),2) * covar.item(1,1) + (2 * np.log(time) * covar.item(0,1))/params[0])
        lower_bound.append(np.power(np.exp(-params[0] * np.power(time, params[1])),np.exp( 1.96 * sqrt))) 
        upper_bound.append(np.power(np.exp(-params[0] * np.power(time, params[1])),np.exp(-1.96 * sqrt)))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame        


def predict_survival_gompertz(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    for time in timerange:
        predicted_survival.append(np.exp((-np.exp(params[0]) * (np.exp(params[1] * time) - 1))/params[1]))
    return predicted_survival

def predict_survival_gompertz_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_bound = []
    upper_bound = []
    for time in timerange:
        constantterm = ((params[1] * time * np.exp(params[1] * time)) - np.exp(params[1] * time) + 1) / ((np.exp(params[1] * time) - 1) * params[1])
        sqrt = np.sqrt(covar.item(0,0) + np.power(constantterm,2) * covar.item(1,1) + 2 * constantterm * covar.item(0,1))
        lower_bound.append(np.power((np.exp((-np.exp(params[0]) * (np.exp(params[1] * time) - 1))/params[1])),np.exp( 1.96 * sqrt)))
        upper_bound.append(np.power((np.exp((-np.exp(params[0]) * (np.exp(params[1] * time) - 1))/params[1])),np.exp(-1.96 * sqrt)))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame    



def predict_survival_loglogistic(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    for time in timerange:
        predicted_survival.append(1/(1 + (params[0] * np.power(time,params[1]))))
    return predicted_survival

def predict_survival_loglogistic_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_bound = []
    upper_bound = []
    for time in timerange:
        constantterm = (np.power(time,params[1])/((1 + params[0] * np.power(time, params[1])) * (np.log(1 + params[0] * np.power(time,params[1])))))
        sqrt = np.sqrt(
        np.power(constantterm,2) * covar.item(0,0) + 
        np.power(constantterm * params[0] * np.log(time),2) * covar.item(1,1) + 
        2 * np.power(constantterm, 2) * params[0] * np.log(time) * covar.item(0,1)
        )
        lower_bound.append(np.power(1/(1 + (params[0] * np.power(time,params[1]))),np.exp( 1.96 * sqrt)))
        upper_bound.append(np.power(1/(1 + (params[0] * np.power(time,params[1]))),np.exp(-1.96 * sqrt)))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame  



def predict_survival_lognormal(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    for time in timerange:
        predicted_survival.append(1 - norm.cdf((np.log(time) - params[0]) * params[1]))
    return predicted_survival

def predict_survival_lognormal_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_bound = []
    upper_bound = []
    for time in timerange:
        constant = (np.log(time) - params[0]) * params[1]
        constantterm = norm.pdf(constant) / (np.log(1 - norm.cdf(constant)) * (1 - norm.cdf(constant)))
        sqrt = np.sqrt(
        np.power(constantterm * params[1], 2) * covar.item(0,0) + 
        np.power(constantterm * (np.log(time) - params[1]), 2) * covar.item(1,1) + 
        2 * np.power(constantterm, 2) * params[1] * (np.log(time) - params[0]) * covar.item(0,1))
        lower_bound.append(np.power(1 - norm.cdf((np.log(time) - params[0]) * params[1]),np.exp( 1.96 * sqrt)))
        upper_bound.append(np.power(1 - norm.cdf((np.log(time) - params[0]) * params[1]),np.exp(-1.96 * sqrt)))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame  


def predict_survival_generalizedgamma(self, params, timerange, *args, **kwargs):
    predicted_survival = []
    scale = params[0]
    kappa = params[1]
    sigma = params[2]
    gamma = np.power(np.abs(kappa),-2)
    if kappa > 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival.append(1 - gammainc(gamma,upsilon))
    if kappa == 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            predicted_survival.append(1 - norm.cdf(zeta))
    if kappa < 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            predicted_survival.append(gammainc(gamma,upsilon))
    return predicted_survival
    
   
def predict_survival_generalizedgamma_cis(self, params, covar, timerange, *args, **kwargs):
    interval_data_frame = pd.DataFrame() 
    scale = params[0]
    kappa = params[1]
    sigma = params[2]
    gamma = np.power(np.abs(kappa),-2)        
    lower_bound = []
    upper_bound = []
    if kappa > 0:
        for time in timerange:
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            incompletegamma = gammafunction(gamma) - (gammainc(gamma,upsilon) * gammafunction(gamma))
            #d(log-(log(S(t))))/dscale:
            numerator = np.abs(kappa) * np.sign(kappa) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) / sigma) - upsilon)
            denominator =  np.power(kappa,2) * sigma * incompletegamma * np.log(incompletegamma/gammafunction(gamma))    
            dsurvdscale = numerator / denominator
            #d(log-(log(S(t))))/dkappa:
            term11 = (-1/np.power(kappa,3)) * 2 * (mpmath.meijerg([[],[1,1]],[[0,0,gamma],[]],upsilon) + np.log(upsilon) * incompletegamma)
            term12 = np.exp(-upsilon) * np.power(upsilon,gamma-1) * (upsilon * ((2 * np.abs(kappa) * DiracDelta(kappa) * ((np.log(time) - scale)/sigma)) + (kappa * np.sign(kappa) * (np.log(time) - scale)/(sigma * np.abs(kappa)))) - (2 * np.exp(np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) / np.power(kappa,3)))
            term1 = (-1/gammafunction(gamma)) * (term11 - term12)
            term2 = 2 * psi(gamma) * incompletegamma / (np.power(kappa,3) * gammafunction(gamma))                             
            numerator = gammafunction(gamma) * (term1 + term2)                 
            denominator = incompletegamma * np.log(incompletegamma/gammafunction(gamma))
            dsurvdkappa = numerator / denominator
            #d(log-(log(S(t))))/dsigma:
            numerator = np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) - upsilon)
            denominator = np.power(kappa,2) * np.power(sigma,2) * incompletegamma * np.log(incompletegamma/gammafunction(gamma))               
            dsurvdsigma = numerator / denominator
            #vector of derivatives of d(log-(log(S(t)))) 
            derivativevector = [dsurvdscale,dsurvdkappa,dsurvdsigma]           
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector)) 
            lower_bound.append(np.power(1 - gammainc(gamma,upsilon),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(1 - gammainc(gamma,upsilon),np.exp(-1.96 * sqrt)))
    if kappa == 0:
        for time in timerange:    
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            term1 = -norm.cdf(zeta) / (np.log(1 - norm.cdf(zeta)) * (1 - norm.cdf(zeta)))
            dsurvdscale = term1 * (- np.sign(kappa)/sigma)
            dsurvdkappa = term1 * ((2 * DiracDelta(kappa) * np.log(time) - scale) / sigma) 
            dsurvdsigma = term1 * (np.sign(kappa) * (scale - np.log * (time))/np.power(sigma,2))
            derivativevector = [dsurvdscale,dsurvdkappa,dsurvdsigma]
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))    
            lower_bound.append(np.power(1 - norm.cdf(zeta)),np.exp( 1.96 * sqrt))
            upper_bound.append(np.power(1 - norm.cdf(zeta)),np.exp(-1.96 * sqrt))
    if kappa < 0:
        for time in timerange:  
            zeta = np.sign(kappa) * (np.log(time) - scale) / sigma
            upsilon = gamma * np.exp(np.abs(kappa)*zeta)
            incompletegamma = gammafunction(gamma) - (gammainc(gamma,upsilon) * gammafunction(gamma))
            #d(log-(log(S(t))))/dscale:
            numerator = -np.abs(kappa) * np.sign(kappa) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) / sigma) - upsilon)
            denominator =  np.power(kappa,2) * sigma * gammafunction(gamma) * (1 - incompletegamma/gammafunction(gamma)) * np.log(1 - incompletegamma/gammafunction(gamma))    
            dsurvdscale = numerator / denominator
            #d(log-(log(S(t))))/dkappa:
            term11 = (-1/np.power(kappa,3)) * 2 * (mpmath.meijerg([[],[1,1]],[[0,0,gamma],[]],upsilon) + np.log(upsilon) * incompletegamma)
            term12 = np.exp(-upsilon) * np.power(upsilon,gamma-1) * (upsilon * ((2 * np.abs(kappa) * DiracDelta(kappa) * ((np.log(time) - scale)/sigma)) + (kappa * np.sign(kappa) * (np.log(time) - scale)/(sigma * np.abs(kappa)))) - (2 * np.exp(np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) / np.power(kappa,3)))
            term1 = (-1/gammafunction(gamma)) * (term11 - term12)
            term2 = 2 * psi(gamma) * incompletegamma / (np.power(kappa,3) * gammafunction(gamma))                             
            numerator = term1 - term2                 
            denominator = (1 - (incompletegamma/gammafunction(gamma))) * np.log(1 - (incompletegamma/gammafunction(gamma)))
            dsurvdkappa = numerator / denominator
            #d(log-(log(S(t))))/dsigma:
            numerator = -np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale) * np.power(upsilon,gamma - 1) * np.exp((np.abs(kappa) * np.sign(kappa) * (np.log(time) - scale)/sigma) - upsilon)
            denominator = np.power(kappa,2) * np.power(sigma,2) * gammafunction(gamma) * (1 - incompletegamma/gammafunction(gamma)) * (np.log(1 - incompletegamma/gammafunction(gamma)))               
            dsurvdsigma = numerator / denominator
            derivativevector = [dsurvdscale,float(dsurvdkappa),dsurvdsigma]
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(gammainc(gamma,upsilon),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(gammainc(gamma,upsilon),np.exp(-1.96 * sqrt)))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_bound
    interval_data_frame['upper'] = upper_bound
    return interval_data_frame    


def predict_survival_exponential_ph(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    XB = np.dot(X,params)
    for time in timerange:
        predicted_survival_units.append(np.exp(-np.exp(XB) * time))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_exponential_ph_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for unit in range(len(X)):
            covariates = list(X.iloc[unit])
            XB = np.dot(covariates,params)
            sqrt = np.sqrt(np.dot(np.dot(covariates,covar),covariates))
            lower = np.power(np.exp(-np.exp(XB) * time),np.exp( 1.96 * sqrt))#np.exp(-np.exp(XB + 1.96 * sqrt + np.log(time))) 
            lower_bound.append(lower)
            upper = np.power(np.exp(-np.exp(XB) * time),np.exp(-1.96 * sqrt))#np.exp(-np.exp(XB - 1.96 * sqrt + np.log(time)))                                 
            upper_bound.append(upper)
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame    

   
def predict_survival_weibull_ph(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    XB = np.dot(X,params[:-1])
    for time in timerange:
        predicted_survival_units.append(np.exp(-np.exp(XB) * np.power(time,params[-1])))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_weibull_ph_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    beta = params[:-1]
    gamma = params[-1]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            XB = np.dot(covariates,beta)
            derivativevector = np.append(covariates,np.log(np.maximum(1,time)))
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(np.exp(-np.exp(XB) * np.power(time,gamma)),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(np.exp(-np.exp(XB) * np.power(time,gamma)),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame   
   

    
def predict_survival_gompertz_ph(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    XB = np.dot(X,params[:-1])
    for time in timerange:
        predicted_survival_units.append(np.exp((-np.exp(XB) * (np.exp(params[-1] * time) - 1))/params[-1]))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_gompertz_ph_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    beta = params[:-1]
    gamma = params[-1]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            XB = np.dot(covariates,beta) 
            derivativevector = np.append(covariates,(time*np.exp(gamma*time)/(np.exp(gamma*time)-1))-(1/gamma))
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(np.exp((-np.exp(XB) * (np.exp(gamma * time) - 1))/gamma),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(np.exp((-np.exp(XB) * (np.exp(gamma * time) - 1))/gamma),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame         
    
def predict_survival_loglogistic_po(self, params, X, timerange,*args, **kwargs):
    predicted_survival_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    XB = np.dot(X,beta)
    for time in timerange:
        predicted_survival_units.append(1/(1 + (np.exp(XB) * np.power(time,gamma))))
    predicted_survival = []
    index = 0    
    for index in range(np.shape(predicted_survival_units)[0]):
        predicted_survival.append(np.mean(predicted_survival_units[index]))
    return predicted_survival

def predict_survival_loglogistic_po_cis(self,params, covar, X, timerange,*args, **kwargs):
    interval_data_frame = pd.DataFrame()    
    lower_all_units = []
    upper_all_units = []
    gamma = params[-1]
    beta = params[0:(len(params) - 1)]
    for time in timerange:
        lower_bound = []
        upper_bound = []
        for patient in range(len(X)):
            covariates = list(X.iloc[patient])
            XB = np.dot(covariates,beta)
            tg = np.power(time,gamma)
            multiplierbetas = (
            (tg*np.exp(XB))/
            (np.log(1+np.exp(XB)*tg)*(1+np.exp(XB)*tg))
            )
            Xmultipliers = np.dot(covariates,multiplierbetas)
            multipliergamma = (
            tg * np.exp(XB) * np.log(time) /                
            (np.log(1+np.exp(XB)*tg)*(1+np.exp(XB)*tg))                
            )
            derivativevector = np.append(Xmultipliers,multipliergamma)
            sqrt = np.sqrt(np.dot(np.dot(derivativevector,covar),derivativevector))
            lower_bound.append(np.power(1/(1 + (np.exp(XB) * np.power(time,gamma))),np.exp( 1.96 * sqrt)))
            upper_bound.append(np.power(1/(1 + (np.exp(XB) * np.power(time,gamma))),np.exp(-1.96 * sqrt)))
        lower_all_units.append(np.mean(lower_bound)) 
        upper_all_units.append(np.mean(upper_bound))
    interval_data_frame['time'] = timerange    
    interval_data_frame['lower'] = lower_all_units
    interval_data_frame['upper'] = upper_all_units
    return interval_data_frame 

