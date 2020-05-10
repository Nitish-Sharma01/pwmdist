import pandas as pd
import statistics as st
import numpy as  np
import random
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as si

def Fitbygpdpwm(data, ci=0.95, threshold= None):
    """
    1) Description:
    The function fits Generalized pareto distiribution to the passed dataset; a timeseries object using probability weighted moments method.
    
    2) Input Parameters:
        data: timeseries dataframe. 
        ci: confidence interval
        threshold: A float, A threshold number obtained from Peak over threshold method
        If the threshold value is passed than the manual threshold value is used else the quantile at a given
        confidence interval is used to calculate the threshold.

    3) Results:
        The function returns a dicitionary, which has data, list of residuals, probability, shape parameter, scale parameter, 
        list of exceedances (case where data[i]>threshold), threshold value, excess(case in exceedances subtracted from threshold.)
        This result dictionary can be used in the plot functions
    4) Example:
        the example of the "data" parameter is as follows:
        Date        log(return)
        25-12-2020  0.11098978
        26-12-2020  0.14787224

    """
    data=data.dropna()
    data=data.iloc[:,0].to_list()
    threshdf=pd.DataFrame(data)
    threshold = threshdf.quantile(ci,axis=0)[0] if threshold == None else threshold
    result_dict={}
    exceedances=[]
    for i in data:
        if i>threshold:
            exceedances.append(i)
        else:
            pass
    excess=[]
    for i in exceedances:
        excess.append(i-threshold)
    sorted_excess=sorted(excess)
    Nu=len(excess)
    a0=st.mean(excess)
    gamma = -0.35
    delta=0
    pvec=[(j+gamma)/(Nu+delta) for j in  [i for i in range(1,Nu+1)] ]
    a1= st.mean([sorted_excess[j]*[(1-i) for i in pvec][j] for j in range(0,len(sorted_excess))])
    xi=2-(a0/(a0-2*a1))
    beta= (2 * a0 * a1)/(a0 - 2 * a1)
    result_dict['Shape']=xi
    result_dict['Scale']=beta
    result_dict['prob']=1-len(exceedances)/len(data)
    result_dict['threshold']=threshold
    result_dict['exceedances']= exceedances
    result_dict['excess']=excess
    result_dict['residuals'] = np.log(1 + (xi * (np.array(exceedances)-threshold))/beta)/xi
    denom=  Nu*(1-2*xi)*(3-2*xi)

    if xi>0.5:
        denom =None
        result_dict['varcov']= None
        result_dict['Parameter Sensitivities']=None
        print("Asymptotic Standard Errors not available for PWM when xi>0.5.")
    else:
        one=(1 - xi) * (1 - xi + 2 * xi**2) * (2 - xi)**2
        two =(7 - 18 * xi + 11 * xi**2 - 2 * xi**3) * beta**2
        cov=  beta * (2 - xi) * (2 - 6 * xi + 7 * xi**2 - 2 * xi**3)
        varcov=np.array([[one,cov],[cov,two]])/denom
        result_dict['Parameter Sensitivities']=np.sqrt(varcov.diagonal()).flatten()
        result_dict['Variance covariance matrix']=varcov.flatten()
        result_dict['data']=data
    return result_dict



def gpdpwmFitCheck(data, ci=0.95, threshold=None):

    """
    1) Description:
        Checks the Fit of GPD with probability weighted moments

    2) Input Parameters:
        data= timeseries dataframe

        ci= confidence interval
        
        threhsold=A float, A threshold number obtained from Peak over threshold method
        If the threshold value is passed than the manual threshold value is used else the quantile at a given
        confidence interval is used to calculate the threshold.
    3) Results:
        A dictionary of parameter estimates, threshold and excess. 
    """
    data=data.dropna()
    x = np.array(data.iloc[:,0].to_list())
    threshdf=pd.DataFrame(data)
    threshold = threshdf.quantile(ci,axis=0)[0] if threshold == None else threshold
    result_check={}
    excess_check=[]
    for i in x:
        if i >threshold:
            excess_check.append(i-threshold)
        else:
            pass
    result_check['Excess']=excess_check
    Nu = len(excess_check)
    gamma=-0.35
    a0 = st.mean(excess_check)
    pvec=[(j+gamma)/(Nu) for j in  [i for i in range(1,Nu+1)]]
    sorted_excess=sorted(excess_check)
    a1=st.mean([sorted_excess[j]*[(1-i) for i in pvec][j] for j in range(0,len(sorted_excess))])
    xi = 2 - a0/(a0 - 2 * a1)
    beta = (2 * a0 * a1)/(a0 - 2 * a1)
    result_check['Shape'] = xi
    result_check['Scale']= beta
    result_check['threshold']=threshold
    return result_check


def gpdSimulation(shape= 0.25, location = 0, scale = 1, n = 1000, seed = None):
    """
    1) Description:
       Generates random variates from a GPD distribution

    2) Input parameters:
        shape, location, scale = the parameter estimates that can be either manually input or taken from Fitbygpdpwm function
        n = number of simulated observations
        seed = by default None
    3) Result:
        list of simulate values from generalized pareto distribution
    """
    # Seed:
    if seed is None:
        seed = None
    else:
        random.seed(seed)

    # Simulate:
    if min([scale])<0:   # ref: N.L Johnson,S. Kotz; N. Balakrishnan (1994) COntinuous Univariate Distribution Volume 1.
        sys.exit()
    if len([shape])!=1:
        sys.exit()

    # Random Variate:
    if shape == 0:
        r = location + scale * np.random.exponential(size=n)
    else:
        r = location + scale * (np.random.uniform(size=n)**(-shape) - 1)/shape

    # Return Value:
    return r



def depd(x, location = 0, scale = 1, shape = 0, log = False):
    """
    1) Description: 
        Density for the Generalized Pareto distribution function

    2) Input parameters:
        scale, location, shape: parameters of GPD
        x is the data element obtained from Fitbygpdpwm function 
        log= by default False
    3) Example:
        fit=Fitbygpdpwm(dataframe, ci=0.95, threshold=None)
        depd(fit['data'], location, scale=fit['scale'],shape= fit['shape'], log=False)

    """
    
    # Check:
    if min([scale])<0:
        sys.exit()
    if len([shape])!=1:
        sys.exit()


    # Density:
    d = [(i - location)/scale for i in x]
    dd=[]
    nn = len(d)
    scale = np.repeat(scale,nn)
    index = [(i > 0 and ((1 + shape * i) > 0)) or np.isnan(i) for i in x]
    if (shape == 0):
        for i in range(nn):
            if index[i]:
                dd.append(np.log(1/scale[i]) - d[i])
            else:
                dd.append(float("-Inf"))
    else:
        for i in range(nn):
            if index[i]:
                dd.append(np.log(1/scale[i]) - (1/shape+1)*np.log(1+shape*d[i]))
            else:
                dd.append(float("-Inf"))

    # Log:
    if log:
        dd = [np.exp(i) for i in dd]

    # Return Value:
    return dd


def pgpd(q, location = 0, scale = 1, shape = 1, lowertail = True):
    """
    1) Description: 
        Probability for the Generalized Pareto distribution function

    2) Input parameters:
        scale, location, shape: parameters of GPD
        lowertail = by default True
    """
    if min([scale])<0:
        sys.exit()
    if len([shape])!=1:
        sys.exit()

    q = (np.maximum(i - location, 0)/scale for i in q)
    if shape == 0:
        p = [1 - np.exp(-i) for i in q]

    else:
        p = [1-np.maximum(1 + shape * i, 0)**(-1/shape) for i in q]

    #Lower Tail:
    if lowertail:
        pass
    else:
        p=[1-i for i in p]

    return p






def qgpd( p, location = 0, scale = 1,shape = 1, lowertail = True):
    """
    1) Description: 
        Quantiles for the Generalized Pareto distribution function

    2) Input parameters:
        scale, location, shape: parameters of GPD
        lowertail = by default True

    """
    if min([scale])<0:
        sys.exit()
    if len([shape])!=1:
        sys.exit()
    if np.nanmin(p) < 0:
        sys.exit()
    if np.nanmax(p) >1:
        sys.exit()
    if lowertail:
        p= np.array([1 - k for k in p])
    # Quantiles:
    if shape == 0:
        q = location - scale * np.log(p)
    else:
        q = location + scale * (p**(-shape) - 1)/shape
    return q


def gpdMoments(shape = 1, location = 0, scale = 1):

    """
    1) Description:
       Compute true statistics for Generalized Pareto distribution

    2) Input parameter:
        shape, location, scale parameters from generalized pareto distribution 

    3) Value:
      Returns true mean of Generalized Pareto distribution
      for xi < 1 else NaN
      Returns true variance of Generalized Pareto distribution
      for xi < 1 else NaN

    """
    # MEAN: Returns 1 for x <= 0 and -Inf's's else
    a = [1,np.nan,np.nan]
    gpdMean = location + scale/(1-shape)*a[int(np.sign(shape-1)+1)]

    # VAR: Returns 1 for x <= 0 and -Inf's's else
    a = [1, np.nan, np.nan]
    gpdVar = scale*scale/(1-shape)**2/(1-2*shape) * a[int(np.sign(2*shape-1)+1)]

    param = '%s %f %s %f %s %f'  % ("Shape=",shape, "Location=", location, "Scale=", scale)
    ans = '%s %s %f %s %f' % (param, "mean = ",gpdMean, "var =", gpdVar)

    return ans