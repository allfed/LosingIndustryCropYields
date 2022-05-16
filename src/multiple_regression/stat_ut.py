'''

Useful functions involving statistical operations, called from various locations in the code.

'''
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#%matplotlib inline

from src import params

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import warnings
#import statsmodels.api as sm

params.importIfNotAlready()


#calculates log likelihood for a continous distribution, taking one parameters: the log probability density function
#of the distribution
def loglik(lpdf):
    LL=np.sum(lpdf)
    return LL

#calculates AIC for a continous distribution, taking two parameters: the log probability density function of the distribution
#and a variable containing the parameters of the distribution
def aic(lpdf, param):
    LogLik=loglik(lpdf)
    k=len(param)
    aic1=2*k - 2*(LogLik)
    return aic1

#calculates BIC for a continuous distribution, taking three parameters: the log probability density function of the distribution
#a variable containing the parameters of the distribution and the dataset the distribution was fitted to
def biccont(lpdf, param):
    LogLik=loglik(lpdf)
    k=len(param)
    n=len(lpdf)
    bic1=k*np.log(n) - 2*(LogLik)
    return bic1


#function calculates loglikelihoods, AICs and BICs for all distributions given as argument.
#Aditionally it takes a list of log pdfs and a dictionary containing the pertaining parameter tuples as input
def stat_overview(distList, pdfList, paramDict):
    #create empty lists to store the result values
    LLs = []
    AICs = []
    BICs = []    
    #loop through all passed log pdfs, caclulate the loglikelihood and pass it into the LLS list
    for pdf in pdfList:
        LL = loglik(pdf)
        LLs.append(LL)
        for i in paramDict:
            #loop through the parameter tuples in the dictionary, calculate AIC and BIC for each tuple
            #and pass it into the AICs and BICs list
            param = paramDict[i]
            AIC = aic(pdf, param)
            AICs.append(AIC)
            BIC = biccont(pdf, param)
            BICs.append(BIC)
    #compile the calculated values in a dataframe with the distribution names and sort them so that the
    #best fit (lowest BIC) is listed on top
    results = pd.DataFrame()
    results['Distribution'] = distList
    results['loglikelihood'] = LLs
    results['AIC'] = AICs
    results['BIC'] = BICs
    results.sort_values(['BIC'], inplace=True)
    print(results)

'''
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [name for name, obj in st.__dict__.items() if isinstance(obj, st.rv_continuous)] 

    # Best holders + statistic holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    best_bic = np.inf
    sse_res = []
    LL_res = []
    AIC_res = []
    BIC_res = []
    #Introducing chi square requires a lot of code and packages
    #chi_square = []
    #p_values = []
    
    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    #percentile_bins = np.linspace(0,100,51)
    #percentile_cutoffs = np.percentile(y_std, percentile_bins)
    #observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    #cum_observed_frequency = np.cumsum(observed_frequency)
    

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                dist = getattr(st, distribution)
                param = dist.fit(data)
                #I don't think it's necessary to separate the parameters
                # Separate parts of parameters
                #arg = params[:-2]
                #loc = params[-2]
                #scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, *param)
                pdf_log = distribution.logpdf(data, *param)
                LL = loglik(pdf_log)
                LL_res.append(LL)
                AIC = aic(pdf_log, param)
                AIC_res.append(AIC)
                BIC = biccont(pdf_log, param)
                BIC_res.append(BIC)
                sse = np.sum(np.power(y - pdf, 2.0))
                sse_res.append(sse)

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0 and best_bic > BIC > 0:
                    best_distribution = distribution
                    best_params = param
                    best_sse = sse
                    best_bic = BIC

        except Exception:
            pass
        
    #results = pd.DataFrame()
    #results['Distribution'] = DISTRIBUTIONS
    #results['loglikelihood'] = LL_res
    #results['AIC'] = AIC_res
    #results['BIC'] = BIC_res
    #results['sse'] = sse_res
    #results.sort_values(['BIC'])

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
ax.set_xlabel(u'Temp (°C)')
ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (°C)')
ax.set_ylabel('Frequency')
'''
