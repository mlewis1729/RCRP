# BACKTESTING
import seaborn as sns
import RCRP as rcrp
import RandomCov as rc
import HRP as hrp
import numpy as np, pandas as pd
import os
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np, pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.signal
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.linalg import solve
import sys
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
import pytz

yearToRun        = int( sys.argv[1] )
outFile          = 'performance' + str( yearToRun ) + '.csv'
wgts_out         = 'weights' + str( yearToRun ) + '.csv'
start_date, end_date = str( yearToRun - 3 ) + '-01-01', str( yearToRun + 1 ) + '-03-01'
#( pd.to_datetime('today') + pd.DateOffset(days=1) ).strftime('%Y-%m-%d')
Ticker_Name_File = 'NYSE.csv'

df = pd.read_csv( Ticker_Name_File )

tickers = df['Symbol'].unique()
ClosePrices = {}
i = 0
t1 = datetime.now()
for ticker in tickers:
    i += 1
    try:
        data = yf.download(ticker, start_date, end_date)
    except:
        continue
    ClosePrices[ticker] = data[data.index >= start_date].Close
    if i % 10 == 0:
        t2 = datetime.now()
        leftover = len(tickers) - i
        rate = (t2-t1).seconds * 1. / i
        print( i, ",", leftover * rate, " seconds to go" )

def Get_HRP_sortIx(corr):
    import scipy.cluster.hierarchy as sch
    dist=hrp.correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=hrp.getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    return sortIx

def MakeEWMCov(returns, halflife):
    alpha = 1-np.exp(np.log(0.5)/halflife)
    weights = (pd.Series((1-alpha)*np.ones(len(returns.dropna()))).cumprod() / (1-alpha)).sort_values().values
    return pd.DataFrame( np.cov(returns.dropna().values.T, aweights=weights) , index=returns.columns, columns=returns.columns )


maxCorr = 0.80
halflife = 60
d1, d2 = '1/1/' + str( yearToRun ),'1/31/' + str( yearToRun+1 )
dateSet = pd.date_range(d1,d2,freq='MS')
BaseReturns = pd.DataFrame(ClosePrices).ffill().pct_change()

# still need to nan out companies that end
for cmpy in BaseReturns.columns:
    try:
        LtDt = ClosePrices[cmpy].index[-1]
        BaseReturns.loc[ BaseReturns.index > LtDt, cmpy ] = np.nan
    except:
        continue

# VZA seems to have bad data
Bad_Companies = [ 'VZA' ]
for bad in Bad_Companies:
    if bad in BaseReturns.columns:
       BaseReturns =  BaseReturns.drop( columns = bad )

def this_Function(T):
    dateN, dateThru = T
    WFile = 'weights.' + dateThru.strftime('%Y%m') + '.csv'
    PFile = 'performance.' + dateThru.strftime('%Y%m') + '.csv'
    if os.path.exists( WFile ) and os.path.exists( PFile ):
        return [], []
    nextdateThru = dateSet[dateN+1]
    returns = BaseReturns.copy()
    returns = returns[returns.index < dateThru]
    UseCompanies  = ( returns.apply(lambda x: np.isnan(x).sum(), axis=0) <= 1 )
    UseCompanies  = UseCompanies & ( returns.apply(lambda x: np.abs(x) > 0., axis=0).sum() >= 1 )
    returns = returns[returns.columns[UseCompanies]]
    CompanySet = returns.columns
    StartLoop = True
    while StartLoop or len( C[C.abs()>maxCorr] ) > 0:
        StartLoop = False
        cov = MakeEWMCov(returns, halflife)
        mu  = returns.ewm(halflife=halflife).mean().iloc[-1]
        corr = rc.cov2corr( cov )
        C = corr.stack()
        C = C[C.index.get_level_values(0) < C.index.get_level_values(1) ]
        while len( C[C.abs()>maxCorr] ) > 0:
            StartLoop = True
            CompanySet = CompanySet[CompanySet!=C[C.abs()>maxCorr].index.get_level_values(1)[0] ]
            returns = returns[CompanySet]
            corr    = corr.loc[ CompanySet, CompanySet ] 
            C       = corr.stack()
            C       = C[C.index.get_level_values(0) < C.index.get_level_values(1) ]

    while np.min(np.diag(cov)) <= 0.:
        tmp = pd.Series(np.diag(cov), index=cov.index)
        CompanySet = CompanySet[~CompanySet.isin(tmp[tmp<=0.].index)]
        returns = returns[CompanySet]
        #cov = returns.ewm(halflife=90).cov().iloc[-len(returns.columns):]
        #cov.index = cov.index.droplevel(0)
        cov = MakeEWMCov(returns, halflife)
        mu  = returns.ewm(halflife=halflife).mean().iloc[-1]
        corr = rc.cov2corr( cov )
    print('Procssing RCRP', flush=True)
    b = pd.Series( 1., index = cov.index )
    Ct = rcrp.Cov_Tree()
    Ct.fit( cov )
    #w     = rcrp.RCRP( cov )
    w     = Ct.solve( b, use_extended_terms=True )
    print('Processing RCRP without extended terms', flush=True)
    #w_net = rcrp.RCRP( cov, use_extended_terms=False )
    w_net = Ct.solve( b, use_extended_terms=False )
    #w_MAX = rcrp.RCRP( cov, a=mu )
    #w_MAX_net = rcrp.RCRP( cov, a=mu, use_extended_terms=False )
    print('Processing HRP', flush=True)
    sortIx = Get_HRP_sortIx(corr) 
    print('gerRecBipart', flush=True)
    w_HRP = hrp.getRecBipart(cov, sortIx)
    #print('Standard', flush=True)
    #ones = pd.Series( 1., index = cov.index )
    #w_std = pd.Series( solve( cov, ones ), index = cov.index ) # sigma_inv * one
    #w_std = w_std / w_std.sum() # weights sum to 1
    #returns_future = pd.DataFrame(ClosePrices).ffill().pct_change()[ ( ClosePrices['A'].index >= dateThru ) & ( ClosePrices['A'].index < nextdateThru )]
    print('Processing IVP')
    w_IVP = pd.Series( 1./ np.diag( cov ), index = cov.index )
    w_IVP = w_IVP / w_IVP.sum()
    print('Processing LW')
    LW = LedoitWolf().fit(returns.dropna()) # not technically exponentiall weighted
    LW = pd.DataFrame( LW.covariance_, index=cov.index, columns=cov.index )
    ones = pd.Series( 1., index = LW.index )
    w_LW = pd.Series( solve( LW, ones ), index = LW.index ) # sigma_inv * one
    w_LW = w_LW / w_LW.sum() # weights sum to 1
    print('Processing OAS')
    oas = OAS().fit(returns.dropna()) # not technically exponentiall weighted
    oas = pd.DataFrame( oas.covariance_, index=cov.index, columns=cov.index )
    ones = pd.Series( 1., index = oas.index )    
    w_oas = pd.Series( solve( oas, ones ), index = oas.index ) # sigma_inv * one
    w_oas = w_oas / w_oas.sum() # weights sum to 1
    returns_future = BaseReturns.copy()
    returns_future = returns_future[ ( returns_future.index >= dateThru ) & ( returns_future.index < nextdateThru ) ]
    returns_future = returns_future[ CompanySet ].ffill(0.) # add 0 returns to dead companies for the rest of the month, technically only wrong for day it dies
    out = [returns_future.dot(w).to_frame('RCRP')]
    out.append(returns_future.dot(w_net).to_frame('RCRP, no extended terms'))
    #out.append(returns_future.dot(w_MAX).to_frame('RCRP, max sharpe'))
    #out.append(returns_future.dot(w_MAX_net).to_frame('RCRP, max sharpe, no extended terms'))
    out.append(returns_future.dot(w_HRP).to_frame('HRP'))
    #out.append(returns_future.dot(w_std).to_frame('Standard'))
    out.append(returns_future.dot(w_IVP).to_frame('IVP'))
    out.append(returns_future.dot(w_LW).to_frame('LW'))
    out.append(returns_future.dot(w_oas).to_frame('OAS'))
    out = pd.concat(out, axis=1)
    weights = []
    weights.append( w.to_frame('RCRP') )
    weights.append( w_net.to_frame('RCRP, no extended terms') )
    weights.append( w_HRP.to_frame('HRP') )
    weights.append( w_IVP.to_frame('IVP') )
    weights.append( w_LW.to_frame('LW') )
    weights.append( w_oas.to_frame('OAS') )
    #weights.append( w_std.to_frame('Standard') )
    weights = pd.concat(weights, axis=1)
    weights.index.name = 'Company'
    weights = weights.reset_index()
    weights['dateThru'] = dateThru
    print("Procsse completed ", T, flush=True)
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    print( nyc_datetime, flush=True )
    #WFile = 'weights.' + dateThru.strftime('%Y%m') + '.csv'
    weights.to_csv( WFile )
    #PFile = 'performance.' + dateThru.strftime('%Y%m') + '.csv'
    out.to_csv( PFile )
    return out, weights

#pool = Pool(processes=4) 
#inputs = enumerate(dateSet[:-1])
#outputs = pool.map(this_Function, inputs)
#pool.close()
results = []
for x in enumerate(dateSet[:-1]):
    results.append( this_Function(x) )

performance = [ x[0] for x in results ]
performance = pd.concat( performance )
#performance.to_csv( outFile ) 

Wgts = [ x[1] for x in results ]
Wgts = pd.concat( Wgts )
#Wgts.to_csv( wgts_out )

#f = open("performance3.txt", "w")  
#for blah in results:
#    blah2 = []
#    for x in blah:
#        if( isinstance(x, str) ):
#            blah2.extend( x.split("\n") )
#        else:
#            blah2.append( x._date_repr + " 00:00:00" )
#    #f.write( "\n".join(blah2))
#    out = "\n".join(blah2) + "\n"
#    f.write(out)
#f.close()

