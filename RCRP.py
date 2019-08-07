# On 20190731 by Michael Lewis <mjlewis@cims.nyu.edu>
# Recursive Clustering Risk Parity
import numpy as np, pandas as pd
import clusterCorr as cC
#------------------------------------------------------------------------------
# Recursively Clustered Risk Parity (RCRP)
#------------------------------------------------------------------------------
def compressCov(cov, a, clusters, w):
    # Takes a covariance matrix and reduces the it to the basis vectors made by the weights from the clusters
    # Similarly, reduces the vector a (e.g. 1s, or expected returns), and reduces by the cluster weights
    wM = pd.DataFrame(0., index=clusters.keys(), columns=cov.index )
    for i in np.sort( clusters.keys() ):
        wM.loc[i, clusters[i] ] = w[clusters[i]]
    a_clustered = wM.dot( a )
    cov_clustered = wM.dot(cov).dot(wM.T)
    return cov_clustered, a_clustered
#------------------------------------------------------------------------------
def getIVPNew(cov, use_extended_terms=False, a=None):
    # Compute the inverse variance portfolio via approximation of the covariance matrix
    if a is None:
        # If a is None, we're doing min variance => a = 1
        # If a is not None, we're doing max sharpe => a = mu
        a = np.ones(cov.shape[0])
    ivp = a / np.diag(cov)
    if use_extended_terms and cov.shape[0] > 1:
        n=float(cov.shape[0])
        corr=cC.cov2corr(cov)
        # Obtain average off-diagonal correlation
        rho=(np.sum(np.sum(corr))-n)/(n**2-n)
        #invSigma=np.sqrt(ivp)
        invSigma=1./np.sqrt(np.diag(cov))
        ivp-=rho*invSigma*np.sum(invSigma*a)/(1.+(n-1)*rho)
    ivp/=ivp.sum()
    return ivp
#------------------------------------------------------------------------------
def RCRP(cov, use_extended_terms=True, a=None):
    # Create a risk balance portfolio by recursively clustering the correlation matrix,
    # using the inverse variance portfolio at the lowest levels, then utilizing the optimal weights at lower levels
    # to compress the covariance matrix, and evaluating the optimal inverse variance portfolio on the compressed covariance matrix
    if a is None:
        a = pd.Series( 1., index = cov.index )
    # default assume use extended terms
    w = pd.Series( 1., index = cov.index )

    # cluster the correlation matrix
    _, clusters, _ = cC.clusterKMeansTop(cC.cov2corr( cov ))
    if len( clusters.keys() ) > 1:
        for clusterId in clusters.keys():
            lbls = clusters[clusterId]
            if len(lbls) > 2:
                w[lbls] = RCRP(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, a=a[lbls])
            else:
                w[lbls] = getIVPNew(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, a=a[lbls].values)
        # compress the covariance matrix (and vector a) using the optimal weights from lower levels
        cov_compressed, a_compressed = compressCov(cov, a, clusters, w)

        # evaluate the inverse variance portfolio on the clustered porfolio
        w_clusters = getIVPNew(cov_compressed, use_extended_terms=use_extended_terms, a=a_compressed.values)

        # update the weights using the optimal cluster weights
        for clusterId in clusters.keys():
            lbls = clusters[clusterId]
            w[lbls] *= w_clusters[clusterId]
    else:
        # Only has one cluster
        w = getIVPNew(cov, use_extended_terms=use_extended_terms, a=a.values)
    return w
#------------------------------------------------------------------------------
# Modified HRP
#------------------------------------------------------------------------------
def getClusterStats(cov, cItems, use_extended_terms=False, returns=None):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    if returns is None:
        w_=getIVPNew(cov_, use_extended_terms=use_extended_terms, a=None).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar, 1.
    else:
        mu_=returns[cItems].values.reshape(-1,1)
        w_=getIVPNew(cov_, use_extended_terms=use_extended_terms, a=mu_[:,0]).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        cMu=np.dot(w_.T,mu_)[0,0]
        return cVar, cMu
#------------------------------------------------------------------------------
def getClusterCorr(cov, cItems0, cItems1, use_extended_terms=False, returns=None):
    # Compute covariance of clusters
    cov_0, cov_1, cov_01 = cov.loc[cItems0,cItems0], cov.loc[cItems1,cItems1], cov.loc[cItems0,cItems1] # matrix slice
    if returns is None:
        w_0=getIVPNew(cov_0, use_extended_terms=use_extended_terms, a=None).reshape(-1,1)
        w_1=getIVPNew(cov_1, use_extended_terms=use_extended_terms, a=None).reshape(-1,1)
    else:
        mu_0, mu_1 = returns[cItems0].values, returns[cItems1].values
        w_0=getIVPNew(cov_0, use_extended_terms=use_extended_terms, a=mu_0).reshape(-1,1)
        w_1=getIVPNew(cov_1, use_extended_terms=use_extended_terms, a=mu_1).reshape(-1,1)
    cCov =np.dot(np.dot(w_0.T,cov_01),w_1)[0,0]
    cVar0=np.dot(np.dot(w_0.T,cov_0),w_0)[0,0]
    cVar1=np.dot(np.dot(w_1.T,cov_1),w_1)[0,0]
    cRho=cCov / np.sqrt(cVar0 * cVar1)
    return cRho
#------------------------------------------------------------------------------
def getRecBipartNew(cov, sortIx, use_extended_terms1=False, use_extended_terms2=False, returns=None):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)/2), \
            (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in xrange(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0,weight0=getClusterStats(cov, cItems0, use_extended_terms=use_extended_terms2, returns=returns)
            cVar1,weight1=getClusterStats(cov, cItems1, use_extended_terms=use_extended_terms2, returns=returns)
            term0, term1 = weight0 / cVar0, weight1 / cVar1
            if use_extended_terms1:
                rho=getClusterCorr(cov, cItems0, cItems1, use_extended_terms=use_extended_terms2, returns=returns)
                sigma0, sigma1 = np.sqrt(cVar0), np.sqrt(cVar1)
                M=rho*(weight0/sigma0 + weight1/sigma1) / (1. + rho)
                term0 -= M / sigma0
                term1 -= M / sigma1
            alpha=term0/(term0 + term1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w
#------------------------------------------------------------------------------