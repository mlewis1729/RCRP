# On 20190731 by Michael Lewis <mjlewis@cims.nyu.edu>
# Recursive Clustering Risk Parity
import numpy as np, pandas as pd
import clusterCorr as cC
#------------------------------------------------------------------------------
# Recursively Clustered Risk Parity (RCRP)
#------------------------------------------------------------------------------
def compressCov(cov, a, clusters, w):
    # Takes a covariance matrix and reduces the it to the basis vectors made by the weights from the clusters
    # Similarly, reduces the vector a (e.g. None, or expected returns), and reduces by the cluster weights
    wM = pd.DataFrame(0., index=clusters.keys(), columns=cov.index )
    #print(clusters)
    for i in np.sort( list( clusters.keys() ) ):
        wM.loc[i, clusters[i] ] = w[clusters[i]]
    a_clustered = None if a is None else wM.dot( a )
    cov_clustered = wM.dot(cov).dot(wM.T)
    return cov_clustered, a_clustered
#------------------------------------------------------------------------------
def getIVPNew(cov, use_extended_terms=False, a=None, limit_shorts=False):
    # Compute the inverse variance portfolio via approximation of the covariance matrix
    n = int(cov.shape[0])
    corr = cC.cov2corr(cov)
    invSigma = 1./np.sqrt(np.diag(cov))
    if a is None:
        # If a is None, we're finding the min variance portfolio
        # If a is not None, we're finding the max sharpe portfolio => a = mu
        if limit_shorts and n > 1 and np.sum(np.sum(corr)) >= n:
            # if average correlation is >= 0 and limiting shorts ( sum(|w|) = 1 )
            a = FindMinPermutation(invSigma)
        else:
            a = np.ones(n)
    ivp = a * (invSigma ** 2)
    if use_extended_terms and n > 1:
        # Obtain average off-diagonal correlation
        rho=(np.sum(np.sum(corr))-n)/(n**2-n)
        ivp-=rho*invSigma*np.sum(invSigma*a)/(1.+(n-1)*rho)
    if limit_shorts:
        # condition: sum(|w|) = 1
        return ivp / abs(ivp).sum()
    else:
        # condition: sum(w) = 1
        return ivp / ivp.sum()
#------------------------------------------------------------------------------
def RCRP(cov, use_extended_terms=True, a=None, limit_shorts=False):
    # Create a risk balance portfolio by recursively clustering the correlation matrix,
    # using the inverse variance portfolio at the lowest levels, then utilizing the optimal weights at lower levels
    # to compress the covariance matrix, and evaluating the optimal inverse variance portfolio on the compressed covariance matrix
    # a = None => min variance portfolio, otherwise a = Expected returns

    # default assume use extended terms
    w = pd.Series( 1., index = cov.index )

    # cluster the correlation matrix
    _, clusters, _ = cC.clusterKMeansTop(cC.cov2corr( cov ))
    if len( clusters.keys() ) > 1:
        for clusterId in clusters.keys():
            lbls = clusters[clusterId]
            if len(lbls) > 2:
                if a is None:
                    w[lbls] = RCRP(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=None)
                else:
                    w[lbls] = RCRP(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=a[lbls])
            else:
                if a is None:
                    w[lbls] = getIVPNew(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=None)
                else:
                    w[lbls] = getIVPNew(cov.loc[lbls, lbls], use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=a[lbls].values)
        # compress the covariance matrix (and vector a) using the optimal weights from lower levels
        cov_compressed, a_compressed = compressCov(cov, a, clusters, w)

        # evaluate the inverse variance portfolio on the clustered porfolio
        if a is None:
            w_clusters = getIVPNew(cov_compressed, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=None)
        else:
            w_clusters = getIVPNew(cov_compressed, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=a_compressed.values)

        # update the weights using the optimal cluster weights
        for clusterId in clusters.keys():
            lbls = clusters[clusterId]
            w[lbls] *= w_clusters[clusterId]
    else:
        # Only has one cluster
        if a is None:
            w = getIVPNew(cov, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=None)
        else:
            w = getIVPNew(cov, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=a.values)
    return w
#------------------------------------------------------------------------------
def Int2Bin(n,L):
    # input: positive integer n < 2**L, positive integer L
    # output: string for n in base 2 of length L
    x = "{0:b}".format(n)
    return '0' * (L-len(x)) + x 
#------------------------------------------------------------------------------
def Bin2Vector(x):
    # input: binary string x of length N, e.g. 01001
    # output: N vector v, e.g. np.asarray([-1 1 -1 -1 1])
    return np.asarray([2*int(b)-1 for b in x])
#------------------------------------------------------------------------------
def FindMinPermutation(invSigma):
    # Input: N vector containing 1/sigma > 0
    # output: N vector a s.t. sum( a * invSigma ) is minimized
    N = len(invSigma)
    maxN = 12
    a, minSum = None, np.inf
    if N <= maxN + 1:
        # N is small enough, just do the optimal value
        # NOTE: We don't need to check all 2^N possibilities; the sum for n2 = 2**N - 1 - n will have -sum of n
        for n in xrange(2**(N-1)):
            v = Bin2Vector(Int2Bin(n,N))
            s = abs(sum(v * invSigma))
            if s < minSum:
                a, minSum = v, s        
    else:
        # N is sufficiently large, find an approximate solution
        X = pd.Series( np.copy(invSigma), index=range(N) )
        for _ in xrange(2**maxN):
            tmp = X.sample(frac=1)
            s = tmp.iloc[0]
            b = '1'
            for i in xrange(1,len(tmp)):
                if s > 0:
                    s -= tmp.iloc[i]
                    b += '0'
                else:
                    s += tmp.iloc[i]
                    b += '1'
            if abs(s) < minSum:
                minSum, a = abs(s), Bin2Vector( ''.join( [b[i] for i in tmp.index.argsort()] ) )
    return a
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
# Build Class struct to hold B-Tree Structure, so structure can be re-used for different vectors
class Cov_Tree:
    def __init__( self, cluster_id = 0 ):
        self.cluster_id = cluster_id
        self.cluster_family_map = {}
        self.cluster_type_map = {}
        self.cov = None
        
    def __del__( self ):
        self.cluster_id = 0
        self.cluster_family_map = {}
        self.cluster_type_map = {}
        self.cov = None
        
    def fit( self, cov ):
        # cluster the correlation matrix
        _, clusters, _ = cC.clusterKMeansTop(cC.cov2corr( cov ))
        main_id = self.cluster_id
        if len( clusters ) > 1:
            this_id = main_id + 1
            cluster_set = []
            for clusterId in clusters.keys():
                lbls = clusters[clusterId]
                cluster_set.append( this_id )
                if len(lbls) > 2:
                    Sub_Tree = Cov_Tree( cluster_id = this_id )
                    Sub_Tree.fit( cov.loc[ lbls, lbls ] )
                    for idx, lbls1 in Sub_Tree.cluster_family_map.items():
                        self.cluster_family_map[ idx ] = lbls1
                        self.cluster_type_map[ idx ] = Sub_Tree.cluster_type_map[ idx ]
                    this_id += len( Sub_Tree.cluster_family_map )
                    del Sub_Tree
                else:
                    # small emough to be a leaf
                    self.cluster_family_map[ this_id ] = lbls
                    self.cluster_type_map[ this_id ] = 'LEAF'
                    this_id += 1                  
            self.cluster_family_map[ main_id ] = cluster_set
            self.cluster_type_map[ main_id ] = 'NODE'
        else:
            clusterId = clusters.keys()[0]
            lbls = clusters[ clusterId ]
            self.cluster_family_map[ main_id ] = lbls
            self.cluster_type_map[ main_id ] = 'LEAF'
        self.cov = cov        
        return self
    
    def solve( self, b, use_extended_terms = True, limit_shorts = False ):
        # solve C x = b
        
        def recursive_solve( idx ):
            cluster_type = self.cluster_type_map[ idx ]
            lbls         = self.cluster_family_map[ idx ]
            if cluster_type == 'LEAF':
                sub_cov = self.cov.loc[ lbls, lbls ]
                sub_b   = b.loc[ lbls ]
                w = getIVPNew( sub_cov, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=sub_b.values)
                w = pd.Series( w, index = lbls )
                return w
            else:
                w = []
                clustering = {}
                for lbl in lbls:
                    sub_w = recursive_solve( lbl )
                    w.append( sub_w )
                    clustering[ lbl ] = sub_w.index
                w = pd.concat( w )
                # compress the covariance matrix (and vector b) using the optimal weights from lower levels
                cov1 = self.cov.loc[ w.index, w.index ]
                b1 = b.loc[ w.index ]
                cov_compressed, b_compressed = compressCov( cov1, b1, clustering, w )

                # evaluate the inverse variance portfolio on the clustered porfolio
                w_clusters = getIVPNew(cov_compressed, use_extended_terms=use_extended_terms, limit_shorts=limit_shorts, a=b_compressed.values)
                w_clusters = pd.Series( w_clusters, index = cov_compressed.index )
                # update the weights using the optimal cluster weights
                for clusterId in clustering.keys():
                    lbls = clustering[clusterId]
                    w[lbls] *= w_clusters[clusterId]
                return w
            
        return recursive_solve( self.cluster_id )
#------------------------------------------------------------------------------
