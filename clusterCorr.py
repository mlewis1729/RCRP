import numpy as np, pandas as pd
import seaborn as sns
#------------------------------------------------------------------------------
def HeatMap(df):
    sns.heatmap(df, 
        xticklabels=df.columns,
        yticklabels=df.columns)
    return
#------------------------------------------------------------------------------
def clusterKMeansBase(corr0,maxNumClusters=10,n_init=10):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series() # distance matrix
    for init in range(n_init):
        for i in xrange(2,maxNumClusters+1): # find optimal num clusters
            kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_=kmeans_.fit(dist)
            silh_=silhouette_samples(dist,kmeans_.labels_)
            stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]: 
                silh,kmeans=silh_,kmeans_
    n_clusters = len( np.unique( kmeans.labels_ ) )
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0] ].tolist() for \
        i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=dist.index)
    return corr1,clstrs,silh
#------------------------------------------------------------------------------
def makeNewOutputs(corr0,clstrs,clstrs2):
    from sklearn.metrics import silhouette_samples
    clstrsNew,newIdx={},[]
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs2[i])
    map(newIdx.extend, clstrsNew.values())
    corrNew=corr0.loc[newIdx,newIdx]
    dist=((1-corr0.fillna(0))/2.)**.5
    kmeans_labels=np.zeros(len(dist.columns))
    for i in clstrsNew.keys():
        idxs=[dist.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs]=i
    silhNew=pd.Series(silhouette_samples(dist,kmeans_labels),index=dist.index)
    return corrNew,clstrsNew,silhNew
#------------------------------------------------------------------------------
def clusterKMeansTop(corr0,maxNumClusters=10,n_init=10):
    corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=min(corr0.shape[1]-1,maxNumClusters),n_init=n_init)
    clusterTstats={i:np.mean(silh[ clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean=np.mean(clusterTstats.values())
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]
    if len(redoClusters)<=2:
        return corr1,clstrs,silh
    else:
        keysRedo=[];map(keysRedo.extend,[clstrs[i] for i in redoClusters])
        corrTmp=corr0.loc[keysRedo,keysRedo]
        meanRedoTstat=np.mean([clusterTstats[i] for i in redoClusters])
        corr2,clstrs2,silh2=clusterKMeansTop(corrTmp, \
            maxNumClusters=maxNumClusters,n_init=n_init)
        # Make new outputs, if necessary
        corrNew,clstrsNew,silhNew=makeNewOutputs(corr0, \
            {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2 )
        newTstatMean=np.mean([np.mean(silhNew[clstrs2[i]])/np.std(silhNew[clstrs2[i]]) \
            for i in clstrs2.keys()])
        if newTstatMean<=meanRedoTstat:
            return corr1,clstrs,silh
        else:
            return corrNew,clstrsNew,silhNew
#------------------------------------------------------------------------------
def clusterKMeansTopMax(corr0,maxNumClusters=10,n_init=10):
    corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=min(corr0.shape[1]-1,maxNumClusters),n_init=n_init)
    clusterTstats={i:np.mean(silh[ clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean=np.mean(clusterTstats.values())
    tStatMax=np.max(clusterTstats.values())
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i]<tStatMax]
    keysRedo=[];map(keysRedo.extend,[clstrs[i] for i in redoClusters])
    corrTmp=corr0.loc[keysRedo,keysRedo]
    medianRedoTstat=np.median([clusterTstats[i] for i in redoClusters])
    if corrTmp.shape[1]<=4:
        return corr1,clstrs,silh
    corr2,clstrs2,silh2=clusterKMeansTop(corrTmp, \
            maxNumClusters=maxNumClusters,n_init=n_init)
    # Make new outputs, if necessary
    corrNew,clstrsNew,silhNew=makeNewOutputs(corr0, \
            {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2 )
    newTstatMedian=np.median([np.mean(silhNew[clstrs2[i]])/np.std(silhNew[clstrs2[i]]) \
            for i in clstrs2.keys()])
    if newTstatMedian<=medianRedoTstat:
            return corr1,clstrs,silh
    else:
            return corrNew,clstrsNew,silhNew
        
#------------------------------------------------------------------------------ 
def getIVP(cov,use_extended_terms=False):
    # Compute the minimum-variance portfolio
    ivp=1./np.diag(cov)
    if use_extended_terms:
        n=float(cov.shape[0])
        corr=cov2corr(cov)
        # Obtain average off-diagonal correlation
        rho=(np.sum(np.sum(corr))-n)/(n**2-n)
        invSigma=np.sqrt(ivp)
        ivp-=rho*invSigma*np.sum(invSigma)/(1.+(n-1)*rho)
    ivp/=ivp.sum()
    return ivp
#------------------------------------------------------------------------------
def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error
    return corr
#------------------------------------------------------------------------------ 
def clusterKMeansBaseV2(corr0):
    from SuperBGM import SuperBayesianGaussianMixture
    from sklearn.metrics import silhouette_samples
    covariance_type='full'
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series() # distance matrix
    k, score = 2, np.nan
    while ( np.isnan(score) or score < sbgm_.score(dist) ) and k < corr0.shape[0]:
        sbgm_ = SuperBayesianGaussianMixture(n_components=k, init_params='hierarchical', \
                                              covariance_type=covariance_type).fit(dist)
        k += 1
        if np.isnan(score) or score < sbgm_.score(dist):
            sbgm, score = sbgm_, sbgm_.score(dist)
    labels = sbgm.predict(dist)
    UniqueLabels = np.unique(labels)
    nClusters = len(UniqueLabels)
    silh = pd.Series(silhouette_samples(dist,labels), index=dist.index)
    clstrs={i:corr0.columns[np.where(labels==UniqueLabels[i])[0]].tolist() for \
        i in range(nClusters) } # cluster members
    newIdx=np.argsort(labels)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    return corr1,clstrs,silh  
#------------------------------------------------------------------------------
def clusterKMeansTopV2(corr0,ThresholdRho1=0.4,ThresholdRho2=0.8):
    if (corr0.fillna(0) - np.identity(corr0.shape[0])).abs().max().max() <= ThresholdRho1:
        # If the absolute value of the maximum off-diagonal correlation is < a threshold,
        # assume they're all individual clusters
        clstrs0 = {i:[ corr0.columns[i] ] for i in range(corr0.shape[0])}
        silh0 = pd.Series(np.nan, index=corr0.index)
        return corr0, clstrs0, silh0
    if corr0.fillna(0).min().min() >= ThresholdRho2:
        # If the min off-diagonal correlation is > a threshold,
        # assume they're all one cluster
        clstrs0 = {0:corr0.index}
        silh0 = pd.Series(np.nan, index=corr0.index)
        return corr0, clstrs0, silh0
    if corr0.shape[0] <= 2:
        # This is too small a matrix, call them individual clusters
        clstrs0 = {i:[ corr0.columns[i] ] for i in range(corr0.shape[0])}
        silh0 = pd.Series(np.nan, index=corr0.index)
        return corr0, clstrs0, silh0
    corr1,clstrs,silh=clusterKMeansBaseV2(corr0)
    clusterTstats={i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatsMean=np.nanmean(clusterTstats.values())
    tStatMax=max(clusterTstats.values())
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i]<tStatMax]
    keysRedo=[];map(keysRedo.extend,[clstrs[i] for i in redoClusters])
    corrTmp=corr0.loc[keysRedo,keysRedo]
    corr2,clstrs2,silh2=clusterKMeansTopV2(corrTmp,ThresholdRho1=ThresholdRho1,ThresholdRho2=ThresholdRho2)
    # Make new outputs, if necessary
    corrNew,clstrsNew,silhNew=makeNewOutputs(corr0, \
                        {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2 )
    newTstatMean=np.nanmean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) \
            for i in clstrsNew.keys()]) # Fixed code here
    blah1, blah2 = [], []
    map(blah1.extend,[ silh[clstrs[i]] for i in clstrs.keys() ])
    map(blah2.extend,[ silhNew[clstrsNew[i]] for i in clstrsNew.keys() ])
    tStatsMean2 = np.nanmean(blah1) / np.nanstd(blah1)
    newTstatMean2 = np.nanmean(blah2) / np.nanstd(blah2)
    #print tStatsMean, tStatsMean2, len(clstrs), silh, clusterTstats.values(), '\n', \
    #      newTstatMean, newTstatMean2, len(clstrsNew), silhNew, [np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) \
    #        for i in clstrsNew.keys()]
    if newTstatMean<=tStatsMean:
        #HeatMap(corr1)
        return corr1,clstrs,silh
    else:
        #HeatMap(corrNew)
        return corrNew,clstrsNew,silhNew
