"""

@author: Francesco Donato

@matricola: 304810

"""

import numpy   as np
import nibabel as nib # to read NII files
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product
import warnings

class CTScan:
    #%% methods
    def read_nii(self, filepath):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(filepath)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        return(array)

    def plotSample(self, array_list, color_map = 'nipy_spectral'):
        '''
        Plots a slice with all available annotations
        '''
        plt.figure(figsize=(18,15))

        plt.subplot(1,4,1)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.title('Original Image')

        plt.subplot(1,4,2)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
        plt.title('Lung Mask')

        plt.subplot(1,4,3)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[2], alpha=0.5, cmap=color_map)
        plt.title('Infection Mask')

        plt.subplot(1,4,4)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
        plt.title('Lung and Infection Mask')

        plt.show()
    

    def filterImage(self, D, NN):
        """D = image (matrix) to be filtered, Nr rows, N columns, scalar values (no RGB color image)
        The image is filtered using a square kernel/impulse response with side 2*NN+1"""
        E=D.copy()
        E[np.isnan(E)]=0
        Df=E*0
        Nr,Nc=D.shape
        rang=np.arange(-NN,NN+1)
        square=np.array([x for x in product(rang, rang)])
        #square=np.array([[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]])
        for kr in range(NN,Nr-NN):
            for kc in range(NN,Nc-NN):
                ir=kr+square[:,0]
                ic=kc+square[:,1]
                Df[kr,kc]=np.sum(E[ir,ic])# Df will have higher values where ones are close to each other in D
        return Df/square.size

    def useDBSCAN(self, D, z, epsv, min_samplesv):
        """D is the image to process, z is the list of image coordinates to be
        clustered"""
        Nr,Nc=D.shape
        clusters =DBSCAN(eps=epsv,min_samples=min_samplesv,metric='euclidean').fit(z)
        a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
        Nclust_DBSCAN=len(a)-1
        Npoints_per_cluster=Npoints_per_cluster[1:]# remove numb. of outliers (-1)
        ii=np.argsort(-Npoints_per_cluster)# from the most to the less populated clusters
        Npoints_per_cluster=Npoints_per_cluster[ii]
        C=np.zeros((Nr,Nc,Nclust_DBSCAN))*np.nan # one image for each cluster
        info=np.zeros((Nclust_DBSCAN,5),dtype=float)
        for k in range(Nclust_DBSCAN):
            i1=ii[k] 
            index=(clusters.labels_==i1)
            jj=z[index,:] # image coordinates of cluster k
            C[jj[:,0],jj[:,1],k]=1 # Ndarray with third coord k stores cluster k
            a=np.mean(jj,axis=0).tolist()
            b=np.var(jj,axis=0).tolist()
            info[k,0:2]=a #  store coordinates of the centroid
            info[k,2:4]=b # store variance
            info[k,4]=Npoints_per_cluster[k] # store points in cluster
        return C,info,clusters
    
#%% main part
if __name__=='__main__':     
    # Read sample
    plt.close('all')    
    plotFlag=True

    fold1='./data/ct_scans'
    fold2='./data/lung_mask'
    fold3='./data/infection_mask'
    fold4='./data/lung_and_infection_mask'
    f1='/coronacases_org_001.nii'
    f2='/coronacases_001.nii'
    
    
    ct=CTScan()
    
    sample_ct   = ct.read_nii(fold1+f1+f1)
    sample_lung = ct.read_nii(fold2+f2+f2)
    sample_infe = ct.read_nii(fold3+f2+f2)
    sample_all  = ct.read_nii(fold4+f2+f2)

    Nr,Nc,Nimages=sample_ct.shape# Nr=512,Nc=512,Nimages=301
    #%% Examine one slice of a ct scan and its annotations
    index=133   #CHOSEN SLICE 133
    sct=sample_ct[...,index]
    sl=sample_lung[...,index]
    si=sample_infe[...,index]
    sa=sample_all[...,index]
    ct.plotSample([sct,sl,si,sa])

    a=np.histogram(sct,200,density=True)
    if plotFlag:
        plt.figure()
        plt.plot(a[1][0:200],a[0])
        plt.title('Histogram of CT values in slice '+str(index))
        plt.grid()
        plt.xlabel('value')
    #%% Use Kmeans to perform color quantization of the image
    Ncluster=5
    kmeans = KMeans(n_clusters=Ncluster,random_state=0)# instantiate Kmeans
    A=sct.reshape(-1,1)# Ndarray, Nr*Nc rows, 1 column
    kmeans.fit(A)# run Kmeans on A
    kmeans_centroids=kmeans.cluster_centers_.flatten()#  centroids/quantized colors
    for k in range(Ncluster):
        ind=(kmeans.labels_==k)# indexes for which the label is equal to k
        A[ind]=kmeans_centroids[k]# set the quantized color
    sctq=A.reshape(Nr,Nc)# quantized image
    vm=sct.min()
    vM=sct.max()

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(sct, cmap='bone',interpolation="nearest")
    ax1.set_title('Original image')
    ax2.imshow(sctq,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
    ax2.set_title('Quantized image')

    ifind=1# second darkest color
    ii=kmeans_centroids.argsort()# sort centroids from lowest to highest
    ind_clust=ii[ifind]# get the index of the desired cluster 
    ind=(kmeans.labels_==ind_clust)# get the indexes of the pixels having the desired color
    D=A*np.nan
    D[ind]=1# set the corresponding values of D  to 1
    D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping
    plt.figure()
    plt.imshow(D,interpolation="nearest")
    plt.title('Image used to identify lungs')
    #%% DBSCAN to find the lungs in the image
    eps=2
    min_samples=5
    C,centroids,clust=ct.useDBSCAN(D,np.argwhere(D==1),eps,min_samples)
    # we want left lung first. If the images are already ordered
    # then the center along the y-axis (horizontal axis) of C[:,:,0] is smaller
    if centroids[1,1]<centroids[0,1]:# swap the two subimages
        print('swap')
        tmp = C[:,:,0]*1
        C[:,:,0] = C[:,:,1]*1
        C[:,:,1] = tmp
        tmp=centroids[0,:]*1
        centroids[0,:]=centroids[1,:]*1
        centroids[1,:]=tmp
    LLung = C[:,:,0].copy()  # left lung
    RLung = C[:,:,1].copy()  # right lung

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(LLung,interpolation="nearest")
    ax1.set_title('Left lung mask - initial')
    ax2.imshow(RLung,interpolation="nearest")
    ax2.set_title('Right lung mask - initial')

    #%% generate a new image with the two darkest colors of the color-quantized image
    D=A*np.nan
    ii=kmeans_centroids.argsort()# sort centroids from lowest to highest
    ind=(kmeans.labels_==ii[0])# get the indexes of the pixels with the darkest color
    D[ind]=1# set the corresponding values of D  to 1
    ind=(kmeans.labels_==ii[1])# get the indexes of the pixels with the 2nd darkest  color
    D[ind]=1# set the corresponding values of D  to 1
    D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping

    C,centers2,clust=ct.useDBSCAN(D,np.argwhere(D==1),2,5)
    ind=np.argwhere(centers2[:,4]<1000) # remove small clusters
    centers2=np.delete(centers2,ind,axis=0)
    distL=np.sum((centroids[0,0:2]-centers2[:,0:2])**2,axis=1)    
    distR=np.sum((centroids[1,0:2]-centers2[:,0:2])**2,axis=1)    
    iL=distL.argmin()
    iR=distR.argmin() 
    LLungMask=C[:,:,iL].copy()
    RLungMask=C[:,:,iR].copy()
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(LLungMask,interpolation="nearest")
    ax1.set_title('Left lung mask - improvement')
    ax2.imshow(RLungMask,interpolation="nearest")
    ax2.set_title('Right lung mask - improvement')

    #%% Final lung masks

    C,centers3,clust=ct.useDBSCAN(LLungMask,np.argwhere(np.isnan(LLungMask)),1,5)
    LLungMask=np.ones((Nr,Nc))
    LLungMask[C[:,:,0]==1]=np.nan
    C,centers3,clust=ct.useDBSCAN(RLungMask,np.argwhere(np.isnan(RLungMask)),1,5)
    RLungMask=np.ones((Nr,Nc))
    RLungMask[C[:,:,0]==1]=np.nan

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(LLungMask,interpolation="nearest")
    ax1.set_title('Left lung mask')
    ax2.imshow(RLungMask,interpolation="nearest")
    ax2.set_title('Right lung mask')


    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(LLungMask*sct,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
    ax1.set_title('Left lung')
    ax2.imshow(RLungMask*sct,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
    ax2.set_title('Right lung')

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(LLungMask*sct,interpolation="nearest")
    ax1.set_title('Left lung')
    ax2.imshow(RLungMask*sct,interpolation="nearest")
    ax2.set_title('Right lung')

    
    #%% Find ground glass opacities
    LLungMask[np.isnan(LLungMask)]=0
    RLungMask[np.isnan(RLungMask)]=0
    LungsMask=LLungMask+RLungMask

    B=LungsMask*sct
    inf_mask=1*(B>-700)&(B<-250)           
    InfectionMask=ct.filterImage(inf_mask,2)   #NN=2 defines the size of the kernel (5x5)
    InfectionMask=1.0*(InfectionMask>0.2)   # threshold to declare opacity     
    InfectionMask[InfectionMask==0]=np.nan
    plt.figure()
    plt.imshow(InfectionMask,interpolation="nearest")
    plt.title('infection mask')

    color_map = 'spring'
    plt.figure()
    plt.imshow(sct,alpha=0.8,vmin=vm,vmax=vM, cmap='bone')
    plt.imshow(InfectionMask*255,alpha=1,vmin=0,vmax=255, cmap=color_map,interpolation="nearest")
    plt.title('Original image with ground glass opacities in yellow')

    #%% FINDING THE PERCENTAGE OF GGO IN THE LUNGS (feature 1)
    Nggo=np.count_nonzero(InfectionMask==1)
    Ntot=np.count_nonzero(LungsMask==1)
    
    
    overall=Nggo/Ntot*100
    
    print(f"\nOverall percentage of ground glass opacities (GGO): {overall}\n")
    
    #%%SEPARATED LUNGS
    BL=LLungMask*sct
    Linf_mask=1*(BL>-700)&(BL<-250)           
    LInfectionMask=ct.filterImage(Linf_mask,2)   #NN=2 defines the size of the kernel (5x5)
    LInfectionMask=1.0*(LInfectionMask>0.2)   # threshold to declare opacity     
    LInfectionMask[LInfectionMask==0]=np.nan
    
    BR=RLungMask*sct
    Rinf_mask=1*(BR>-700)&(BR<-250)           
    RInfectionMask=ct.filterImage(Rinf_mask,2)   #NN=2 defines the size of the kernel (5x5)
    RInfectionMask=1.0*(RInfectionMask>0.2)   # threshold to declare opacity     
    RInfectionMask[RInfectionMask==0]=np.nan
    
    NggoL=np.count_nonzero(LInfectionMask==1)
    Nl=np.count_nonzero(LLungMask==1)
    
    NggoR=np.count_nonzero(RInfectionMask==1)
    Nr=np.count_nonzero(RLungMask==1)
    
    left=NggoL/Nl*100
    right=NggoR/Nr*100

    
    print(f"Percentage of GGO (left lung): {left}\n")
    print(f"Percentage of GGO (right lung): {right}\n\n")
    
    
    
    

########## SECOND FEATURE: NOT RELIABLE, AS A CONSEQUENCE IT IS NOT IN THE REPORT ############

    #%%CLUSTERS=LOCAL DENSE ZONES OF GGO
    #%%OUTLIERS=LESS DENSE BUT LARGER DISTRIBUTION OF GGO OVER THE LUNGS
    
    #%% FINDING CLUSTERS OF GGO (feature 2) 
    N=11
    eps=np.zeros(18, dtype=float)
    M=np.zeros(11, dtype=float)
    ln=np.zeros(10*8, dtype=float)

    eps[0]=3
    for i in range(N+10-4):
        
        eps[i+1]=eps[i]+1

    M[0]=2
    for i in range(N-1):
        
        M[i+1]=M[i]+1
        
    i=0
    j=0
    t=0
    an=np.zeros((200,4), dtype=float)
    #%% ALGORITHM FOR FINDING REASONABLE VALUES FOR eps AND M IN ORDER TO IDENTIFY MAIN CLUSTERS AND OUTLIERS IN TERMS OF GGO IN THE IMAGE
    for i in eps:        
        for j in M:
            
                warnings.filterwarnings("ignore")
                clustersggo =DBSCAN(i,j,metric='euclidean').fit(np.argwhere(InfectionMask==1))
                numclust,Npoints_per_clusterggo=np.unique(clustersggo.labels_,return_counts=True)
                
                if len(numclust)-1>4 and Npoints_per_clusterggo[0]>0:
                    an[t,0]=i      #eps
                    an[t,1]=j      #M
                    an[t,2]=len(numclust)-1    #number of clusters
                    an[t,3]=Npoints_per_clusterggo[0]     #number of outliers
                    t=t+1
            
            
            
    #%% FINDING CLUSTERS AND OUTLIERS WITH CHOSEN VALUES FOR eps AND M       
    clustersggo =DBSCAN(12,12,metric='euclidean').fit(np.argwhere(InfectionMask==1))             
    numclust,Npoints_per_clusterggo=np.unique(clustersggo.labels_,return_counts=True)

    print(f"number of clusters: {len(numclust)-1}\n")
    print(f"number of outliers: {Npoints_per_clusterggo[0]}\n")
    for i in range(len(numclust)):
        
        if i!=0:
            print(f"cluster {i}: {Npoints_per_clusterggo[i]} points\n")    
##########################################################################################