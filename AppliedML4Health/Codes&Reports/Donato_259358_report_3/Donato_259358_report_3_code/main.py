#with classes
"""
@author: Francesco Donato

@matricola: 259358

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Covid:

        
    def findROC(self,x,y):   # METHOD FOR ROC
    
        if x.min()>0:# add a couple of zeros, in order to have the zero threshold
            x=np.insert(x,0,0)# add a zero as the first element of xs
            y=np.insert(y,0,0)# also add a zero in y
    
        ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
        ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
        x0=x[ii0]# test values for healthy patients
        x1=x[ii1]# test values for ill patients
        xs=np.sort(x)# sort test values: they represent all the possible  thresholds
        # if x> thresh -> test is positive
        # if x <= thresh -> test is negative
        # number of cases for which x0> thresh represent false positives
        # number of cases for which x0<= thresh represent true negatives
        # number of cases for which x1> thresh represent true positives
        # number of cases for which x1<= thresh represent false negatives
        # sensitivity = P(x>thresh|the patient is ill)=
        #             = P(x>thresh, the patient is ill)/P(the patient is ill)
        #             = number of positives in x1/number of positives in y
        # false alarm = P(x>thresh|the patient is healthy)
        #             = number of positives in x0/number of negatives in y
        Np=ii1.size# number of positive cases
        Nn=ii0.size# number of negative cases
        data=np.zeros((Np+Nn,3),dtype=float)
        i=0
        ROCarea=0
    
        for thresh in xs:
            n1=np.sum(x1>thresh)#true positives
            sens=n1/Np
            n2=np.sum(x0>thresh)#false positives
            falsealarm=n2/Nn
            spec=1-falsealarm
            data[i,0]=thresh
            data[i,1]=falsealarm
            data[i,2]=sens
            if i>0:
                ROCarea=ROCarea+sens*(data[i-1,1]-data[i,1])
          
        
            
            i=i+1
        return data,ROCarea

    def prob(self,x,y,test):     # METHOD FOR CALCULATING AND PLOTTING OF PROBABILITIES
    
        if x.min()>0:# add a couple of zeros, in order to have the zero threshold
            x=np.insert(x,0,0)# add a zero as the first element of xs
            y=np.insert(y,0,0)# also add a zero in y
    
        ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
        ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
        x0=x[ii0]# test values for healthy patients
        x1=x[ii1]# test values for ill patients
        xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    
        Np=ii1.size# number of positive cases
        Nn=ii0.size# number of negative cases
        d_given_pos=np.zeros((len(xs),1),dtype=float)
        h_given_neg=np.zeros((len(xs),1),dtype=float)
        d_given_neg=np.zeros((len(xs),1), dtype=float)
        h_given_pos=np.zeros((len(xs),1), dtype=float)
        i=0
        prevalence_D=0.02
        prevalence_H=0.98
        epsilon=0.00000001
        false_neg=np.zeros((len(xs),1),dtype=float)
        sensitivity=np.zeros((len(xs),1),dtype=float)
        specificity=np.zeros((len(xs),1),dtype=float)
    
        #%% POS AND NEG WITH THE CHOSEN THRESHOLD
        threshold_1=7.71
        threshold_2=0.3
        positive_=0
        negative_=0
        positive_s=0
        negative_s=0
    
        if test=="1":
            for o in range(len(x)):
                if x[o]>threshold_1:
                        positive_=positive_+1
                elif x[o]<threshold_1:
                        negative_=negative_+1
                
        if test=="2":
            for o in range(len(x)):
                if x[o]>threshold_2:
                            positive_=positive_+1
                elif x[o]<threshold_2:
                            negative_=negative_+1
                
        print(f"Test {test}: {positive_} positives and {negative_} negatives\n")
    
        if test=="1":
            for o in range(len(x)):
                if x[o]>threshold_1 and y[o]==1:
                    positive_s=positive_s+1
                elif x[o]<threshold_1 and y[o]==0:
                    negative_s=negative_s+1
                
        if test=="2":
            for o in range(len(x)):
                if x[o]>threshold_2 and y[o]==1:
                            positive_s=positive_s+1
                elif x[o]<threshold_2 and y[o]==0:
                            negative_s=negative_s+1
                
        print(f"Test {test} + swab: {positive_s} positives and {negative_s} negatives\n")
                #%%
                
        for thresh in xs:
        
            n1=np.sum(x1>thresh)#true positives
            n3=np.sum(x1<thresh)#false negatives
            n2=np.sum(x0>thresh)#false positives
            n4=np.sum(x0<thresh)#true negatives
        
            falsealarm=n2/Nn #false alarm
            f_n=n3/Np
            sens=n1/Np #sensitivity
            t_n=n4/Nn
            spec=1-falsealarm #specificity
        
            # FINDING THE THRESHOLD WHEN SPEC=SENS:
            #if test=="1":
                #if spec<sens+0.003 and spec>sens-0.003:
                
                    #print(f"\nthreshold test 1 (spec=sens)= {thresh}\n spec= {spec}\n sens= {sens}\n")
                
            #if test=="2":
                #if spec<sens+0.001 and spec>sens-0.001:
                
                    #print(f"threshold test 2 (spec=sens)= {thresh}\n spec= {spec}\n sens= {sens}\n")
                
                
                
            d_given_pos_val=(sens*prevalence_D)/((sens*prevalence_D+falsealarm*prevalence_H)+epsilon)    # epsilon is to avoid division for zero
            h_given_neg_val=(t_n*prevalence_H)/(f_n*prevalence_D+t_n*prevalence_H)
            h_given_pos_val=1-d_given_pos_val
            d_given_neg_val=1-h_given_neg_val
        
        
            d_given_pos[i,0]=d_given_pos_val
        
            h_given_neg[i,0]=h_given_neg_val
        
            h_given_pos[i,0]=h_given_pos_val
        
            d_given_neg[i,0]=d_given_neg_val
        
        
            sensitivity[i,0]=sens
            specificity[i,0]=spec
            false_neg[i,0]=f_n
        
        
            if test=="1" and thresh==9.28:
            
                print(f"\nTest1 - thresh={thresh}:\nP(D|Tp)={d_given_pos[i,0]}\nP(D|Tn)={d_given_neg[i,0]}\nP(H|Tp)={h_given_pos[i,0]}\nP(H|Tn)={h_given_neg[i,0]}\nsensitivity={sens}\nspecificity={spec}\nfalse negative={f_n}\nfalse positive={falsealarm}")
        
            if test=="2" and thresh==0.44:
                
                print(f"\nTest2 - thresh={thresh}:\nP(D|Tp)={d_given_pos[i,0]}\nP(D|Tn)={d_given_neg[i,0]}\nP(H|Tp)={h_given_pos[i,0]}\nP(H|Tn)={h_given_neg[i,0]}\nsensitivity={sens}\nspecificity={spec}\nfalse negative={f_n}\nfalse positive={falsealarm}")
        
        
            if test=="1" and thresh==7.59:      #SETTING THE THRESHOLD: if test=="1" and d_given_pos[i,0]<0.15284 and d_given_pos[i,0]>0.13 and spec>0.82 and thresh<8.65:
            
                print(f"\nTest1 - thresh={thresh}:\nP(D|Tp)={d_given_pos[i,0]}\nP(D|Tn)={d_given_neg[i,0]}\nP(H|Tp)={h_given_pos[i,0]}\nP(H|Tn)={h_given_neg[i,0]}\nsensitivity={sens}\nspecificity={spec}\nfalse negative={f_n}\nfalse positive={falsealarm}")
        
            if test=="2" and thresh==0.3:       #SETTING THE THRESHOLD: if test=="2" and d_given_pos[i,0]<0.185 and d_given_pos[i,0]>0.16 and spec>0.82 and thresh<0.44:
                
                print(f"\nTest2 - thresh={thresh}:\nP(D|Tp)={d_given_pos[i,0]}\nP(D|Tn)={d_given_neg[i,0]}\nP(H|Tp)={h_given_pos[i,0]}\nP(H|Tn)={h_given_neg[i,0]}\nsensitivity={sens}\nspecificity={spec}\nfalse negative={f_n}\nfalse positive={falsealarm}")
        
        
            i=i+1
        
        plt.figure()
        plt.plot(xs, d_given_pos[:,0],'-',label='P(D|Tp)')
        plt.plot(xs, d_given_neg[:,0],'-',label='P(D|Tn)')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'Test {test}')
        plt.grid()
        
        plt.figure()
        plt.plot(xs, h_given_neg[:,0],'-',label='P(H|Tn)')
        plt.plot(xs, h_given_pos[:,0],'-',label='P(H|Tp)')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'Test {test}')
        plt.grid()
            
        plt.figure()
        plt.plot(xs, d_given_pos[:,0],'-',label='P(D|Tp)')
        plt.plot(xs, d_given_neg[:,0],'-',label='P(D|Tn)')
        plt.plot(xs, sensitivity[:,0],'-',label='sensitivity')
        plt.plot(xs, specificity[:,0],'-',label='specificity')
        plt.plot(xs, false_neg[:,0],'-',label='false negative')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'Test {test}')
        plt.grid()             
            
    def plots(self, FA, sens, threshold, spec, test): #PLOTS
    
        plt.figure()
        plt.plot(FA,sens,'-',label=f'Test {test}')
        plt.xlabel('FA')
        plt.ylabel('Sens')
        plt.grid()
        plt.legend()
        plt.title(f'ROC - Test {test}')
        plt.figure()
        plt.plot(threshold,FA,'.',label='False alarm')
        plt.plot(threshold,sens,'.',label='Sensitivity')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'Test {test}')
        plt.grid()

        plt.figure()
        plt.plot(threshold,spec,'-',label='Specificity')
        plt.plot(threshold,sens,'-',label='Sensitivity')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'Test {test}')
        plt.grid()
    
      
        


#%% DATA ANALYSIS
plt.close('all')
xx=pd.read_csv("covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
ii=np.argwhere(swab==1).flatten()
swab=np.delete(swab,ii)
swab=swab//2
Test1=np.delete(Test1,ii)
Test2=np.delete(Test2,ii)

neg=0
pos=0
for i in range(len(Test2)):
    
    if swab[i]==0:
        neg=neg+1
    else:
        pos=pos+1
        
print(f"\nPositive swab tests= {pos}")
print(f"Negative swab tests= {neg}\n")

#%% REMOVING OUTLIERS FROM TEST1
Test1_matrix=np.zeros((862,1), dtype=float)
Test1_matrix[:,0]=Test1      #  DBSCAN needs a matrix

plt.figure()
plt.plot(Test1_matrix[:,0])
plt.legend()
plt.title('Tes1 - with outliers')
plt.grid()


db=DBSCAN(eps=7, min_samples=2).fit(Test1_matrix)  # eps=3, min_samples=5
labels = db.labels_
ol=0
ol_pos=0
ol_neg=0
for i in range(len(labels)):
    
    if labels[i]==-1:        # it means there is an outlier
        ol=ol+1 
        
        if swab[i]==0:
            
            ol_neg=ol_neg+1
        
        elif swab[i]==1:
            
            ol_pos=ol_pos+1

print(f"Among the removed outliers, {ol_pos} were positive and {ol_neg} were negative.\n")

Test1_no_outl=np.zeros((862-ol,1), dtype=float)    # test1 matrix without outliers
swab1=np.zeros((862-ol,1), dtype=int)
j=0
for i in range(len(labels)):
    
    if labels[i]!=-1:
        Test1_no_outl[j]=Test1_matrix[i]
        swab1[j]=swab[i]              # generating a vector without the swab tests corresponding to outliers of test1
        j=j+1
        
        
plt.figure()
plt.plot(Test1_no_outl[:,0])
plt.legend()
plt.title('Tes1 - without outliers')
plt.grid()

no_clusters = len(np.unique(labels) )
no_outl = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of outliers: %d\n' % no_outl)




# TEST 2
#%%
ii0=np.argwhere(swab==0)
ii1=np.argwhere(swab==1)
plt.figure()
plt.hist(Test2[ii0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test2[ii1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()

plt.title('Test2')



# TEST 1
#%%
ii0_1=np.argwhere(swab1[:,0]==0)
ii1_1=np.argwhere(swab1[:,0]==1)
plt.figure()
plt.hist(Test1_no_outl[ii0_1,0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test1_no_outl[ii1_1,0],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()

plt.title('Test1')



if __name__=="__main__":
    
    cov=Covid()
    
    data_Test1, area_ROC1=cov.findROC(Test1_no_outl,swab1)
    data_Test2, area_ROC2=cov.findROC(Test2,swab)
    
    cov.prob(Test1_no_outl,swab1[:,0], "1")
    cov.prob(Test2, swab, "2")
    
    
    cov.plots(data_Test1[:,1], data_Test1[:,2], data_Test1[:,0], 1-data_Test1[:,1], "1")
    cov.plots(data_Test2[:,1], data_Test2[:,2], data_Test2[:,0], 1-data_Test2[:,1], "2")
    
    print(f"Area ROC - Test1 = {area_ROC1}\n")
    print(f"Area ROC - Test2 = {area_ROC2}\n")