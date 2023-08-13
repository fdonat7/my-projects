# with classes
"""

@author: Francesco Donato

@matricola: 259358

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
xx=pd.read_csv("parkinsons_updrs.csv") # read the dataset
z=xx.describe().T # gives the statistical description of the content of each column
#xx.info()
# features=list(xx.columns)
features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
#%% scatter plots
todrop=['subject#', 'sex', 'test_time',  
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']
x1=xx.copy(deep=True)
X=x1.drop(todrop,axis=1)
#%% Generate the shuffled dataframe
np.random.seed(259358)
Xsh = X.sample(frac=1).reset_index(drop=True)
[Np,Nc]=Xsh.shape
F=Nc-1
#%% Generate training, validation and testing matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of testing points
X_tr=Xsh[0:Ntr] # training dataset
# find mean and standard deviations for the features in the training dataset
mm=X_tr.mean()
ss=X_tr.std()
my=mm['total_UPDRS']# get mean for the regressand
sy=ss['total_UPDRS']# get std for the regressand
# normalize data
Xsh_norm=(Xsh-mm)/ss
ysh_norm=Xsh_norm['total_UPDRS']
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
Xsh_norm=Xsh_norm.values
ysh_norm=ysh_norm.values
# get the training, validation, test normalized data
X_train_norm=Xsh_norm[0:Ntr]
X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
X_test_norm=Xsh_norm[Ntr+Nva:]
y_train_norm=ysh_norm[0:Ntr]
y_val_norm=ysh_norm[Ntr:Ntr+Nva]
y_test_norm=ysh_norm[Ntr+Nva:]
y_train=y_train_norm*sy+my
y_val=y_val_norm*sy+my
y_test=y_test_norm*sy+my

N=10
r2=3
s2=1e-3


MSE_val=np.zeros((100000,3),dtype=float)

class Plots:
    
    def regression(self, yhat_test, title):
        
        plt.figure()
        plt.plot(y_test,yhat_test,'.b')
        plt.plot(y_test,y_test,'r')
        plt.grid()
        plt.xlabel('y')
        plt.ylabel('yhat')
        plt.title(title)
        v=plt.axis()
        N1=(v[0]+v[1])*0.5
        N2=(v[2]+v[3])*0.5
        
    def regression_error_bars(self, yhat_test, sigmahat_test, title):
        
        plt.figure()
        plt.errorbar(y_test,yhat_test,yerr=3*sigmahat_test*sy,fmt='o',ms=2)
        plt.plot(y_test,y_test,'r')
        plt.grid()
        plt.xlabel('y')
        plt.ylabel('yhat')
        plt.title(title)
        v=plt.axis()
        N1=(v[0]+v[1])*0.5
        N2=(v[2]+v[3])*0.5
        
    def histogram(self, err_train, err_test, title):
        
        e=[err_train,err_test]
        plt.figure()
        plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Test'])
        plt.xlabel('error')
        plt.ylabel('P(error in bin)')
        plt.legend()
        plt.grid()
        plt.title(title)
        v=plt.axis()
        N1=(v[0]+v[1])*0.5
        N2=(v[2]+v[3])*0.5
        
    def optimal_r2_s2(self, s2_1, s2_2, s2_3, s2_4, s2_5):
        
        #%%plotting the MSE in function of r2, with a fixed s2 (with 5 chosen s2 samples)
        s2_samples=[s2_1, s2_2, s2_3, s2_4, s2_5]   
        MSE_val_s1=np.zeros((18,3),dtype=float)       
        MSE_val_s2=np.zeros((18,3),dtype=float)
        MSE_val_s3=np.zeros((18,3),dtype=float)
        MSE_val_s4=np.zeros((18,3),dtype=float)
        MSE_val_s5=np.zeros((18,3),dtype=float)
        for s2 in s2_samples:
            
           s=0
           for r2 in np.arange(1, 10, 0.5):    #r2 goes from 1 to 10 with a 0.5 step
                         
                
                yhat_val_norm,sigmahat_val=gpr.GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)  #take advantage of the validation dataset to find the optimal s2 and r2 values
                yhat_val=yhat_val_norm*sy+my
                
                if s2==s2_1:
                
                    MSE_val_s1[s,0]=r2
                    MSE_val_s1[s,1]=s2
                    MSE_val_s1[s,2]=np.mean((yhat_val-y_val)**2)
                    
                if s2==s2_2:
                
                    MSE_val_s2[s,0]=r2
                    MSE_val_s2[s,1]=s2
                    MSE_val_s2[s,2]=np.mean((yhat_val-y_val)**2)
                    
                if s2==s2_3:
                
                    MSE_val_s3[s,0]=r2
                    MSE_val_s3[s,1]=s2
                    MSE_val_s3[s,2]=np.mean((yhat_val-y_val)**2)
                
                if s2==s2_4:
                
                    MSE_val_s4[s,0]=r2
                    MSE_val_s4[s,1]=s2
                    MSE_val_s4[s,2]=np.mean((yhat_val-y_val)**2)
                
                if s2==s2_5:
                
                    MSE_val_s5[s,0]=r2
                    MSE_val_s5[s,1]=s2
                    MSE_val_s5[s,2]=np.mean((yhat_val-y_val)**2)
                
                s=s+1

        plt.figure()
                
        plt.plot(MSE_val_s1[:,0],MSE_val_s1[:,2], label="s2= 0.007")
        plt.plot(MSE_val_s2[:,0],MSE_val_s2[:,2], label="s2= 0.005")
        plt.plot(MSE_val_s3[:,0],MSE_val_s3[:,2], label="s2= 0.0008")
        plt.plot(MSE_val_s4[:,0],MSE_val_s4[:,2], label="s2= 0.0004")
        plt.plot(MSE_val_s5[:,0],MSE_val_s5[:,2], label="s2= 0.0001")
        plt.scatter(8,3.2983340610854817, label="min_MSE")
        plt.legend()
        plt.xlabel('r2')
        plt.ylabel('MSE_val')
        plt.margins(0.01,0.1)# leave some space between the max/min value and the frame of the plot
                
        plt.grid()
        plt.show()
        
    def results(self, err_train, err_test, title):
        
        #%% LLS values
        print(f'MSE train {title}',round(np.mean((err_train)**2),3))
        print(f'MSE test {title}',round(np.mean((err_test)**2),3))
        print(f'Mean error train {title}',round(np.mean(err_train),4))
        print(f'Mean error test {title}',round(np.mean(err_test),4))
        print(f'St dev error train {title}',round(np.std(err_train),3))
        print(f'St dev error test {title}',round(np.std(err_test),3))
        print(f'R^2 train {title}',round(1-np.mean((err_train)**2)/np.std(y_train**2),4))
        print(f'R^2 test {title}',round(1-np.mean((err_test)**2)/np.std(y_test**2),4))
        
        
class GPR_alg(Plots):
    
    def GPR(self, X_train,y_train,X_val,r2,s2):
           
        """ Estimates the output y_val given the input X_val, using the training data 
        and  hyperparameters r2 and s2"""
        Nva=X_val.shape[0]
        yhat_val=np.zeros((Nva,))
        sigmahat_val=np.zeros((Nva,))
        for k in range(Nva):
            x=X_val[k,:]# k-th point in the validation dataset
            A=X_train-np.ones((Ntr,1))*x
            dist2=np.sum(A**2,axis=1)
            ii=np.argsort(dist2)
            ii=ii[0:N-1];
            refX=X_train[ii,:]
            Z=np.vstack((refX,x))
            sc=np.dot(Z,Z.T)# dot products
            e=np.diagonal(sc).reshape(N,1)# square norms
            D=e+e.T-2*sc# matrix with the square distances 
            R_N=np.exp(-D/2/r2)+s2*np.identity(N)#covariance matrix
            R_Nm1=R_N[0:N-1,0:N-1]#(N-1)x(N-1) submatrix 
            K=R_N[0:N-1,N-1]# (N-1)x1 column
            d=R_N[N-1,N-1]# scalar value
            C=np.linalg.inv(R_Nm1)
            refY=y_train[ii]
            mu=K.T@C@refY# estimation of y_val for X_val[k,:]
            sigma2=d-K.T@C@K
            sigmahat_val[k]=np.sqrt(sigma2)
            yhat_val[k]=mu 
            
        return yhat_val,sigmahat_val
            
            
class LLS_alg(Plots):
    
    def LLS(self,X_train,y_train):
        
        #%% Linear Least Squares
        w_hat=np.linalg.inv(X_train.T@X_train)@(X_train.T@y_train)
        
        return w_hat 
            
  
            
  
    
if __name__=="__main__":
    
       
    gpr=GPR_alg()
    
    #%% FINDING THE OPTIMAL r2 AND s2
    s2_val=np.zeros(19)
    s2_val[0]=1e-2 

    for k in range(18):    
        
        if k!=18:           
        
            if s2_val[k]<=0.001:
                
                s2_val[k+1]=s2_val[k]-0.0001 
                
            else:
            
                s2_val[k+1]=s2_val[k]-0.001         
        
        

    i=0
    for r2 in np.arange(1, 10, 0.5):    #r2 goes from 1 to 10 with a 0.5 step
        for s2 in s2_val: #s2 goes from 0.01 to 0.0001
            
            yhat_val_norm,sigmahat_val=gpr.GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)  #take advantage of the validation dataset to find the optimal s2 and r2 values
            yhat_val=yhat_val_norm*sy+my
            
            MSE_val[i,0]=r2
            MSE_val[i,1]=s2
            MSE_val[i,2]=np.mean((yhat_val-y_val)**2)
            
                
            if r2==1 and s2==0.01:    
                
                min_MSE=MSE_val[i,2]
                r2_optimal=MSE_val[i,0]
                s2_optimal=MSE_val[i,1]
                print(f"first MSE (validation)={min_MSE}\n")
                print(f"first r2={r2_optimal}\n")
                print(f"first s2={s2_optimal}\n")
                
                
                      
            if MSE_val[i-1,2]<MSE_val[i-2,2] and MSE_val[i-1,2]<MSE_val[i,2] and MSE_val[i-1,2]<min_MSE:
                
                min_MSE=MSE_val[i-1,2]
                r2_optimal=MSE_val[i-1,0]
                s2_optimal=MSE_val[i-1,1]
                

            i=i+1


    print(f"min_MSE (validation)={min_MSE}\n")
    print(f"optimal r2={r2_optimal}\n")
    print(f"optimal s2={s2_optimal}\n")


    
    #%% Apply Gaussian Process Regression
    yhat_train_norm,sigmahat_train=gpr.GPR(X_train_norm,y_train_norm,X_train_norm,r2_optimal,s2_optimal)
    yhat_train=yhat_train_norm*sy+my

    yhat_val_norm,sigmahat_val=gpr.GPR(X_train_norm,y_train_norm,X_val_norm,r2_optimal,s2_optimal)  #take advantage of the validation dataset to find the optimal s2 and r2 values
    yhat_val=yhat_val_norm*sy+my

    yhat_test_norm,sigmahat_test=gpr.GPR(X_train_norm,y_train_norm,X_test_norm,r2_optimal,s2_optimal)
    yhat_test=yhat_test_norm*sy+my
    
    
    err_train=y_train-yhat_train
    err_test=y_test-yhat_test
    err_val=y_val-yhat_val
    
    # plot regression GPR
    gpr.regression(yhat_test, 'Gaussian Process Regression')
    
    # plot regression with error bars GPR
    gpr.regression_error_bars(yhat_test, sigmahat_test, 'Gaussian Process Regression - with errorbars')
    
    # plot GPR error histogram
    gpr.histogram(err_train, err_test, 'GPR-Error histograms')
    
    # plot r2 s2 
    gpr.optimal_r2_s2(0.007, 0.005, 0.0008, 0.0004, 0.0001)
    
    # plot results   
    gpr.results(err_train, err_test, "GPR")
    
    
    
####    
    # 19 FEATURES FOR LLS
    xx=pd.read_csv("parkinsons_updrs.csv") # read the dataset
    z=xx.describe().T # gives the statistical description of the content of each column
    #xx.info()
    # features=list(xx.columns)
    features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
           'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
           'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
           'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    #%% scatter plots
    todrop=['subject#', 'test_time']
    x1=xx.copy(deep=True)
    X=x1.drop(todrop,axis=1)
    #%% Generate the shuffled dataframe
    np.random.seed(259358)
    Xsh = X.sample(frac=1).reset_index(drop=True)
    [Np,Nc]=Xsh.shape
    F=Nc-1
    #%% Generate training, validation and testing matrices
    Ntr=int(Np*0.5)  # number of training points
    Nva=int(Np*0.25) # number of validation points
    Nte=Np-Ntr-Nva   # number of testing points
    X_tr=Xsh[0:Ntr] # training dataset
    # find mean and standard deviations for the features in the training dataset
    mm=X_tr.mean()
    ss=X_tr.std()
    my=mm['total_UPDRS']# get mean for the regressand
    sy=ss['total_UPDRS']# get std for the regressand
    # normalize data
    Xsh_norm=(Xsh-mm)/ss
    ysh_norm=Xsh_norm['total_UPDRS']
    Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
    Xsh_norm=Xsh_norm.values
    ysh_norm=ysh_norm.values
    # get the training, validation, test normalized data
    X_train_norm=Xsh_norm[0:Ntr]
    X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
    X_test_norm=Xsh_norm[Ntr+Nva:]
    y_train_norm=ysh_norm[0:Ntr]
    y_val_norm=ysh_norm[Ntr:Ntr+Nva]
    y_test_norm=ysh_norm[Ntr+Nva:]
    y_train=y_train_norm*sy+my
    y_val=y_val_norm*sy+my
    y_test=y_test_norm*sy+my
####    


    
    #%% Apply LLS Regression
    lls=LLS_alg()
    w_hat=lls.LLS(X_train_norm,y_train_norm)

    yhat_test_norm_LLS=X_test_norm@w_hat
    yhat_train_norm_LLS=X_train_norm@w_hat
    yhat_test_LLS=sy*yhat_test_norm_LLS+my
    yhat_train_LLS=sy*yhat_train_norm_LLS+my



    err_train_LLS=y_train-yhat_train_LLS
    err_test_LLS=y_test-yhat_test_LLS

    
    
    # plot regression LLS
    lls.regression(yhat_test_LLS, 'LLS Regression')
    
    # plot LLS error histogram
    lls.histogram(err_train_LLS, err_test_LLS, 'LLS-Error histograms')
      
    # plot results
    lls.results(err_train_LLS, err_test_LLS, "LLS")
    