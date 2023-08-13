
"""
@author: Francesco Donato

@matricola: 304810

"""
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

class Kidney:
    
    def data (self):
        
        # define the feature names:
        feat_names=['age','bp','sg','al','su','rbc','pc',
        'pcc','ba','bgr','bu','sc','sod','pot','hemo',
        'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
        'ane','classk']
        ff=np.array(feat_names)
        feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
                 'num','num','num','num','num','num','num','num','num',
                 'cat','cat','cat','cat','cat','cat','cat'])
        # import the dataframe:
        #xx=pd.read_csv("./data/chronic_kidney_disease.arff",sep=',',
        #               skiprows=29,names=feat_names, 
        #               header=None,na_values=['?','\t?'],
        #               warn_bad_lines=True)
        xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
            skiprows=29,names=feat_names, 
            header=None,na_values=['?','\t?'],)
        Np,Nf=xx.shape
        #%% change categorical data into numbers:
        key_list=["normal","abnormal","present","notpresent","yes",
        "no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
        key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
        xx=xx.replace(key_list,key_val)
        print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

        #%% manage the missing data through regression
        print(xx.info())
        x=xx.copy()
        # drop rows with less than 19=Nf-6 recorded features:
        x=x.dropna(thresh=19)
        x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
        n=x.isnull().sum(axis=1)# check the number of missing values in each row
        print('max number of missing values in the reduced dataset: ',n.max())
        print('number of points in the reduced dataset: ',len(n))
        # take the rows with exctly Nf=25 useful features; this is going to be the training dataset
        # for regression
        Xtrain=x.dropna(thresh=25)
        Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
        # get the possible values (i.e. alphabet) for the categorical features
        alphabets=[]
        for k in range(len(feat_cat)):
            if feat_cat[k]=='cat':
                val=Xtrain.iloc[:,k]
                val=val.unique()
                alphabets.append(val)
            else:
                alphabets.append('num')

        #%% run regression tree on all the missing data
        #normalize the training dataset
        mm=Xtrain.mean(axis=0)
        ss=Xtrain.std(axis=0)
        Xtrain_norm=(Xtrain-mm)/ss
        # get the data subset that contains missing values 
        Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
        Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
        Xtest_norm=(Xtest-mm)/ss # nomralization
        Np,Nf=Xtest_norm.shape
        regr=tree.DecisionTreeRegressor() # instantiate the regressor
        for kk in range(Np):
            xrow=Xtest_norm.iloc[kk]#k-th row
            mask=xrow.isna()# columns with nan in row kk
            Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
            y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
            regr=regr.fit(Data_tr_norm,y_tr_norm)
            Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1) # row vector
            ytest_norm=regr.predict(Data_te_norm)
            Xtest_norm.iloc[kk][mask]=ytest_norm # substitute nan with regressed values
        Xtest_new=Xtest_norm*ss+mm # denormalize
        # substitute regressed numerical values with the closest element in the alphabet
        index=np.argwhere(feat_cat=='cat').flatten()
        for k in index:
            val=alphabets[k].flatten() # possible values for the feature
            c=Xtest_new.iloc[:,k].values # values in the column
            c=c.reshape(-1,1)# column vector
            val=val.reshape(1,-1) # row vector
            d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
            ii=d.argmin(axis=1) # find the index of the closest alphabet value
            Xtest_new.iloc[:,k]=val[0,ii]
        print(Xtest_new.nunique())
        print(Xtest_new.describe().T)
        
        return Xtrain, Xtest_new, feat_names
        
    def shuffle (self, Xtrain, Xtest_new, seed):
        
        X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
        np.random.seed(seed)    # seeds used for shuffling: 304800, 304810, 304820
        indexsh=np.arange(len(X_new))
        np.random.shuffle(indexsh)
        X_new=X_new.set_axis(indexsh, axis=0, inplace=False)
        X_new=X_new.sort_index(axis=0)
        
        return X_new
    
    def decision_tree (self, X, feat_names):
        
        target_names = ['notckd','ckd']
        labels = X[0:158].loc[:,'classk']
        data = X[0:158].drop('classk', axis=1)
        clfX = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)    #default random_state=4
        clfX = clfX.fit(data,labels)

        test_pred = clfX.predict(X[159:349].drop('classk', axis=1))
        from sklearn.metrics import accuracy_score
        print('Accuracy =', accuracy_score(X[159:349].loc[:,'classk'],test_pred))
        from sklearn.metrics import confusion_matrix
        print('Confusion matrix')
        print(confusion_matrix(X[159:349].loc[:,'classk'],test_pred))
        
        #%% export to graghviz to draw a grahp
        # dot_data = tree.export_graphviz(clfXtrain, out_file=None,feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True) 
        # graph = graphviz.Source(dot_data) 
        # graph.render("Tree_Xtrain") 


        #black and white option
        tree.plot_tree(clfX)
        #text option
        text_representation = tree.export_text(clfX)
        print(text_representation)
        #option with colors
        fig = plt.figure(figsize=(50,50))
        tree.plot_tree(clfX,
                            feature_names=feat_names[:24],
                            class_names=target_names,
                            filled=True, rounded=True)
        fig = plt.figure(figsize=(50,50))


        return data, labels, test_pred
    
    def conf_matrix(self, X, random_state):
        
        target_names = ['notckd','ckd']
        labels = X[0:158].loc[:,'classk']
        data = X[0:158].drop('classk', axis=1)
        clfX = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
        clfX = clfX.fit(data,labels)

        test_pred = clfX.predict(X[159:349].drop('classk', axis=1))
        
        # IN ORDER TO ACCESS THE ELEMENTS OF THE CONFUSION MATRIX
        df_confusion = pd.crosstab(X[159:349].loc[:,'classk'], test_pred) 
        conf_matrix=df_confusion.to_numpy()
    
        return conf_matrix
    
    def random_forest (self, X, data_new, labels_new):
        
        
        clf=RandomForestClassifier(n_estimators=1000) 


        data_new_test = X[159:349].drop('classk', axis=1)
        labels_new_test = X[159:349].loc[:,'classk']

        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(data_new,labels_new)

        y_pred=clf.predict(data_new_test)

        # Model Accuracy
        print("Random Forest Accuracy:",metrics.accuracy_score(labels_new_test, y_pred))

        from sklearn.metrics import confusion_matrix
        print('Random Forest confusion matrix')
        print(confusion_matrix(X_new[159:349].loc[:,'classk'],y_pred))
        
        df_confusion = pd.crosstab(X[159:349].loc[:,'classk'], y_pred) 
        conf_matrix=df_confusion.to_numpy()
        
        sens=(conf_matrix[0,0]+conf_matrix[0,1])/((conf_matrix[0,0]+conf_matrix[0,1])+conf_matrix[0,1])
        
        spec=(conf_matrix[1,0]+conf_matrix[1,1])/((conf_matrix[1,0]+conf_matrix[1,1])+conf_matrix[1,0])
        
        #acc=(conf_matrix[0,0]+conf_matrix[1,1])/(conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[0,0]+conf_matrix[0,1])
        
        print(f"\nRandom Forest sensitivity: {sens}\nRandom Forest specificity: {spec}\n\n")
              
if __name__=="__main__":

    seed=304810 # seeds used for shuffling in order to check overfitting on X_new decision trees: 304800, 304810, 304820, 304830, 304840, 304850
    
    kd=Kidney()
    
    Xtrain, Xtest_new, feat_names=kd.data() 

    X= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
    
    X_new=kd.shuffle(Xtrain, Xtest_new, seed)
    
    # Xtrain decision tree
    kd.decision_tree(X, feat_names)
    
    # X_new decision tree
    data, labels, test_pred=kd.decision_tree(X_new, feat_names)
    
    # Random Forest
    kd.random_forest(X_new, data, labels)
    
    
    #ACCURACY, SENSITIVITY AND SPECIFICITY STATISTICS
    N=150
    vect_seed=np.zeros((N,), dtype=int)
    vect_seed[0]=304810
    
    for j in range(N-1):
        
        vect_seed[j+1]=vect_seed[j]+20
        
    sens=np.zeros((N,), dtype=float)
    spec=np.zeros((N,), dtype=float)
    acc=np.zeros((N,), dtype=float)
    
    
    k=0
    rs=1
    for i in vect_seed:
        
        X_new=kd.shuffle(Xtrain, Xtest_new, i)
        
        conf_matrix=kd.conf_matrix(X_new, rs)
        
        sens[k]=(conf_matrix[0,0]+conf_matrix[0,1])/((conf_matrix[0,0]+conf_matrix[0,1])+conf_matrix[0,1])
        
        spec[k]=(conf_matrix[1,0]+conf_matrix[1,1])/((conf_matrix[1,0]+conf_matrix[1,1])+conf_matrix[1,0])
        
        acc[k]=(conf_matrix[0,0]+conf_matrix[1,1])/(conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[0,0]+conf_matrix[0,1])
        
        k=k+1
        rs=rs+1
        
        
    max_se=sens[0]
    min_se=sens[0]

    max_sp=spec[0]
    min_sp=spec[0]  
    
    max_ac=acc[0]
    min_ac=acc[0]
    
    sum_se=0
    sum_sp=0
    sum_ac=0
    for t in range(N):
        if sens[t]>max_se:
            max_se=sens[t]
        if sens[t]<min_se:
            min_se=sens[t]

        if spec[t]>max_sp:
            max_sp=spec[t]
        if spec[t]<min_sp:
            min_sp=spec[t]

        if acc[t]>max_ac:
            max_ac=acc[t]
        if acc[t]<min_ac:
            min_ac=acc[t]
        
        sum_se=sum_se+sens[t]
        mean_se=sum_se/N
        
        sum_sp=sum_sp+spec[t]
        mean_sp=sum_sp/N
        
        sum_ac=sum_ac+acc[t]
        mean_ac=sum_ac/N
        
        
        
    print(f"\n\nmax sens= {max_se}\nmax spec= {max_sp}\nmax acc= {max_ac}\n\n")
    
    print(f"min sens= {min_se}\nmin spec= {min_sp}\nmin acc= {min_ac}\n\n")
    
    print(f"mean sens= {mean_se}\nmean spec= {mean_sp}\nmean acc= {mean_ac}\n\n")
    
    
    

    fig, ax =plt.subplots(1,1)
    data=[[format(max_se, ".2f"),format(min_se, ".2f"),format(mean_se, ".2f"), format(round(np.std(sens), 3))],
        [format(max_sp, ".2f"),format(min_sp, ".2f"),format(mean_sp, ".2f"), format(round(np.std(spec), 3))],
        [format(max_ac, ".2f"),format(min_ac, ".2f"),format(mean_ac, ".2f"), format(round(np.std(acc), 3))]]
    column_labels=["MAX", "MIN", "MEAN", "STD"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,colLabels=column_labels,rowLabels=["SENSITIVITY","SPECIFICITY","ACCURACY"],loc="center")
    plt.show()