clear all
load 'XwindowsDocData.mat'
MAXTAU=500;
interval=0.25;
N=900;
D=600;
Njc=zeros(D,2);
Nc=zeros(2,1);
for i=1:N
    c=ytrain(i);
    Nc(c)=Nc(c)+1;
    for j=1:D
        if xtrain(i,j)==1
            Njc(j,c)=Njc(j,c)+1;
        end
    end
end
pi=zeros(2,1);
pi(1)=Nc(1)/N;
pi(2)=Nc(2)/N;
teta=zeros(600,2);
for i=1:D
    teta(i,1)=Njc(i,1)/Nc(1);
    teta(i,2)=Njc(i,2)/Nc(2);
end
figure
plot(1:1:600, teta(:,1))
figure
plot(1:1:600, teta(:,2))
uninf_err=0.002;
uninf=zeros(D,1);
count=0;
for i=1:D
    if abs(teta(i,1)-teta(i,2))<uninf_err
        count=count+1;
        uninf(i)=1;
    end
end
TPR=zeros(10000,1);
FPR=zeros(10000,1);
count=0;
for tau=0:interval:MAXTAU
    tau
    count=count+1;
    predicted_class_train=zeros(900,1);
    true_positive=0;
    false_positive=0;
    prob_class_train=zeros(900,2);
    for j=1:900
        p_x_givenyc=zeros(2,1);
        for i=1:D
            if xtrain(j,i)==1
                p_x_givenyc(1,1)=p_x_givenyc(1,1)+log(teta(i,1));
            elseif (teta(i,1))>0 && xtrain(j,i)==0
                p_x_givenyc(1,1)=p_x_givenyc(1,1)+log(1-teta(i,1));
            end
            if xtrain(j,i)==1
                p_x_givenyc(2,1)=p_x_givenyc(2,1)+log(teta(i,2));
            elseif (teta(i,2))>0 && xtrain(j,i)==0
                p_x_givenyc(2,1)=p_x_givenyc(2,1)+log(1-teta(i,2));
            end
        end 
        prob_class_train(j,1)=p_x_givenyc(1);
        prob_class_train(j,2)=p_x_givenyc(2);
        if p_x_givenyc(1)-p_x_givenyc(2)>log(tau)
            predicted_class_train(j)=1;
            if predicted_class_train(j)==ytrain(j)
                true_positive=true_positive+1;                
            else
                false_positive=false_positive+1;
            end
        end
        %   predicted_class=rand
        
    end    
    TPR(count)=true_positive/Nc(1);
    FPR(count)=false_positive/Nc(2);
end    
plot(FPR, TPR)
title('tau=0:0.5:1000')%con tau=0:0.1:5 e tau=0:0.01:5 non è uscito bene (FPR si ferma a 0.25 e TPR a 0.5)