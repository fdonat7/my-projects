clear all
load('heightWeight.mat')
countm=0;
NROWS=210;
for i=1:NROWS
    if heightWeightData(i,1)==1
        countm=countm+1;
    end
end
countf=NROWS-countm;

%creating test matrix
maleMatrix=zeros(countm, 3);
femaleMatrix=zeros(countf, 3);
m=1;
f=1;
for i=1:NROWS
    if heightWeightData(i,1)==1
        maleMatrix(m,1)=heightWeightData(i,2);
        maleMatrix(m,2)=heightWeightData(i,3);
        maleMatrix(m,3)=heightWeightData(i,1);
        m=m+1;
    end
    if heightWeightData(i,1)==2
        femaleMatrix(f,1)=heightWeightData(i,2);
        femaleMatrix(f,2)=heightWeightData(i,3);
        femaleMatrix(f,3)=heightWeightData(i,1);
        f=f+1;
    end
end
testMatrix=zeros(55,3);
j=1;
for t=countm-19:countm
    testMatrix(j,1)=maleMatrix(t,1);
    testMatrix(j,2)=maleMatrix(t,2);
    testMatrix(j,3)=maleMatrix(t,3);

    j=j+1;
end
for g=countf-34:countf
    testMatrix(j,1)=femaleMatrix(g,1);
    testMatrix(j,2)=femaleMatrix(g,2);
    testMatrix(j,3)=femaleMatrix(g,3);

    j=j+1;
end
m=m-20;
f=f-35;
meanMatM=(sum(maleMatrix(1:countm-19,1:2))/m)';
meanMatF=(sum(femaleMatrix(1:countf-34,1:2))/f)';
    %covariance matrix male
covMatM=zeros(2,2);
for i=1:m
    covMatM=covMatM+((maleMatrix(i,1:2)'-meanMatM)*(maleMatrix(i,1:2)'-meanMatM)');
    %covMatM=covMatM+(maleMatrix(i,1:2)'*maleMatrix(i,1:2))-meanMatM'*meanMatM;

end
covMatM=covMatM/m;
    %covariance matrix female
covMatF=zeros(2,2);
for i=1:f
    covMatF=covMatF+((femaleMatrix(i,1:2)'-meanMatF)*(femaleMatrix(i,1:2)'-meanMatF)');
    %covMatF=covMatF+(femaleMatrix(i,1:2)'*femaleMatrix(i,1:2))-meanMatF'*meanMatF;
end
covMatF=covMatF/f;
assigned_class=zeros(55,2);
%FIRST METHOD
p_yc_givenxteta_M=zeros(55,2);
p_yc_givenxteta_F=zeros(55,2);
right=0;
wrong=0;
for i=1:55    
    %p_yc_givenxteta_M(i,1)=0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatM)'*(inv(covMatM))*(testMatrix(i,1:2)'-meanMatM))/(0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatM)'*(inv(covMatM))*(testMatrix(i,1:2)'-meanMatM)))+(0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatF)'*(inv(covMatF))*(testMatrix(i,1:2)'-meanMatF)));
    %p_yc_givenxteta_F(i,1)=0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatF)'*(inv(covMatF))*(testMatrix(i,1:2)'-meanMatF))/(0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatM)'*(inv(covMatM))*(testMatrix(i,1:2)'-meanMatM)))+(0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)'-meanMatF)'*(inv(covMatF))*(testMatrix(i,1:2)'-meanMatF)));

    p_yc_givenxteta_M(i,1)=0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM))*(testMatrix(i,1:2)-meanMatM')')/(0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM))*(testMatrix(i,1:2)-meanMatM')')+0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF))*(testMatrix(i,1:2)-meanMatF')'));
    p_yc_givenxteta_F(i,1)=0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF))*(testMatrix(i,1:2)-meanMatF')')/(0.5*(det(2*pi*covMatM))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM))*(testMatrix(i,1:2)-meanMatM')')+0.5*(det(2*pi*covMatF))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF))*(testMatrix(i,1:2)-meanMatF')'));
    if p_yc_givenxteta_M(i,1)>p_yc_givenxteta_F(i,1)
        assigned_class(i,1)=1;
        if testMatrix(i,3)==1 
            right=right+1;
        else            
            wrong=wrong+1;        
        end        
    end
    if p_yc_givenxteta_M(i,1)<p_yc_givenxteta_F(i,1)
        assigned_class(i,1)=2;
        if testMatrix(i,3)==2
            right=right+1;
        else          
            wrong=wrong+1;       
        end        
    end
end
accuracy_1=right/(right+wrong)

%SECOND METHOD 
covMatM_2=covMatM;
covMatF_2=covMatF;
for i=1:2
    for j=1:2
        if i~=j         
            covMatM_2(i,j)=0;
            covMatF_2(i,j)=0;
        end
    end
end

%p_yc_givenxteta_2_M=zeros(55,1);
%p_yc_givenxteta_2_F=zeros(55,1);

right=0;
wrong=0;
for i=1:55    
    p_yc_givenxteta_M(i,2)=0.5*(2*pi*det(covMatM_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM_2))*(testMatrix(i,1:2)-meanMatM')')/(0.5*(det(2*pi*covMatM_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM_2))*(testMatrix(i,1:2)-meanMatM')')+0.5*(det(2*pi*covMatF_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF_2))*(testMatrix(i,1:2)-meanMatF')'));
    p_yc_givenxteta_F(i,2)=0.5*(2*pi*det(covMatF_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF_2))*(testMatrix(i,1:2)-meanMatF')')/(0.5*(det(2*pi*covMatM_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatM')*(inv(covMatM_2))*(testMatrix(i,1:2)-meanMatM')')+0.5*(det(2*pi*covMatF_2))^(-0.5)*exp(-0.5*(testMatrix(i,1:2)-meanMatF')*(inv(covMatF_2))*(testMatrix(i,1:2)-meanMatF')'));   
    if p_yc_givenxteta_M(i,2)>p_yc_givenxteta_F(i,2)
        assigned_class(i,2)=1;
        if testMatrix(i,3)==1
            right=right+1;
        else            
            wrong=wrong+1;        
        end        
    end
    if p_yc_givenxteta_M(i,2)<p_yc_givenxteta_F(i,2)
        assigned_class(i,2)=2;
        if testMatrix(i,3)==2
            right=right+1;
        else            
            wrong=wrong+1;       
        end        
    end
end
accuracy_2=right/(right+wrong)

%THIRD METHOD
trainingMatrix=zeros(210-55, 2);
for i=1:m
    trainingMatrix(i,1)=maleMatrix(i,1);
    trainingMatrix(i,2)=maleMatrix(i,2);
end
for j=1:f
    trainingMatrix(i,1)=femaleMatrix(j,1);
    trainingMatrix(i,2)=femaleMatrix(j,2);
    i=i+1;
end
tot_train=m+f-2;
meanMat_tot=(sum(trainingMatrix(1:tot_train,:))/tot_train)';
covMat_tot=zeros(2,2);

for i=1:tot_train
    covMat_tot=covMat_tot+((trainingMatrix(i,:)'-meanMat_tot)*(trainingMatrix(i,:)'-meanMat_tot)');
end
covMat_tot=covMat_tot/tot_train;

tot_test=55;
p_yc_givenxteta_3_M=zeros(tot_test, 1);
p_yc_givenxteta_3_F=zeros(tot_test, 1);
beta_cM=(inv(covMat_tot)*meanMatM);
beta_cF=(inv(covMat_tot)*meanMatF);
gamma_cM=(-0.5*meanMatM'*inv(covMat_tot)*meanMatM+log(0.5));
gamma_cF=(-0.5*meanMatF'*inv(covMat_tot)*meanMatF+log(0.5));
%p_yc_givenxteta_3_M=(exp(beta_cM'*testMatrix(i,1:2)'+gamma_cM))/(exp(beta_cM'*testMatrix(i,1:2)'+(gamma_cM))+exp(beta_cF'*testMatrix(i,1:2)'+(gamma_cF)));
%p_yc_givenxteta_3_F=(exp(beta_cF'*testMatrix(i,1:2)'+gamma_cF))/(exp(beta_cM'*testMatrix(i,1:2)'+(gamma_cM))+exp(beta_cF'*testMatrix(i,1:2)'+(gamma_cF)));
right=0;
wrong=0;

for i=1:55    
    p_yc_givenxteta_3_M(i)=(exp(beta_cM'*testMatrix(i,1:2)'+gamma_cM))/(exp(beta_cM'*testMatrix(i,1:2)'+(gamma_cM))+exp(beta_cF'*testMatrix(i,1:2)'+(gamma_cF)));
    p_yc_givenxteta_3_F(i)=(exp(beta_cF'*testMatrix(i,1:2)'+gamma_cF))/(exp(beta_cM'*testMatrix(i,1:2)'+(gamma_cM))+exp(beta_cF'*testMatrix(i,1:2)'+(gamma_cF)));    
    if p_yc_givenxteta_3_M(i,1)>p_yc_givenxteta_3_F(i,1)
        assigned_class(i,1)=1;
        if testMatrix(i,3)==1
            right=right+1;
        else            
            wrong=wrong+1;        
        end
    end
    if p_yc_givenxteta_3_M(i,1)<p_yc_givenxteta_3_F(i,1)
        assigned_class(i,1)=2;
        if testMatrix(i,3)==2
            right=right+1;
        else
            wrong=wrong+1;        
        end
    end
end
accuracy_3=right/(right+wrong)
