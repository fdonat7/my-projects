clear all
load('heightWeight.mat')
countm=0;
NROWS=210;
%punto1
for i=1:NROWS
    if heightWeightData(i,1)==1
        countm=countm+1;
    end
end
maleMatrix=zeros(countm, 2);
femaleMatrix=zeros(NROWS-countm, 2);
m=1;
f=1;
for i=1:NROWS
    if heightWeightData(i,1)==1
        maleMatrix(m,1)=heightWeightData(i,2);
        maleMatrix(m,2)=heightWeightData(i,3);
        m=m+1;
    end
    if heightWeightData(i,1)==2
        femaleMatrix(f,1)=heightWeightData(i,2);
        femaleMatrix(f,2)=heightWeightData(i,3);
        f=f+1;
    end
end
m=m-1;
f=f-1;
%punto2
%figure 
scatter(maleMatrix(:,1), maleMatrix(:,2));
%figure 
scatter(femaleMatrix(:,1), femaleMatrix(:,2));
%punto3
%figure 
hist(maleMatrix(:,1))
%figure 
hist(maleMatrix(:,2))
%figure 
hist(femaleMatrix(:,1))
%figure 
hist(femaleMatrix(:,2))
%punto4

meanMatM=(sum(maleMatrix)/m)';
meanMatF=(sum(femaleMatrix)/f)';
    %covariance matrix male
covMatM=zeros(2,2);
for i=1:m
    covMatM=covMatM+((maleMatrix(i,:)'-meanMatM)*(maleMatrix(i,:)'-meanMatM)');
end
covMatM=covMatM/m;
    %covariance matrix female
covMatF=zeros(2,2);
for i=1:f
    covMatF=covMatF+((femaleMatrix(i,:)'-meanMatF)*(femaleMatrix(i,:)'-meanMatF)');
end
covMatF=covMatF/f;
%punto5
figure
x1 = 130:1:205; x2 = 40:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],meanMatM',covMatM);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([160 205 50 130 0 max(F(:))])
xlabel('weight'); ylabel('height'); zlabel('Probability Density - males')


