clear all
clc
load localization.mat
    %we made a loop for K=1:120 to understand the better value of K
accuracy=zeros(120,1);
for k=1:120
right=0;
wrong=0;
    %we test on all the TestSet using j
predicted_Class=zeros(24,1);
for t=1:120
    dist=zeros(120,7,2);
    for j=1:120  
        for i=1:7 %calculating all distances from the chosen point of the TrainingSet and all TrainingSet points            
                dist(j,i,1)=abs(testdata(i,fix((t-1)/24)+1,mod(t-1,24)+1)-traindata(i,mod(j-1,5)+1,fix((j-1)/5)+1));
                dist(j,i,2)=(fix((j-1)/5))+1; %recording class
        end
    end
    count=zeros(24,1);
    for s=1:7 %ordering sitances
        dist_mat=zeros(120,2);
        for p=1:120
            dist_mat(p,1)=dist(p,s,1);
            dist_mat(p,2)=dist(p,s,2);
        end
        dist_mat=sortrows(dist_mat,1);
        dist(:,s,:)=dist_mat; 
    end
    for s=1:7 %counting K nearest
        for n=1:k
            index=dist(n,s,2);
            count(index)=count(index)+1;
        end    
    end
    [M,I]=max(count);
    
    if mod(t-1,24)+1==I
        right=right+1;
    end 
end
accuracy(k)=right/1.20;
end
plot(1:120, accuracy);