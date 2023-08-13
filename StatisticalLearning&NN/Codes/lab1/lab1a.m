clc
load synthetic.mat
    %we made a loop for K=1:100 to understand the better value of K
accuracy_train=zeros(100,1);
accuracy_test=zeros(100,1);
for K=1:100
   right=0; 
   wrong=0;
    %we test on all the TestSet using j
    predicted_Class=zeros(100,1);
for j=1:100
    dist=zeros(100,2);
    for i=1:100 %calculating all distances from the chosen point of the TrainingSet and all TrainingSet points
        dist(i,1)=(knnClassify2dTrain(i,1)-knnClassify2dTest(j,1))^2+(knnClassify2dTrain(i,2)-knnClassify2dTest(j,2))^2;
        dist(i,2)=knnClassify2dTrain(i,3);
    end
    dist=sortrows(dist,1);%ordering distances
    countClass1=0;
    countClass2=0;
    for l=1:K %seeing classes of the nearest K points to make the prediction
        if dist(l,2)==1
            countClass1=countClass1+1;
        elseif dist(l,2)==2
            countClass2=countClass2+1;
        end
    end
	%make the prediction and verifying it
    if countClass1>countClass2 
        predicted_Class(j)=1;
    elseif countClass1<countClass2 
        predicted_Class(j)=2;
    elseif countClass1==countClass2 
         if (rand(1,2)>=0.5)
                predicted_Class(j)=1;
         else
             predicted_Class(j)=2;
         end
    end
    if predicted_Class(j)==knnClassify2dTest(j,3)
        right=right+1;
    else
        wrong=wrong+1;
    end
    
end
accuracy_test(K)=(right/(right+wrong))*100;
end
for K=1:100
 right=0; 
 wrong=0;
    %we test on all the TestSet using j
    predicted_Class=zeros(100,1);
for j=1:100
    dist=zeros(100,2);
    for i=1:100 %calculating all distances from the chosen point of the TrainingSet and all TrainingSet points
        dist(i,1)=(knnClassify2dTrain(i,1)-knnClassify2dTrain(j,1))^2+(knnClassify2dTrain(i,2)-knnClassify2dTrain(j,2))^2;
        dist(i,2)=knnClassify2dTrain(i,3);
    end
    dist=sortrows(dist,1);%ordering distances
    countClass1=0;
    countClass2=0;
    for l=1:K %seeing classes of the nearest K points to make the prediction
        if dist(l,2)==1
            countClass1=countClass1+1;
        elseif dist(l,2)==2
            countClass2=countClass2+1;
        end
    end
	%make the prediction and verifying it
    if countClass1>countClass2 
        predicted_Class(j)=1;
    elseif countClass1<countClass2 
        predicted_Class(j)=2;
    elseif countClass1==countClass2 
         if (rand(1,2)>=0.5)
                predicted_Class(j)=1;
         else
             predicted_Class(j)=2;
         end
    end
    if predicted_Class(j)==knnClassify2dTest(j,3)
        right=right+1;
    else
        wrong=wrong+1;
    end
    
end
accuracy_train(K)=(right/(right+wrong))*100;
end
figure
plot(1:100, accuracy_test);
hold on
plot(1:100, accuracy_train);