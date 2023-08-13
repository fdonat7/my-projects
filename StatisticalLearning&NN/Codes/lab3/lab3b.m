clear all
load('Indian_Pines_Dataset.mat')
rows=145;
columns=145;
class_a=0;
class_b=1;
count_a=0;
count_b=0;
%conta i vettori di classe a e b
for i=1:1:rows
    for j=1:1:columns
        if indian_pines_gt(i,j) == class_a
            count_a=count_a+1;
        elseif indian_pines_gt(i,j) == class_b
            count_b=count_b+1;
        end
    end
end
spec_a=zeros(count_a,220);
spec_b=zeros(count_b,220);
x_class=zeros(count_b+count_a,1);
c_a=0;
c_b=0;
%salva i vettori spettrali di classe a e b
c=1;
for i=1:1:rows
    for j=1:1:columns
        if indian_pines_gt(i,j) == class_a
            x_class(c)=class_a;
            c=c+1;
            c_a=c_a+1;
            spec_a(c_a,:)=indian_pines(i,j,:);
        elseif indian_pines_gt(i,j) == class_b
            x_class(c)=class_b;
            c=c+1;
            c_a=c_a+1;
            c_b=c_b+1;
            spec_b(c_b,:)=indian_pines(i,j,:);
        end
    end
end
x=[spec_a;spec_b];
%salviamo le medie
mean_values_a=zeros(220,1);
mean_values_b=zeros(220,1);

for s=1:1:220
    sum_a=0;
    sum_b=0;
    for i=1:1:count_a
        sum_a=sum_a+spec_b(i,s);
    end
    for i=1:1:count_b
        sum_b=sum_b+spec_b(i,s);
    end
    mean_values_a(s)=sum/(count_a);
    mean_values_a(s)=sum/(count_a);
end
%covarinace matrix
sigma=zeros(220,220);
for k=1:1:count_a+count_b
    sigma=sigma+1/(count_a+count_b)*x(k,:)'*x(k,:);
end
sigma=eye(220);
%sorted eigenvectors
[e_vect, e_val]=eig(sigma);
[e_v, ind]=sort(diag(e_val));
e_val_sorted=e_val(ind,ind);  
e_vect_sorted=e_vect(:,ind);
MSE_vect=zeros(220,1);
for k=1:1:220
    W=e_vect_sorted(:,220-k+1:1:220);
%PCA coefficients
    z=zeros(count_a+count_b,k);
    for l=1:1:count_a+count_b
        z(l,:)=W'*x(l,:)';
    end
    x_hat=zeros(count_a+count_b,220);
    MSE=0;
    for i=1:1:count_a+count_b
        x_hat(i,:)=W*z(i,:)';
        MSE=MSE+1/(count_a+count_b)*(x(i,:)-x_hat(i,:))*(x(i,:)-x_hat(i,:))';
    end
    MSE_vect(k)=MSE
end

plot(1:1:220, sqrt(MSE_vect));

%%%%%%%%%%% 2 %%%%%%%%%%%%%%%%

%training and testset
train_rows=fix(rows*0.75);
test_rows=rows-train_rows;

mu_0=zeros(220,1);
mu_1=zeros(220,1);
count_0=0;
count_1=0;
%mean vector of the 2 classes
for s=1:1:220
    for i=1:1:train_rows
        for j=1:1:columns
            if indian_pines_gt(i,j)==0
                count_0=count_0+1;
                mu_0(s)=mu_0(s)+indian_pines(i,j,s)/1000;
            elseif indian_pines_gt(i,j)==1
                count_1=count_1+1;
                mu_1(s)=mu_1(s)+indian_pines(i,j,s)/1000;
            end    
        end
    end
end
for s=1:1:220
    mu_0(s)=mu_0(s)/count_0*1000;
    mu_1(s)=mu_1(s)/count_1*1000;
end
x0=(mu_0+mu_1)/2;
w=mu_1-mu_0;


right=0;
wrong=0;
for i=1:10822
    if sign(w'*(x(i,:)'-x0))==-1 & x_class(i)==0
        right=right+1;
    else 
        wrong=wrong+1;
    end

    if sign(w'*(x(i,:)'-x0))==1 & x_class(i)==1
        right=right+1;
    else 
        wrong=wrong+1;
    end
end

accuracy=right/(right+wrong)