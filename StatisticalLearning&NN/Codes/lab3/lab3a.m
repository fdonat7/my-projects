clear all
load('Indian_Pines_Dataset.mat')
rows=145;
columns=145;
class_a=1;
class_b=2;
count_a=0;
count_b=0;
%conta i vettori di lcasse a e b
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

c_a=0;
c_b=0;
%salva i vettori spettrali di classe a e b
for i=1:1:rows
    for j=1:1:columns
        if indian_pines_gt(i,j) == class_a
            c_a=c_a+1;
            spec_a(c_a,:)=indian_pines(i,j,:);
        elseif indian_pines_gt(i,j) == class_b
            c_b=c_b+1;
            spec_b(c_b,:)=indian_pines(i,j,:);
        end
    end
end
x=[spec_a;spec_b];
%salviamo le medie
mean_values=zeros(220,1);
for s=1:1:220
    sum=0;
    for i=1:1:count_a+count_b
        sum=sum+x(i,s);
    end
    mean_values(s)=sum/(count_a+count_b);
    for i=1:1:count_a+count_b
        x(i,s)=x(i,s)-sum/(count_a+count_b);
    end
end

%covarinace matrix
sigma=0
for k=1:1:count_a+count_b
    sigma=sigma+1/(count_a+count_b)*x(k,:)'*x(k,:);
end
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
        if i>1470
            ciao=3;
        end
    end
    MSE_vect(k)=MSE
end

plot(1:1:220, MSE_vect);

