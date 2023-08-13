% Lab 4

clear all
close all

% Simulation of object motion

N=100; % number of time instants
Delta=1.5;
A=[1 0 Delta 0; 0 1 0 Delta; 0 0 1 0; 0 0 0 1];
sigma_Qx= 2; %2  %0.005  %5
sigma_Qv= 0.5; %0.5  %0.005  %5
epsilon=zeros(4,N);
epsilon(1:2,:)=sigma_Qx*randn(2,N);
epsilon(3:4,:)=sigma_Qv*randn(2,N);
z=zeros(4,N); % state vector (over time)
z(:,1)=[0 0 Delta Delta].'; % Initial state: coordinates at time 0 are (0,0)
for i=2:N
    z(:,i)=A*z(:,i-1)+epsilon(:,i);
    %sum=sum+z(:,i)
end

%mu=sum/N
%%%%


C=[1 0 0 0 ; 0 1 0 0];
sigma_R=20;
delta=sigma_R*randn(2,N);
y=zeros(2,N);
y(:,1)=[0 0].';
for i=2:N
    y(:,i)=C*z(:,i)+delta(:,i);
end

% This figure plots object motion trajectory
figure
plot(z(1,:),z(2,:))
hold on
%plot(y(1,:),y(2,:))

mu=[0; 0;Delta;Delta];
sigma=[1 0 Delta 0; 0 1 0 Delta; 0 0 1 0; 0 0 0 1];

%mu=[0; 1;5;5];
%sigma=[1 5 4 0; 0 4 2 10; 2 2 2 2; 6 4 5 8];
y_hat=zeros(2,N);
mu_vect=zeros(4,N);
sigma_err=[(sigma_Qx)^2,0,0,0;0,(sigma_Qx)^2,0,0;0,0,(sigma_Qv)^2,0;0,0,0,(sigma_Qv)^2] %covarianza dell'errore sul sistema
for i=2:N
    mu=A*mu;
    sigma=A*sigma*A'+sigma_err;
    y_hat(:,i)=C*mu;
    k=sigma*C'*inv((C*sigma*C'+eye(2)*(sigma_R)^2));
    r=y(:,i)-y_hat(:,i);
    mu=mu+k*r;
    sigma=(eye(4)-k*C)*sigma;
    mu_vect(:,i)=mu;
end
plot(z(1,:),z(2,:),'b')
hold on
plot(y(1,:),y(2,:),'r')
plot(mu_vect(1,:),mu_vect(2,:),'g')
legend('True coordinate','Observed coordinate','Estimated coordinate')
figure
plot(1:1:100,z(1,:),'b')
hold on
plot(1:1:100,y(1,:),'r')
plot(1:1:100, mu_vect(1,:),'g')
legend('True coordinate','Observed coordinate','Estimated coordinate')
hold off


