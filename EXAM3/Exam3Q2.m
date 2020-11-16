clear;close all;clc;

%load('Exam3.mat','Dtrain1','label','numberOfSamples1');

%%  Generate Data
i=1;
j=1;
k=1;
numberOfClasses = 3;
numberOfSamples1 = 1000;
[Dtrain,trainlabels] = generateMultiringDataset(numberOfClasses,numberOfSamples1);
[Dtest,testlabels] = generateMultiringDataset1(numberOfClasses,10000);

%Labels1 convert to 3*10000 (0,1) matrix
for b=1:numberOfClasses
    label1(b,:)=trainlabels==b;
end

%Compute prior probability
for a = 1:size(label1,1)
    classprior(a)=length(find(label1(a,:)==1));
    prior(a)=classprior(a)/numberOfSamples1;
end

for b =1:size(label1,2)
    if label1(1,b)==1
        
        trainCla1(:,i)=Dtrain(:,b);
        i=i+1;
    elseif label1(2,b)==1
        
        trainCla2(:,j)=Dtrain(:,b);
        j=j+1;
    elseif label1(3,b)==1
        
        trainCla3(:,k)=Dtrain(:,b);
        k=k+1;
    end
end

%select the number of Gaussian components for the associated GMM
number1 = cross_val(trainCla1,length(trainCla1));
number2 = cross_val(trainCla2,length(trainCla2));
number3 = cross_val(trainCla3,length(trainCla3));

%Compute alpha,mu,Sigma of every GMM
[alpha1,mu1,Sigma1] = EMforGMM(number1,trainCla1,length(trainCla1));
[alpha2,mu2,Sigma2] = EMforGMM(number2,trainCla2,length(trainCla2));
[alpha3,mu3,Sigma3] = EMforGMM(number3,trainCla3,length(trainCla3));

%compute GMM (use evalGMM)
pdf1 = evalGMM(Dtest,alpha1,mu1,Sigma1);
pdf2 = evalGMM(Dtest,alpha2,mu2,Sigma2);
pdf3 = evalGMM(Dtest,alpha3,mu3,Sigma3);

%MAP
p_condition = [pdf1*prior(1);pdf2*prior(2);pdf3*prior(3)];
[~,value] = max(p_condition,[],1);

%compute Perror of testlabel and value
E = length(find(value == testlabels));
disp('the best model order of class 1 is : ');
disp(number1);
disp('the best model order of class 2 is : ');
disp(number2);
disp('the best model order of class 3 is : ');
disp(number3);
Perror = 1-(E/10000)


%% number of Gaussian components
function best_GMM = cross_val(X,N)

% Performs EM algorithm to estimate parameters and evaluete performance
% on each data set B times, with 1 through M GMM models considered

K = 10; M = 10;            %repetitions per data set; max GMM considered
perf_array = zeros(K,M);    %save space for performance evaluation


dummy = ceil(linspace(0,N,K+1));       % Divide the data set into K approximately-equal-sized partitions
for n = 1:K
    indPartitionLimits(n,:) = [dummy(n)+1,dummy(n+1)];
end


for k = 1:K                     % Parse data into K folds....
    
    indValidate = (indPartitionLimits(k, 1) : indPartitionLimits(k, 2));
    if k == 1
        indTrain = [indPartitionLimits(k,2)+1:N];
    elseif k == K
        indTrain = [1:indPartitionLimits(k,1)-1];
    else
        indTrain = [1:indPartitionLimits(k-1,2) indPartitionLimits(k+1,1):N];
    end
    
    xTrain = X(:,indTrain);       % Using all other folds as training set
    
    xValidate = X(:,indValidate); % Using fold k as validation set
    
    for m=1:M
        
        
        %Non-Buil-In:run EM algorith to estimate parameters
        %[alpha,mu,sigma]=EMforGMM(m,xTrain,size(xTrain,2),xValidate);
        
        %Built-In function: run EM algorithm to estimate parameters
        GMModel=fitgmdist(xTrain',m,'RegularizationValue',1e-10);
        alpha = GMModel.ComponentProportion;
        mu = (GMModel.mu)';
        sigma = GMModel.Sigma;
        
        % Calculate log-likelihood performance with new parameters
        perf_array(k,m)=sum(log(evalGMM(xValidate,alpha,mu,sigma)));
    end
end
% Calculate average performance for each M and find best fit

avg_perf =sum(perf_array)/K;
disp(avg_perf);
best_GMM = find(avg_perf == max(avg_perf),1);
end


%% Functions
function [alpha_est,mu,Sigma]=EMforGMM(M,x,N)
delta = 0.2; % tolerance for EM stopping criterion
reg_weight = 1e-2; % regularization parameter for covariance estimates
d = size(x,1); %dimensionality of data

% Initialize the GMM to randomly selected samples
alpha_est= ones(1,M)/M; %start with equal alpha estimates

% Set initial mu as random M value pairs from data array
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M));
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean

for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + reg_weight*eye(d,d);
end
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

%Run EM algorith until it converges
Converged = 0; % Not converged at the beginning
for i=1:10000  %Calculate GMM distribution according to parameters
    for l = 1:M
        temp(l,:) = repmat(alpha_est(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    pl_given_x = temp./sum(temp,1);
    %clear temp
    alpha_new = mean(pl_given_x,2);
    w = pl_given_x./repmat(sum(pl_given_x,2),1,N);
    mu_new = x*w';
    for l = 1:M
        v = x-repmat(mu_new(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        Sigma_new(:,:,l) = u*v' + reg_weight*eye(d,d); % adding a small regularization term
    end
    
    Dalpha = sum(abs(alpha_new-alpha_est'));
    Dmu = sum(sum(abs(mu_new-mu)));
    DSigma = sum(sum(abs(abs(Sigma_new-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    %     if Converged
    %         break
    %     end
    alpha_est = alpha_new; mu = mu_new; Sigma = Sigma_new;
    t=t+1;
    i=i+1;
end
end


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
% Evaluates GMM on the grid based on parameter values given
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end