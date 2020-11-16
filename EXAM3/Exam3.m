clear;close all;clc;

%%Generate Data
numberOfClasses = 3;
numberOfSamples1 = 1000;

[Dtrain1,labels1] = generateMultiringDataset(numberOfClasses,numberOfSamples1);

%Labels1 convert to 3*100 (0,1) matrix
for a=1:numberOfClasses                         
    label(a,:)=labels1==a;
end

numberOfSamples2 = 10000;
[Dtest,labels2] = generateMultiringDataset1(numberOfClasses,numberOfSamples2);

%Labels1 convert to 3*10000 (0,1) matrix
for b=1:numberOfClasses                         
    label1(b,:)=labels2==b;
end

%Choosing the maximum number of perceptrons
perceptrons = 10;
%Choosing the number of folds
fold = 10;

PS = struct();%Store the parameters

%Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,numberOfSamples1,fold+1));
for k = 1:fold
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

%% LOOP
for nPerceptrons = 1:perceptrons
    for j=1:2
        
        % Initialize model parameters
        params.A = rand(nPerceptrons,2);
        params.b = rand(nPerceptrons,1);
        params.C = rand(3,nPerceptrons);
        params.d = mean(label,2);%;rand(nY,1) % initialize to mean of y
        %K-Fold Cross-Validation
        for i = 1:fold
            indValidate = [indPartitionLimits(i,1):indPartitionLimits(i,2)];
            %Using fold k as validation set
            x1Validate = Dtrain1(1,indValidate);
            x2Validate = Dtrain1(2,indValidate);
            xValidate = [x1Validate; x2Validate];
            yValidate = label(:,indValidate);
            if i == 1
                indTrain = [indPartitionLimits(i,2)+1:numberOfSamples1];
            elseif i == fold
                indTrain = [1:indPartitionLimits(i,1)-1];
            else
                indTrain = [1:indPartitionLimits(i-1,2),indPartitionLimits(i+1,1):numberOfSamples1];
            end
            
            %Using all other folds as training set
            x1Train = Dtrain1(1,indTrain);
            x2Train = Dtrain1(2,indTrain);
            xTrain = [x1Train; x2Train];
            yTrain = label(:,indTrain);
            Ntrain = length(indTrain); Nvalidate = length(indValidate);
            
             % Determine/specify sizes of parameter matrices/vectors
        nX = size(xTrain,1);%2;
        nY = size(yTrain,1);%3;
        sizeParams = [nX;nPerceptrons;nY];
            
            
            %params = paramsTrue;
            vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
            %vecParamsInit = vecParamsTrue; % Override init weights with true weights
            
            % Optimize model
            options = optimset('MaxFunEvals',5000,'MaxIter',5000);
            vecParams = fminsearch(@(vecParams)(objectiveFunction(xTrain,yTrain,sizeParams,vecParams,j)),vecParamsInit,options);
            
            % Visualize model output for training data
            params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
            params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
            params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
            params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
            
            
            H = mlpModel(xValidate,params,j);
            Er(nPerceptrons,i)=Error(yValidate,H);        %Calculate overall error per fold
            if j==1
                avgError1(nPerceptrons)=mean(Er(nPerceptrons,:));          %average fold error
            else
                avgError2(nPerceptrons)=mean(Er(nPerceptrons,:));          %average fold error
            end
        end
        PS(j,nPerceptrons).A = params.A;
    PS(j,nPerceptrons).b = params.b;
    PS(j,nPerceptrons).C = params.C;
    PS(j,nPerceptrons).d = params.d;
        j=j+1;
    end
    nPerceptrons = nPerceptrons + 1;
end
%save('Exam3.mat','Dtrain1','label','numberOfSamples1');
%% Find best combination of model order and actiation nonlinearity
avgError = [avgError1;avgError2];
[val,idx] = min(avgError(:));
[row,col] = ind2sub(size(avgError),idx);

%% Train an MLP with specification using Dtrain
% Determine/specify sizes of parameter matrices/vectors
nX = 2;%size(X,1);
nY = 3;
nPerceptrons = col;
sizeParams = [nX;nPerceptrons;nY];

% Initialize model parameters
params.A = PS(row,nPerceptrons).A;%zeros(nPerceptrons,nX);
params.b = PS(row,nPerceptrons).b;%zeros(nPerceptrons,1);
params.C = PS(row,nPerceptrons).C;%zeros(nY,nPerceptrons);
params.d = PS(row,nPerceptrons).d;%mean(Y,2);%zeros(nY,1); % initialize to mean of y

vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
%vecParamsInit = vecParamsTrue; % Override init weights with true weights

% Optimize model
options = optimset('MaxFunEvals',5000,'MaxIter',5000);
vecParams = fminsearch(@(vecParams)(objectiveFunction(Dtrain1,label,sizeParams,vecParams,row)),vecParamsInit,options);

% Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

H = mlpModel(Dtest,params,row);
ferror=Error(label1,H);
disp(["Accuracy:",1-ferror])

%% Plot

Class = H==max(H);
lr = size(label1,1);
lc = size(label1,2);
  
for a=1:lr
   
    for b=1:lc
        if all(Class(lr,lc)==label1(lr,lc))
            plot(Dtest(1,b),Dtest(2,b),'.g');    % correct classification
            
        elseif any(Class(h,i)~=label(h,i))
            plot(Dtest(1,i),Dtest(2,i),'.r');   % wrong classification
        end
    end
end
% figure(3), clf, plot5(Dtest(1,:),Dtest(2,:),H(1,:),H(2,:),H(3,:),'.g');
% figure(3), hold on, plot5(Dtest(1,:),Dtest(2,:),label1(1,:),label1(2,:),label1(3,:),'.r');
xlabel('X_1'); ylabel('X_2');
title('Final Performance')
% figure(3), clf, plot(label1,H,'.'); axis equal,
% xlabel('Desired Output'); ylabel('Model Output');
% title('Model Output Visualization For Training Data')

%% Functions
function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams,type)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params,type);
%objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
objFncValue = sum(-sum(Y.*log(H),1),2)/N;
% Change objective function to make this MLE for class posterior modeling
end

function H = mlpModel(X,params,type)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U,type);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
%H = V; % linear output layer activations
H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
end

function out = activationFunction(in,type)
if type==1
    out = 1./(1+exp(-in)); % logistic function
else
    out = in./sqrt(1+in.^2); % ISRU
end
end

function error= Error(yValidate,H)
Class = H==max(H);      %make max probability 1 and other ones 0
A=size(yValidate,1);
B=size(yValidate,2);
C=0;
for i=1:A
    for j=1:B
        if yValidate(i,j)==1 && Class(i,j)==1
            C=C+1;
        end
        j=j+1;
    end
    i=i+1;
end
error=1-(C/B);
end
