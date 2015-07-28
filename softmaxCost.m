function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));  %K*MµÄ¾ØÕó
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%cost
thetaX = exp(theta*data);
normX = sum(thetaX);
elementNormThetaX = thetaX./(ones(numClasses,1)*normX);
cost =  -1 *sum( sum(groundTruth.*log(elementNormThetaX)))/numCases;

%regularization
regular = lambda*trace(theta'*theta);
cost = cost + regular;

%grad
resial = groundTruth - elementNormThetaX;

thetagrad = (resial * data')/(-numCases);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

