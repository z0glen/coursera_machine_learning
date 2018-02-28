function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

coef = 1 / m;

h = sigmoid(X * theta);

term1 = -y .* log(h);
term2 = (1 - y) .* log(1 - h);

reg = (lambda / (2 * m)) * sum(theta(2:end) .^2);

J = coef * sum(term1 - term2) + reg;

delta = h - y;

s = X' * delta;

reg2 = lambda / m * theta(2:end);

grad(1) = coef * s(1);
grad(2:end) = coef * s(2:end) + reg2;

% =============================================================

end
