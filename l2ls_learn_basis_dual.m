function D = l2ls_learn_basis_dual(X, Z, l2norm, Binit)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_D   0.5*||X - D*Z||^2
%    subject to   ||D(:,j)||_2 <= l2norm, forall j=1...size(Z,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

L = size(X,1); %Patch size
N = size(X,2); %Number of samples
M = size(Z, 1); %Dictionary size

tic
ZZt = Z*Z';
XZt = X*Z';

if exist('Binit', 'var')
    dual_lambda = diag(Binit\XZt - ZZt);
else
    dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
end

c = l2norm^2;
trXXt = sum(sum(X.^2));

lb=zeros(size(dual_lambda));

options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');

fprintf(['fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt) answer: \n']);
fobj_basis_dual(dual_lambda, ZZt, XZt, X, c, trXXt)

%Finds a contrained minimum of a function of several variables
%x0: initial value = dual_lambda
%lb: lower bound = lb
%x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
[x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, ZZt, XZt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);
% output.iterations

fval_opt = -0.5*N*fval;
dual_lambda= x;

Dt = (ZZt+diag(dual_lambda)) \ XZt';
D_dual= Dt';
fobjective_dual = fval_opt;


D= D_dual;
fobjective = fobjective_dual;
toc

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
