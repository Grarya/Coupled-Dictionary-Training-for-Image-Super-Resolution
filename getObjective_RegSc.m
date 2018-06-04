function [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(X, D, Z, Sigma, beta, gamma)

%Get the error
Err = X - D*Z;

%Compute the energy of the error
fresidue = 0.5*sum(sum(Err.^2));

%
fsparsity = gamma*sum(sum(abs(Z)));

fregs = 0;
for ii = size(Z, 1)
    fregs = fregs + beta*Z(:, ii)'*Sigma*Z(:, ii);
end

fobj = fresidue + fsparsity + fregs;