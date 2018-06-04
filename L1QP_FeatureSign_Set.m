function [Z] = L1QP_FeatureSign_Set(xb, Dy, Sigma, beta, gamma)

%Get number of samples
[dFea, nSmp] = size(xb);
%Get the dictionary size
nBases = size(Dy, 2);

% sparse codes of the features
Z = sparse(nBases, nSmp); %Create a sparse matrix

%%A = B'B, c = -A'Y
A = Dy'*Dy + 2*beta*Sigma; 
%4: for i=1, 2, ..., N do
for ii = 1:nSmp
    b = -Dy'*xb(:, ii); %Multiply basis with ii sample of X
    Z(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);
end

end