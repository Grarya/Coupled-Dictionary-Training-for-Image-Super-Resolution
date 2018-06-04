
function [f,g,H] = fobj_basis_dual(dual_lambda, ZZt, XZt, X, c, trXXt)

% Compute the objective function value at x
L= size(XZt,1);
M= length(dual_lambda);

ZZt_inv = inv(ZZt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(ZZt_inv*(XZt'*XZt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XZt*ZZt_inv*XZt')+trXXt-c*sum(dual_lambda);
end
f= -f;

%If gradient or Hessian required
if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XZt*ZZt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    %If Hessian required
    if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((XZt*ZZt_inv'*XZt*ZZt_inv).*ZZt_inv);
        H = -2.*((temp'*temp).*ZZt_inv);
        H = -H;
    end
end

return