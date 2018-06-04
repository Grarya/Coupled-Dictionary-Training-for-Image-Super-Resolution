function [Dx, Dy, error] = train_coupled_dict_SR(x, y, dict_size, lambda, upscale, Dx, Dy, gamma)

addpath(genpath('RegularizedSC'));

% dictionary training
pars = struct;
pars.patch_size = size(x,1);
pars.num_patches = size(x,2);
pars.num_bases = dict_size;
pars.num_trials = 5;
pars.gamma = gamma;
pars.lambda = lambda;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom
pars.beta = 0;

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('ResultsCD/reg_sc_b%d_%s', dict_size, datestr(now, 30));	
end

Sigma = [];
n=0;
t=1;

pars

% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

hDim = size(x, 1); %Block
lDim = size(y, 1); %Vector feature
nSmp = size(x, 2); %Number of samples
Z = sparse(dict_size, nSmp); %Create a sparse matrix

%Pre-normalize Xh and Xl !
hNorm = sqrt(sum(x.^2));
lNorm = sqrt(sum(y.^2));
Idx = find( hNorm & lNorm ); %Find index nonzero values in Xl and Xh

x = x(:, Idx);
y = y(:, Idx);

%Normalize Xh and Xl
x = x./repmat(sqrt(sum(x.^2)), size(x, 1), 1); %
y = y./repmat(sqrt(sum(y.^2)), size(y, 1), 1); %

hist_grad = 0;

 while n<pars.num_trials
        start_time= cputime;
        stat.fobj_total=0;  
        % Take a random permutation of the samples
        indperm = randperm(size(x,2));
        xb = x(:,indperm); %Random permutation
        yb = y(:,indperm);
        %t=1;
        
        %4: for i=1, 2, ..., N do
        sparsity = [];
    
        for ii = 1:nSmp
            %Dy is fixed
            %Compute the sparse representation when Dy is fixed
            b = -Dy'*yb(:, ii); %Multiply basis with ii sample of X
            A = Dy'*Dy; 
            Z(:,ii) = L1QP_FeatureSign_yang(lambda, A, b);
            
            %Compute the gradient a = d L(Dx(n), Dy(n), xi, yi)/dDy
            indzh = abs(Z(:,ii))>0;
            Zh = Z(indzh,ii);
            
            %No need to compute anything if all coef are zero
             if ~isempty(Zh)
              Dxh =Dx(:,indzh);
              Dyh =Dy(:,indzh);

              dRxdz=2*Dxh'*(Dx*Z(:,ii)-xb(:,ii)); % |Zh| x 1
              dRydz=2*Dyh'*(Dy*Z(:,ii)-yb(:,ii)); % |Zh| x 1
              dRydDy=2*(Dy*Z(:,ii)-yb(:,ii))*Z(:,ii)';  % 100 x 1024

              dDytDy = zeros(size(Dyh,2),size(Dyh,2)); 
              dDytDy(:,1) = conj(Dyh(1,:)); 
              dDytDy(1,:) = dDytDy(1,:)+ Dyh(1,:); 
              dDyyz = dDytDy*Zh; % |Zh| x 1

              dDyty = zeros(size(Dyh,2),1);
              dDyty(1,1) = yb(1,ii); %

              dzdDy = ((Dyh'*Dyh)^(-1))*(dDyty-dDyyz); % |Zh|x|Zh| * |Zh|x1

              a = 0.5*(((gamma*dRxdz+(1-gamma)*dRydz)'*dzdDy) + (1-gamma)*dRydDy);
              
              %Applying adaptative gradiente: Adagrad 
              hist_grad =+ a.^2;
              adj_grad = a./(1e-6 + sqrt(hist_grad));
              
             %Update Dy(n) = Dy(n) -n(t)*a
              C = 0.01;
              nt= adj_grad*C;
              Dy = Dy  - nt.*a;

              %Project the columns of Dy(n) onto the unil ball
              Dynorm = sqrt(sum(Dy.^2));
              Dy = Dy./repmat(Dynorm, size(Dy, 1), 1); %
              %error(t) = getObjective(xb, yb, Dx, Dy, Z,gamma);
              %t=t+1
              t=t+1;
              % get objective 
             
             end
        end
        %11: Update Dx(n+1) according to (11) and Dy(n+1): fixed
        %Compute the sparse representation when Dy is fixed
        % update basis 
        Dx = l2ls_learn_basis_dual(xb, Z, pars.VAR_basis);

        %Number of values ~=0 over the total number of values
        sparsity(end+1) = length(find(Z(:) ~= 0))/length(Z(:));
        
        % get objective 
        [fobj] = getObjective(xb, yb, Dx, Dy, Z,gamma);
        stat.fobj_total = stat.fobj_total + fobj;
       
         
        %Concatenate both dictionary to store
        B(1:hDim, :) = Dx;
        B(hDim+1:hDim+lDim,:) = Dy;
        n=n+1;
        % get statistics
        stat.fobj_avg(n)      = stat.fobj_total / pars.num_patches;
        stat.elapsed_time(n)  = (cputime - start_time);
        fprintf(['n = %d, t = %d, sparsity = %f, fobj= %f, took %0.2f ' ...
                 'seconds\n'], n, t, mean(sparsity), stat.fobj_avg(n), stat.elapsed_time(n));
      
        error(n)=stat.fobj_total / pars.num_patches;
 end
 
% save results
fprintf('saving results ...\n');
experiment = [];
experiment.matfname = sprintf('%s.mat', pars.filename);     
save(experiment.matfname, 't', 'pars', 'B', 'stat');
fprintf('saved as %s\n', experiment.matfname);
return


function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return