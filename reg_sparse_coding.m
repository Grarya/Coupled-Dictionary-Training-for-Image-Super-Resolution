function [D, Z, stat] = reg_sparse_coding(X, dict_size, Sigma, beta, gamma, num_iters, batch_size, initD, fname_save)
%
% Regularized sparse coding
%
% Inputs
%       X           -data samples, column wise
%       dict_size   -number of bases
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       gamma       -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%       initB       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       D           -learned dictionary
%       Z           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = dict_size;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = gamma;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

if ~isa(X, 'double')
    X = cast(X, 'double');
end

if isempty(Sigma)
	Sigma = eye(pars.num_bases);
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('Results/reg_sc_b%d_%s', dict_size, datestr(now, 30));	
end

pars

% Initialize basis - random matrix if not given 
if ~exist('initD') || isempty(initD)
    D = rand(pars.patch_size, pars.num_bases)-0.5; % Random numbers [-0.5,0.5]
	D = D - repmat(mean(D,1), size(D,1),1); % Minus the mean of every column
    D = D*diag(1./sqrt(sum(D.*D))); %Divide over the diag matrix of the norm
else
    disp('Using initial D...');
    D = initD;
end

t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

% 3: optimization loop - repeat num_trials
while t < pars.num_trials
    t=t+1;
    start_time= cputime;
    stat.fobj_total=0;    
    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
    
    sparsity = [];
    
   
    for batch=1:(size(X,2)/pars.batch_size)
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx); %Random permutation
        
        % learn conjugate gradient
        Z = L1QP_FeatureSign_Set(Xb, D, Sigma, pars.beta, pars.gamma);
        
        %Number of values ~=0 over the total number of values
        sparsity(end+1) = length(find(Z(:) ~= 0))/length(Z(:));
        
        % get objective 
        [fobj] = getObjective_RegSc(Xb, D, Z, Sigma, pars.beta, pars.gamma);       
        stat.fobj_total = stat.fobj_total + fobj;
        
        % update basis 
        D = l2ls_learn_basis_dual(Xb, Z, pars.VAR_basis);
    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    
    fprintf(['epoch= %d, sparsity = %f, fobj= %f, took %0.2f ' ...
             'seconds\n'], t, mean(sparsity), stat.fobj_avg(t), stat.elapsed_time(t));
         
    % save results
    fprintf('saving results ...\n');
    experiment = [];
    experiment.matfname = sprintf('%s.mat', pars.filename);     
    save(experiment.matfname, 't', 'pars', 'D', 'stat');
    fprintf('saved as %s\n', experiment.matfname);
end

return

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
