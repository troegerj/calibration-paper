function [utheta, options] = solve_aao(cost,utheta_init,varargin)

% default values
default_method = 'fsolve';
default_dcost = nan;
default_theta_lb = nan;

% input parser
p = inputParser;
addOptional(p,'method',default_method);
addOptional(p,'dcost',default_dcost);
addOptional(p,'theta_lb',default_theta_lb);
parse(p,varargin{:});

% check input
if ~isa(cost,'function_handle')
    error('Cost must be given as a function handle.')
end
if isa(p.Results.dcost,'function_handle')
    dcost_given = true;
    dcost = p.Results.dcost;
else
    dcost_given = false;
end
cost_init = cost(utheta_init);
if length(cost_init) == 1
    is_vec = false;
    cost_scalar = @(utheta) cost(utheta);
else
    is_vec = true;
    cost_scalar = @(utheta) sum(cost(utheta).^2);
end
lb = -Inf(size(utheta_init));
if ~any(isnan(p.Results.theta_lb))
    lb(1:length(p.Results.theta_lb)) = p.Results.theta_lb;    
end

disp('========== Solve AAO ==========')

% initial cost
disp(['Initial cost: ' num2str(cost_scalar(utheta_init))])

% solver
if strcmp(p.Results.method,'fsolve')
    if ~dcost_given
        error('Method fsolves requires the gradient of the cost function.')
    end
    options = optimoptions('fsolve','Display','none','PlotFcn',@optimplotfirstorderopt);
    utheta = fsolve(dcost,utheta_init,options); % nonlinear solve
elseif strcmp(p.Results.method,'lsqnonlin')
    if ~is_vec
        error('Method lsqnonlin requires cost function represented as a vector.')
    end
    if ~dcost_given
        options = optimset( ...
            'MaxFunEvals',3, ...
            'TolFun',1e-9, ...
            'TolX',1e-9, ...
            'Display','iter' ... % 'off' or 'iter'
            );
        utheta = lsqnonlin(cost,utheta_init,lb,[],options);
    elseif dcost_given
        % Caution: in this case dcost must be a function handle that
        % returns both the cost and the derivative, i.e., [cost, dcost].
        options = optimoptions( ...
            @lsqnonlin, ...
            'SpecifyObjectiveGradient',true, ...
            'MaxFunEvals',500, ...
            'MaxIter',500, ...
            'TolFun',1e-10, ...
            'TolX',1e-10, ... % Norm of step
            'Display','iter' ... % 'off' or 'iter'
            );
        utheta = lsqnonlin(dcost,utheta_init,lb,[],options);
    end
    
end

% final cost
disp(['Final cost: ' num2str(cost_scalar(utheta))])

end






