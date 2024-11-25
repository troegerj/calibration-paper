function [cost_vec, dcost_vec_dutheta] = cost_vec_aao(theta,u,reaction,u_exp,mesh,lambda_r,lambda_d,varargin)

% default values
default_norm_type = 'I';
default_derivative_type = 'OFF';
default_dof_u_exp = ones(size(u),'logical');

% input parser
p = inputParser;
addOptional(p,'norm_type',default_norm_type);
addOptional(p,'derivative_type',default_derivative_type);
addOptional(p,'dof_u_exp',default_dof_u_exp);
parse(p,varargin{:});

% system matrices and vectors
K = assemble_K(theta,mesh,lambda_r);
b = assemble_b(reaction,mesh,lambda_r);

% norm kernel
if strcmp(p.Results.norm_type,'I')
    kern = eye(size(K));
elseif strcmp(p.Results.norm_type,'K')
    kern = K;
end

% scarce data
if sum(p.Results.dof_u_exp) == length(u)
    is_scarce = false;
else
    is_scarce = true;
    if strcmp(p.Results.norm_type,'K')
        error('This type of norm in the cost function requires dense data (not scarce data).')
    end
    kern = kern(p.Results.dof_u_exp,p.Results.dof_u_exp);
end
if sum(p.Results.dof_u_exp) ~= length(u_exp)
    error('The dimension of the given displacement data and the corresponding degrees of freedom do not agree.')
end

% compute cost
cost_vec = [(K*u - b); sqrt(lambda_d) * (kern*(u(p.Results.dof_u_exp)-u_exp))];

% compute derivative
if strcmp(p.Results.derivative_type,'OFF')
    dcost_vec_dutheta = zeros(length(cost_vec),length(theta)+length(u));
elseif strcmp(p.Results.derivative_type,'ANA')
    % system matrices and vectors
    A = assemble_A(u,mesh,lambda_r);
    kern_unsym = eye(size(K));
    kern_unsym = kern_unsym(p.Results.dof_u_exp,:);
    
    if strcmp(p.Results.norm_type,'I')
        dcost_vec_dtheta = [A; zeros(length(u_exp),size(A,2))];
    elseif strcmp(p.Results.norm_type,'K')
        % system matrices and vectors
        A_exp = assemble_A((u-u_exp),mesh,lambda_r);
        dcost_vec_dtheta = [A; sqrt(lambda_d) * A_exp];
    end
    dcost_vec_du = [K; sqrt(lambda_d) * kern_unsym];
    dcost_vec_dutheta = [dcost_vec_dtheta, dcost_vec_du];
elseif strcmp(p.Results.derivative_type,'FD')
    if is_scarce
        error('Not yet implemented.')
    end
    D = 1e-6;
    dcost_vec_dtheta = zeros(length(cost_vec),length(theta));
    for idx = 1:length(theta)
        theta_plus = theta; theta_plus(idx) = theta_plus(idx) + D;
        theta_minus = theta; theta_minus(idx) = theta_minus(idx) - D;
        cost_vec_plus = cost_vec_aao(theta_plus,u,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        cost_vec_minus = cost_vec_aao(theta_minus,u,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        dcost_vec_dtheta(:,idx) = (cost_vec_plus-cost_vec_minus) / (2*D);
    end
    dcost_vec_du = zeros(length(cost_vec),length(u));
    for idx = 1:length(u)
        u_plus = u; u_plus(idx) = u_plus(idx) + D;
        u_minus = u; u_minus(idx) = u_minus(idx) - D;
        cost_vec_plus = cost_vec_aao(theta,u_plus,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        cost_vec_minus = cost_vec_aao(theta,u_minus,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        dcost_vec_du(:,idx) = (cost_vec_plus-cost_vec_minus) / (2*D);
    end
    dcost_vec_dutheta = [dcost_vec_dtheta, dcost_vec_du];
end

end













