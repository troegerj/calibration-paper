function [cost, dcost_dutheta] = cost_aao(theta,u,reaction,u_exp,mesh,lambda_r,lambda_d,varargin)

% default values
default_norm_type = 'I';
default_derivative_type = 'OFF';
default_dof_u_exp = false;

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

% compute cost
cost = sum((K*u - b).^2) + lambda_d * sum((kern*(u-u_exp)).^2);

% compute derivative
if strcmp(p.Results.derivative_type,'OFF')
    dcost_dutheta = 0*[theta;u];
elseif strcmp(p.Results.derivative_type,'ANA')
    % system matrices and vectors
    A = assemble_A(u,mesh,lambda_r);
    A_exp = assemble_A((u-u_exp),mesh,lambda_r);

    % matrix squares
    K_sq = (K'*K);
    A_sq = (A'*A);
    A_exp_sq = (A_exp'*A_exp);
    kern_sq = (kern'*kern);

    if strcmp(p.Results.norm_type,'I')
        dcost_dtheta = 2*( ...
            A_sq*theta - A'*b ...
            );
    elseif strcmp(p.Results.norm_type,'K')
        dcost_dtheta = 2*( ...
            A_sq*theta - A'*b + lambda_d*A_exp_sq*theta ...
            );
    end
    dcost_du = 2*( ...
        K_sq*u - K'*b + lambda_d*kern_sq*(u-u_exp) ...
        );
    dcost_dutheta = [dcost_dtheta; dcost_du];
elseif strcmp(p.Results.derivative_type,'FD')
    D = 1e-6;
    dcost_dtheta = zeros(size(theta));
    for idx = 1:length(dcost_dtheta)
        theta_plus = theta; theta_plus(idx) = theta_plus(idx) + D;
        theta_minus = theta; theta_minus(idx) = theta_minus(idx) - D;
        cost_plus = cost_aao(theta_plus,u,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        cost_minus = cost_aao(theta_minus,u,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        dcost_dtheta(idx) = (cost_plus-cost_minus) / (2*D);
    end
    dcost_du = zeros(size(u));
    for idx = 1:length(dcost_du)
        u_plus = u; u_plus(idx) = u_plus(idx) + D;
        u_minus = u; u_minus(idx) = u_minus(idx) - D;
        cost_plus = cost_aao(theta,u_plus,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        cost_minus = cost_aao(theta,u_minus,reaction,u_exp,mesh,lambda_r,lambda_d, ...
            'norm_type',p.Results.norm_type,'dof_u_exp',p.Results.dof_u_exp);
        dcost_du(idx) = (cost_plus-cost_minus) / (2*D);
    end
    dcost_dutheta = [dcost_dtheta; dcost_du];
end

end













