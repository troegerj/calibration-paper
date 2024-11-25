function dcost_dutheta = dcost_aao(theta,u,reaction,u_exp,mesh,lambda_r,lambda_d,varargin)

% default values
default_norm_type = 'I';
default_derivative_type = 'ANA';
default_dof_u_exp = false;

% input parser
p = inputParser;
addOptional(p,'norm_type',default_norm_type);
addOptional(p,'derivative_type',default_derivative_type);
addOptional(p,'dof_u_exp',default_dof_u_exp);
parse(p,varargin{:});

[~, dcost_dutheta] = cost_aao(theta,u,reaction,u_exp,mesh,lambda_r,lambda_d, ...
    'norm_type',p.Results.norm_type, ...
    'derivative_type',p.Results.derivative_type, ...
    'dof_u_exp',p.Results.dof_u_exp);

end













