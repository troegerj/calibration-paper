function opt_para = set_optPara_hyperelas(const_mod)

if strcmp(const_mod.constitutiveModel,'NH') % Neo-Hooke
    % set elastic parameters for optimization
    % K
    opt_para(1).label           = 'K';
    opt_para(1).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(1).initial_value   = 1.000000E+00;                 % initial value [N/mm^2]
    opt_para(1).value           = opt_para(1).initial_value;
    opt_para(1).lb              = -Inf;                         % lower bound
    opt_para(1).ub              = Inf;                          % upper bound
    %
    % c10
    opt_para(2).label           = 'c10';
    opt_para(2).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(2).initial_value   = 1.000000E-01;                 % initial value [N/mm^2]
    opt_para(2).value           = opt_para(2).initial_value;
    opt_para(2).lb              = -Inf;                         % lower bound
    opt_para(2).ub              = Inf;                          % upper bound
    %
elseif strcmp(const_mod.constitutiveModel,'IH') % Isihara
    % set elastic parameters for optimization
    % K
    opt_para(1).label           = 'K';
    opt_para(1).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(1).initial_value   = 1.000000E+00;                 % initial value [N/mm^2]
    opt_para(1).value           = opt_para(1).initial_value;
    opt_para(1).lb              = -Inf;                         % lower bound
    opt_para(1).ub              = Inf;                          % upper bound
    %
    % c10
    opt_para(2).label           = 'c10';
    opt_para(2).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(2).initial_value   = 4.000000E-01;                 % initial value [N/mm^2]
    opt_para(2).value           = opt_para(2).initial_value;
    opt_para(2).lb              = -Inf;                         % lower bound
    opt_para(2).ub              = Inf;                          % upper bound
    %
    % c01
    opt_para(3).label           = 'c01';
    opt_para(3).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(3).initial_value   = 2.000000E-01;                 % initial value [N/mm^2]
    opt_para(3).value           = opt_para(3).initial_value;
    opt_para(3).lb              = -Inf;                         % lower bound
    opt_para(3).ub              = Inf;                          % upper bound
    %
    % c20
    opt_para(4).label           = 'c20';
    opt_para(4).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(4).initial_value   = 1.000000E-01;                 % initial value [N/mm^2]
    opt_para(4).value           = opt_para(4).initial_value;
    opt_para(4).lb              = -Inf;                         % lower bound
    opt_para(4).ub              = Inf;                          % upper bound
    %
elseif strcmp(const_mod.constitutiveModel,'HW') % Haines-Wilson
    % set elastic parameters for optimization
    % K
    opt_para(1).label           = 'K';
    opt_para(1).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(1).initial_value   = 1.000000E+00;                 % initial value [N/mm^2]
    opt_para(1).value           = opt_para(1).initial_value;
    opt_para(1).lb              = -Inf;                         % lower bound
    opt_para(1).ub              = Inf;                          % upper bound
    %
    % c10
    opt_para(2).label           = 'c10';
    opt_para(2).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(2).initial_value   = 4.000000E-01;                 % initial value [N/mm^2]
    opt_para(2).value           = opt_para(2).initial_value;
    opt_para(2).lb              = -Inf;                         % lower bound
    opt_para(2).ub              = Inf;                          % upper bound
    %
    % c01
    opt_para(3).label           = 'c01';
    opt_para(3).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(3).initial_value   = 2.000000E-01;                 % initial value [N/mm^2]
    opt_para(3).value           = opt_para(3).initial_value;
    opt_para(3).lb              = -Inf;                         % lower bound
    opt_para(3).ub              = Inf;                          % upper bound
    %
    % c11
    opt_para(4).label           = 'c11';
    opt_para(4).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(4).initial_value   = 1.000000E-01;                 % initial value [N/mm^2]
    opt_para(4).value           = opt_para(4).initial_value;
    opt_para(4).lb              = -Inf;                         % lower bound
    opt_para(4).ub              = Inf;                          % upper bound
    %
    % c30
    opt_para(5).label           = 'c30';
    opt_para(5).opt             = true;                         % optimization flag: true (optimize parameter) or false (do not optimize parameter)
    opt_para(5).initial_value   = 1.000000E-01;                 % initial value [N/mm^2]
    opt_para(5).value           = opt_para(5).initial_value;
    opt_para(5).lb              = -Inf;                         % lower bound
    opt_para(5).ub              = Inf;                          % upper bound
    %
else
    error('%s: Wrong input for constitutive model',mfilename)
end

end