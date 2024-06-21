function opt_para = set_optPara_plasticity(const_mod)

% set plastic switch
if const_mod.matpar == 3
    const_mod.plastic = true;
else
    const_mod.plastic = false;
end

% set parameters for optimization 
% K
opt_para(1).label            = 'bulk modulus';
if ~const_mod.plastic
    opt_para(1).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)
elseif const_mod.plastic
    opt_para(1).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(1).initial_value    = 1.509914161955137e+05;     % initial value [N/mm^2]
opt_para(1).value            = opt_para(1).initial_value;
opt_para(1).lb               = -Inf;                      % lower bound
opt_para(1).ub               = Inf;                       % upper bound
%
% G
opt_para(2).label            = 'shear modulus';
if ~const_mod.plastic
    opt_para(2).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)
elseif const_mod.plastic
    opt_para(2).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(2).initial_value    = 7.932095418001128e+04;     % initial value [N/mm^2]
opt_para(2).value            = opt_para(2).initial_value;
opt_para(2).lb               = -Inf;                      % lower bound
opt_para(2).ub               = Inf;                       % upper bound
%
% initial yield stress
opt_para(3).label            = 'init. yield stress';
if ~const_mod.plastic
    opt_para(3).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
elseif const_mod.plastic
    opt_para(3).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(3).initial_value    = 2.900e+02;                 % initial value [N/mm^2]
opt_para(3).value            = opt_para(3).initial_value;
opt_para(3).lb               = -Inf;                      % lower bound
opt_para(3).ub               = Inf;                       % upper bound
%
% saturation yield stress (isotropic hardening)
opt_para(4).label            = 'sat. yield stress';
opt_para(4).opt              = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
opt_para(4).initial_value    = 0.000e+00;                 % initial value [N/mm^2]
opt_para(4).value            = opt_para(4).initial_value;
opt_para(4).lb               = -Inf;                      % lower bound
opt_para(4).ub               = Inf;                       % upper bound
%
% exponential parameter (isotropic hardening)
opt_para(5).label            = 'exp. para. (isotropic hard.)';
opt_para(5).opt              = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
opt_para(5).initial_value    = 0.00e+00;                  % initial value [N/mm^2]
opt_para(5).value            = opt_para(5).initial_value;
opt_para(5).lb               = -Inf;                      % lower bound
opt_para(5).ub               = Inf;                       % upper bound
%
% linear kinematic hardening parameter c
opt_para(6).label            = 'linear kin. hardening parameter';
if ~const_mod.plastic
    opt_para(6).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
elseif const_mod.plastic
    opt_para(6).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(6).initial_value    = 3.000e+03;                 % initial value [N/mm^2]
opt_para(6).value            = opt_para(6).initial_value;
opt_para(6).lb               = -Inf;                      % lower bound
opt_para(6).ub               = Inf;                       % upper bound
%
% nonlinear kinematic hardening parameter b
opt_para(7).label            = 'nonlinear kin. hardening parameter';
if ~const_mod.plastic
    opt_para(7).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
elseif const_mod.plastic
    opt_para(7).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(7).initial_value    = 3.50000e+01;               % initial value [-]
opt_para(7).value            = opt_para(7).initial_value;
opt_para(7).lb               = -Inf;                      % lower bound
opt_para(7).ub               = Inf;                       % upper bound
%
% viscosity
opt_para(8).label           = 'viscosity';
opt_para(8).opt             = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
opt_para(8).initial_value   = 0.0000e+00;                % initial value [-]
opt_para(8).value           = opt_para(8).initial_value;
opt_para(8).lb              = -Inf;                      % lower bound
opt_para(8).ub              = Inf;                       % upper bound
%
% viscosity exponent
opt_para(9).label           = 'viscosity exponent';
opt_para(9).opt             = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)
opt_para(9).initial_value   = 1.0000e+00;                % initial value [-]
opt_para(9).value           = opt_para(9).initial_value;
opt_para(9).lb              = -Inf;                      % lower bound
opt_para(9).ub              = Inf;                       % upper bound
%

end