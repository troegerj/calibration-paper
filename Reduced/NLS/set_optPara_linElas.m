function opt_para = set_optPara_linElas(opt_E,opt_nu)
% function to set the elasticity parameters for calibration. Initial
% guesses and possible lower and upper bounds are defined.
%
% INPUT
%  opt_E -- switch whether Young's modulus is calibrated (true) or not (false)
%  opt_nu -- switch whether Poisson's ratio is calibrated (true) or not (false)
%
% OUTPUT
%  opt_para -- parameter structure for calibration

% E
opt_para(1).label            = 'Youngs modulus';
if opt_E
    opt_para(1).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)
else
    opt_para(1).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(1).initial_value    = 1.0000e+05;                % initial value [N/mm^2]
opt_para(1).value            = opt_para(1).initial_value;
opt_para(1).lb               = -Inf;                      % lower bound
opt_para(1).ub               = Inf;                       % upper bound
%
% nu
opt_para(2).label            = 'Poissons ratio';
if opt_nu
    opt_para(2).opt          = true;                      % optimization flag: true (optimize parameter) or false (do not optimize parameter)
else
    opt_para(2).opt          = false;                     % optimization flag: true (optimize parameter) or false (do not optimize parameter)    
end
opt_para(2).initial_value    = 1.00e-01;                  % initial value [-]
opt_para(2).value            = opt_para(2).initial_value;
opt_para(2).lb               = -Inf;                      % lower bound
opt_para(2).ub               = Inf;                       % upper bound
%

end