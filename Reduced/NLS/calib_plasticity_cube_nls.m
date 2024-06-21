% script for calibration of small strain elasto-plastic constitutive model 
% using experimental data from uniaxial tensile tests. Calibration is 
% carried out with the weighted non-linear least-squares method by
% using an one-element cube with respective loading conditions. The
% elasticity parameters are calibrated by simply evaluating the well-known
% relations for axial stress and lateral strain, i.e. no finite element
% computation is carried out for elasticity.

clc
clear 
close all

% settings
%
% switch for calibrated material parameters
% 1 - elastic modulus E
% 2 - Poisson ratio nu
% 3 - plastic parameters (already known K and G, calibrate k, b, and c)
const_mod.matpar = 3;
% maximum axial strain for elastic domain
const_mod.maxStrainElas = 1.e-3; 
% number of data -- in calibration-paper: 5 (elasticity) or 50 (plasticity)
if const_mod.matpar == 1 || const_mod.matpar == 2
    const_mod.ndata = 5; 
elseif const_mod.matpar == 3
    const_mod.ndata = 50;
end
% use weighted (true) or unweighted (false) non-linear least-squares
const_mod.weightRes = true;

% define output directory
path.out_dir = 'opt_output';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% determine experimental data (stress-strain data and lateral strain information)
% by interpolation from uniaxial tensile testing data
[expData,const_mod.weights] = eval_exp_plasticity(const_mod);

% add path to finite element code
addpath(genpath('../FEM/'));

% set inputfile for finite element code
sim_inpFile = 'cube';

% setup parameters for optimization 
if const_mod.matpar == 1 % calibration of Young's modulus
    opt_para = set_optPara_linElas(true,false);
elseif const_mod.matpar == 2 % calibration of Poisson's ratio
    opt_para = set_optPara_linElas(false,true);
elseif const_mod.matpar == 3 % calibration of plasticity parameters
    opt_para = set_optPara_plasticity(const_mod);
else
    error('%s: wrong switch for material parameters given',mfilename);
end

% setup optimization settings by creating an options structure
opt_options = set_optimizer;

% determine number of parameters for optimization
nopt = sum([opt_para(:).opt]);
if (nopt <= 0)
    error('%s: number of optimized parameters has to be at least one',mfilename)
end

% select method for user defined finite differences
switch opt_options.FiniteDifferenceType
    case 'forward'
        const_mod.finite_diff_method = 1;
    case 'central'
        const_mod.finite_diff_method = 2;
    otherwise
        error('%s: wrong input for finite difference type',mfilename)
end

%% simulation with initial values

% compute nodal displacements with initial guess of parameters and finite
% element code
if const_mod.matpar == 3
    idx = [1; 2; 3; 6; 7]; % material parameters for plasticity
    res_initial = mainFE(sim_inpFile,[opt_para(idx).initial_value]);
end

%% optimization with lsqnonlin

% optimization routine
[opt_para,t_elapsed,opt_resnorm,opt_residual,opt_exitflag,opt_output,opt_lambda,...
    opt_jacobian_normalized] = perform_optimization(expData,sim_inpFile,opt_para,opt_options,const_mod);

% evaluate statistical quantities and perform simulation with optimized
% parameters
[statistics,res_optimized] = run_statistics(expData,sim_inpFile,opt_para,const_mod);

% save variables
out = fullfile(pwd,path.out_dir,'out.mat');
save(out)