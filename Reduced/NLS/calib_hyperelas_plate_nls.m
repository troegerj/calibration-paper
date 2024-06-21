% script for calibration of hyperelastic plate with a hole-example using 
% weighted non-linear least-squares method. Model response is evaluated 
% using provided finite element code. 
% The synthetic experimental data is taken from the EUCLID-repository: 
%  https://github.com/EUCLID-code/EUCLID-hyperelasticity (see links therein
%  as well).
% It is assumed that the EUCLID-data is stored in a subdirectory
% './EUCLID_Data/' -- adapt eval_exp_EUCLID.m if necessary

clc
clear 
close all

% settings
const_mod.weightRes = true;         % use weighted (true) or unweighted (false) non-linear least-squares
const_mod.constitutiveModel = 'NH'; % choose constitutive model: NH - Neo-Hooke, IH - Isihara, HW - Haines-Wilson
const_mod.noiseLevel = 0;           % switch for noise level: 0 - no noise, 1 - low noise, 2 - high noise

% define output directory
path.out_dir = 'opt_output';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% get finite element node positions for interpolation (residual between 
% model response and data is computed at finite element nodes)
nodeCoord = readmatrix('../FEM/inputfiles/modelData_hyperelas_plate-with-hole/hyperelas_plate_nodeCoordinates.csv');

% determine synthetical experimental data -- linear interpolation onto the
% finite element node positions is carried out
[expData,const_mod.weights] = eval_exp_EUCLID(const_mod,nodeCoord);

% add path to finite element code
addpath(genpath('../FEM/'));

% set inputfile for finite element code
sim_inpFile = strcat('plate_hyperelas_',const_mod.constitutiveModel);

% setup parameters for optimization
opt_para = set_optPara_hyperelas(const_mod);

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
res_initial = mainFE(sim_inpFile,[opt_para(:).initial_value]);

%% optimization with lsqnonlin

% optimization routine
[opt_para,t_elapsed,opt_resnorm,opt_residual,opt_exitflag,opt_output,opt_lambda,...
    opt_jacobian_normalized] = perform_optimization(expData,sim_inpFile,opt_para,opt_options,const_mod);

% save variables
out = fullfile(pwd,path.out_dir,'out.mat');
save(out)