% script for calibration of linear elastic plate with a hole-example using 
% weighted non-linear least-squares method. Model response is evaluated 
% using provided finite element code. Can be easily adapted to, for example,
% parametric PINN or other surrogate models.

clc
clear 
close all

% settings
const_mod.weightRes = true; % use weighted (true) or unweighted (false) non-linear least-squares

% define output directory
path.out_dir = 'opt_output';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% read synthetic experimental data with or without artificial noise
% order is [displacements x_1; displacements x_2] in expData
generatedData = readmatrix('../../Experimental_Data/linElas_plate_syntheticalExperiment/displacements_withoutNoise.csv');
expData = [generatedData(:,3); generatedData(:,4)];

% add path to finite element code
addpath(genpath('../FEM/'));

% set inputfile for finite element code
sim_inpFile = 'plate_linElas';

% determine weighting factors
if const_mod.weightRes
    const_mod.weightDisp_x1 = 1/max(abs(generatedData(:,3)));
    const_mod.weightDisp_x2 = 1/max(abs(generatedData(:,4)));
end

% setup parameters for optimization (both parameters are optimized, Young's
% modulus and Poisson's ratio)
opt_para = set_optPara_linElas(true,true);

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
% parametric PINN
%%% adapt this line for model evaluation using a pre-trained parametric PINN

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