function [expData,weights] = eval_exp_plasticity(const_mod)
% DESCRIPTION
%  function to extract the experimental data for calibration from uniaxial
%  tensile testing data (axial and lateral strain, axial stress). Data is
%  linearly interpolated to the evaluated time-points in model evaluation.
%
% INPUT
%  const_mod -- structure with information about optimization parameters
%
% OUTPUT
%  expData -- experimental data interpolated to model evalution points (in time)
%  weights -- weighting factors (here taken from variance of experimental observations)

% set number of experiments
nexp = 5;

% path to experimental data from uniaxial tensile tests
expPath = '../../Experimental_Data/plasticity_steel_tensileData/';

% build axial strain array
if const_mod.matpar == 1 || const_mod.matpar == 2 % elasticity
    axStrain = linspace(const_mod.maxStrainElas/const_mod.ndata,...
            const_mod.maxStrainElas,const_mod.ndata)';
elseif const_mod.matpar == 3 % plasticity, max. strain 
    maxStrain = 0.05; % choosen maximum strain for present study 
    axStrain = linspace(maxStrain/const_mod.ndata,maxStrain,const_mod.ndata)';
else
    error('%s: wrong input for material parameter switch',mfilename);
end

% read experimental data from files
for i = 1 : nexp

    % read axial strain [-]   lateral strain [-]   axial stress [MPa]
    exp_mod.data{i} = readmatrix(strcat(expPath,'TS275_000',num2str(i),'.csv'));

    % build interpolant depending on calibration setting 
    if const_mod.matpar == 1 
        % stress is quantity of interest
        %
        % build interpolant by considering only data in elastic domain
        F = griddedInterpolant(exp_mod.data{i}(exp_mod.data{i}(:,1) <= const_mod.maxStrainElas,1),...
            exp_mod.data{i}(exp_mod.data{i}(:,1) <= const_mod.maxStrainElas,3));
    elseif const_mod.matpar == 2
        % lateral strain is quantity of interest
        %
        % build interpolant by considering only data in elastic domain
        F = griddedInterpolant(exp_mod.data{i}(exp_mod.data{i}(:,1) <= const_mod.maxStrainElas,1),...
            exp_mod.data{i}(exp_mod.data{i}(:,1) <= const_mod.maxStrainElas,2));
    elseif const_mod.matpar == 3
        % stress is quantity of interest
        %
        % build interpolant by considering only data in plastic domain
        F = griddedInterpolant(exp_mod.data{i}(exp_mod.data{i}(:,1) > const_mod.maxStrainElas,1),...
            exp_mod.data{i}(exp_mod.data{i}(:,1) > const_mod.maxStrainElas,3));
    end

    % evaluate interpolant for given strain array
    expDataInterp(:,i) = F(axStrain);

end

% compute mean value for experimental data based on nexp repetitions
% -> calibration is carried out using the "mean data", variance of
% experimental observations is used for weighting
expData = mean(expDataInterp,2);

% determine diagonal components of weight matrix or use fixed variance in
% case of calibrating Poisson's ratio
if const_mod.matpar ~= 2
    weights = std(expDataInterp,0,2); % standard deviation
    weights = weights.*weights; % compute variance
elseif const_mod.matpar == 2
    % assume constant variance for lateral strains
    weights = (2.e-05)^2*ones(const_mod.ndata,1);
end

end