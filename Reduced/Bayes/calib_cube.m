% script to perform the model calibration for elasto-plastic material
% assuming small strains and using real tensile test data

clc
clear
close all

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

% choose settings for MCMC
stepSize = 4; % should be larger than 1, see Goodman/Weare (2010)
chainLength = 100;
numberWalker = 80;

% define output directory
path.out_dir = 'opt_output';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% set default for random number generator
rng('default')

% determine experimental data (stress-strain data and lateral strain information)
% by interpolation from uniaxial tensile testing data. 
[expData,weights] = eval_exp_plasticity(const_mod);
numd = size(expData,1);
% build Sigma (covariance matrix of experimental observations)
Sigma = diag(weights);

if const_mod.matpar == 1
    numberParameter = 1;
    matparTrue = 200000; % for scaling, adapt in linelas_stress as well
    modelData = @(parameter) (linElas_stress(parameter));
elseif const_mod.matpar == 2
    numberParameter = 1;
    matparTrue = 0.275; % for scaling, adapt in linelas_latStrain as well
    modelData = @(parameter) (linElas_latStrain(parameter));
    % use constant noise to prevent from small determinant in Likelihood
    noise = 2.e-05; 
elseif const_mod.matpar == 3
    numberParameter = 3;
    matparTrue = [290 3000 35]; % for scaling, check also values in perform_sim-routine
    modelData = @(parameter) (perform_sim_cube(parameter));
    % add path to finite element code
    addpath(genpath('../FEM/'));
else
    error('%s: Invalid switch for set of material parameters',mfilename)
end

% likelihood for multiple experiments -- modelData should yield column vector
if const_mod.matpar ~= 2
    % use covariance matrix of observations
    logLikelihood = @(parameter) (-.5*(expData(:,1)-modelData(parameter))'*inv(Sigma)*(expData(:,1)-modelData(parameter)) - ...
        numd/2*log(2*pi) - .5*log(det(Sigma)));
elseif const_mod.matpar == 2
    % assumption of constant noise
    logLikelihood = @(parameter) (-.5*norm(expData(:,1)-modelData(parameter)).^2/noise^2 - numd/2*log(2*pi) - numd*log(noise));
end

% prior and parameter limits
if numberParameter == 1
    parameterLower = 0.0;
    parameterUpper = 1.0;
    logPrior = @(parameter) log((parameter(1)>parameterLower(1))*(parameter(1)<parameterUpper(1))/1.0^2);
elseif numberParameter == 2
    parameterLower = [0.0,0.0];
    parameterUpper = [1.0,1.0];
    logPrior = @(parameter) log(((parameter(1)>parameterLower(1))*(parameter(1)<parameterUpper(1))) * ...
        ((parameter(2)>parameterLower(2))*(parameter(2)<parameterUpper(2)))/1.0^2);
elseif numberParameter == 3
    parameterLower = [0.0,0.0,0.0];
    parameterUpper = [1.0,1.0,1.0];
    logPrior = @(parameter) log(((parameter(1)>parameterLower(1))*(parameter(1)<parameterUpper(1))) * ...
        ((parameter(2)>parameterLower(2))*(parameter(2)<parameterUpper(2))) * ...
        ((parameter(3)>parameterLower(3))*(parameter(3)<parameterUpper(3)))/1.0^2);
else
    error('Invalid number of parameters')
end

%% perform Markov-Chain Monte-Carlo evaluations

% initial ensemble
ensembleInit = 0.0 + 1.0*rand(numberParameter,numberWalker);

% perform MCMC
[chainMCMC, acceptanceRate] = MCMC(stepSize,chainLength,ensembleInit,logLikelihood,logPrior);

%% Post-processing

% Markov chain
if numberParameter == 2
    figure;
    plot(chainMCMC(1,:),chainMCMC(2,:),'k.')
    print(strcat(path.out_dir,'/scatter_MarkovChain'),'-dpng','-r600')
elseif numberParameter == 3
    figure;
    plot(chainMCMC(1,:),chainMCMC(2,:),'k.')
    print(strcat(path.out_dir,'/scatter_MarkovChain_par12'),'-dpng','-r600')
    figure;
    plot(chainMCMC(1,:),chainMCMC(3,:),'k.')
    print(strcat(path.out_dir,'/scatter_MarkovChain_par13'),'-dpng','-r600')
    figure;
    plot(chainMCMC(2,:),chainMCMC(3,:),'k.')
    print(strcat(path.out_dir,'/scatter_MarkovChain_par23'),'-dpng','-r600')
end

% compute moments
meanBayesian = zeros(numberParameter,1);
stdBayesian = zeros(numberParameter,1);
for i = 1 : numberParameter
    % compute parameters from means
    meanBayesian(i,1) = mean(chainMCMC(i,:));
    % compute standard deviation
    stdBayesian(i,1) = std(chainMCMC(i,:));
end

% de-normalize parameters
matparIdentified = zeros(numberParameter,1);
uncertIdentified = zeros(numberParameter,1);
if const_mod.matpar == 1 || const_mod.matpar == 2
    % +- 20% range for elasticity parameters
    matparIdentified = 0.4*matparTrue(:).*meanBayesian(:,1) + 0.8*matparTrue(:);
    uncertIdentified = 0.4*matparTrue(:).*stdBayesian(:,1);
elseif const_mod.matpar == 3
    % +- 20% range for yield stress
    matparIdentified(1,1) = 0.4*matparTrue(1).*meanBayesian(1,1) + 0.8*matparTrue(1);
    uncertIdentified(1,1) = 0.4*matparTrue(1).*stdBayesian(1,1);
    % +- 30% range for hardening parameters
    matparIdentified(2,1) = 0.6*matparTrue(2).*meanBayesian(2,1) + 0.7*matparTrue(2);
    uncertIdentified(2,1) = 0.6*matparTrue(2).*stdBayesian(2,1);
    matparIdentified(3,1) = 0.6*matparTrue(3).*meanBayesian(3,1) + 0.7*matparTrue(3);
    uncertIdentified(3,1) = 0.6*matparTrue(3).*stdBayesian(3,1);
end

% plot fit
if const_mod.matpar == 1 
    figure;
    hold on
    % experimental data
    plot(linspace(0,0.001,numd+1)',[0;expData(:,1)])
    % evaluate and plot model data
    modelResponse = modelData(meanBayesian(:,1));
    plot(linspace(0,0.001,numd+1)',[0;modelResponse],'-r','LineWidth',2);
    hold off
    print(strcat(path.out_dir,'/fit_stresses_elastic'),'-dpng','-r600')
elseif const_mod.matpar == 2
    figure;
    hold on
    % experimental data
    plot(linspace(0,0.001,numd+1)',[0;expData(:,1)])
    % evaluate and plot model data
    modelResponse = modelData(meanBayesian(:,1));
    plot(linspace(0,0.001,numd+1)',[0;modelResponse],'-r','LineWidth',2);
    hold off
    print(strcat(path.out_dir,'/fit_latStrain_elastic'),'-dpng','-r600')
elseif const_mod.matpar == 3
    figure;
    hold on
    % experimental data
    plot(linspace(0.001,0.05,numd)',expData)
    % evaluate and plot model data
    modelResponse = modelData(meanBayesian(:,1));
    plot(linspace(0.001,0.05,numd)',modelResponse,'-r','LineWidth',2);
    hold off
    print(strcat(path.out_dir,'/fit_stresses_plastic'),'-dpng','-r600') 
end

% entire distributions
if numberParameter == 1
    figure;
    histogram(chainMCMC(1,:),'Normalization','Probability')
elseif numberParameter == 2
    figure;
    subplot(1,2,1)
    histogram(chainMCMC(1,:),'Normalization','Probability')
    subplot(1,2,2)
    histogram(chainMCMC(2,:),'Normalization','Probability')
elseif numberParameter == 3
    figure;
    subplot(1,3,1)
    histogram(chainMCMC(1,:),'Normalization','Probability')
    subplot(1,3,2)
    histogram(chainMCMC(2,:),'Normalization','Probability')
    subplot(1,3,3)
    histogram(chainMCMC(3,:),'Normalization','Probability')
end
print(strcat(path.out_dir,'/distributions'),'-dpng','-r600')

% check stationarity
if numberParameter == 1
    figure;
    plot(chainMCMC(1,:))
elseif numberParameter == 2
    figure;
    subplot(1,2,1)
    plot(chainMCMC(1,:))
    subplot(1,2,2)
    plot(chainMCMC(2,:))
elseif numberParameter == 3
    figure;
    subplot(1,3,1)
    plot(chainMCMC(1,:))
    subplot(1,3,2)
    plot(chainMCMC(2,:))
    subplot(1,3,3)
    plot(chainMCMC(3,:))
end
print(strcat(path.out_dir,'/stationarity'),'-dpng','-r600')

% save variables
out = fullfile(pwd,path.out_dir,'out.mat');
save(out);