% script to perform the model calibration for linear elastic plate with a 
% hole assuming small strains and using synthetical experimental data

clc
clear
close all

% choose settings for MCMC
stepSize = 5; % should be larger than 1, see Goodman/Weare (2010)
chainLength = 100;
numberWalker = 50;

% define output directory
path.out_dir = 'opt_output';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% set default for random number generator
rng('default')

% number of parameters
numberParameter = 2;
% function handle for displacement computation
displacements = @(parameter) (perform_sim_plate(parameter));
% add path to finite element code
addpath('../FEM/');

% use normalized parameters in interval [0 1]
parameterTrue = [0.5,0.5];
parameterLower = [0.0,0.0];
parameterUpper = [1.0,1.0];

% read synthetical data with or without artificial noise
noise = 2e-4;
generatedData = readmatrix('../../Experimental_Data/linElas_plate_syntheticalExperiment/displacements_withNoise2e-04.csv');
expData = [generatedData(:,3); generatedData(:,4)];
numd = size(expData,1);

% log-likelihood and uniform prior
logLikelihood = @(parameter) -.5*norm(expData-displacements(parameter)).^2/noise^2 - numd/2*log(2*pi) - numd*log(noise);
logPrior = @(parameter) log(((parameter(1)>parameterLower(1))*(parameter(1)<parameterUpper(1))) * ...
    ((parameter(2)>parameterLower(2))*(parameter(2)<parameterUpper(2)))/1.0^2);

%% perform Markov-Chain Monte-Carlo evaluations

% initial ensemble
ensembleInit = 0.0 + 1.0*rand(numberParameter,numberWalker);

% perform MCMC
[chainMCMC, acceptanceRate] = MCMC(stepSize,chainLength,ensembleInit,logLikelihood,logPrior);

%% Post-processing

% Markov chain
figure;
plot(chainMCMC(1,:),chainMCMC(2,:),'k.')
print(strcat(path.out_dir,'/scatter_MarkovChain'),'-dpng','-r600')

% compute moments
meanBayesian(1) = mean(chainMCMC(1,:));
meanBayesian(2) = mean(chainMCMC(2,:));
stdBayesian(1) = std(chainMCMC(1,:));
stdBayesian(2) = std(chainMCMC(2,:));

% de-normalize parameters
matparTrue = [210000, 0.3];
% +- 10% range for elasticity parameters
matparIdentified = 0.2*matparTrue(:).*meanBayesian(:) + 0.9*matparTrue(:);
uncertIdentified = 0.2*matparTrue(:).*stdBayesian(:);

% entire distributions
figure
subplot(1,2,1)
histogram(chainMCMC(1,:),'Normalization','Probability')
subplot(1,2,2)
histogram(chainMCMC(2,:),'Normalization','Probability')
print(strcat(path.out_dir,'/distributions'),'-dpng','-r600')

% check stationarity
figure
subplot(1,2,1)
plot(chainMCMC(1,:))    
subplot(1,2,2)
plot(chainMCMC(2,:))    
print(strcat(path.out_dir,'/stationarity'),'-dpng','-r600')

% save variables
out = fullfile(pwd,path.out_dir,'out.mat');
save(out);