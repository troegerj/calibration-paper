% script to evaluate bulk and shear modulus from Young's modulus and
% Poisson ratio using Monte Carlo method

clc
clear
close all

% load samples for E and nu, e.g. load('chainE.mat') and load('chainNu.mat')
% or from files

% define output directory
path.out_dir = 'output_MC_KG';
% check for existence of opt_output directory
if not(isfolder(fullfile(pwd,path.out_dir)))
    mkdir(fullfile(pwd,path.out_dir))
end

% function handles to compute bulk and shear modulus
Kfh = @(E,nu) (E/(3*(1-2*nu)));
Gfh = @(E,nu) (E/(2*(1+nu)));

%% perform Monte Carlo method

% number of samples
nsample = 4000;

% set random number generator to
rng('default')

% sample E and nu directly from distribution
E_samples = randsample(chainE(1,:),nsample)';
nu_samples = randsample(chainNu(1,:),nsample)';

% de-normalize parameters (has to fit to calibration step with MCMC)
E_samples = 0.4*200000*E_samples + 0.8*200000;
nu_samples = 0.4*0.275*nu_samples + 0.8*0.275;

% evaluate bulk and shear modulus from samples
K_MCsamples = zeros(nsample,1);
G_MCsamples = zeros(nsample,1);
for i = 1 : nsample
    K_MCsamples(i,1) = Kfh(E_samples(i,1),nu_samples(i,1));
    G_MCsamples(i,1) = Gfh(E_samples(i,1),nu_samples(i,1));
end

% plot distributions
figure;
histogram(K_MCsamples(:,1),'Normalization','Probability');
title('K');
print(strcat(path.out_dir,'/distributionK'),'-dpng','-r600')
figure;
histogram(G_MCsamples(:,1),'Normalization','Probability');
title('G');
print(strcat(path.out_dir,'/distributionG'),'-dpng','-r600')

% determine moments
K_MC = mean(K_MCsamples);
G_MC = mean(G_MCsamples);
DeltaK_MC = std(K_MCsamples);
DeltaG_MC = std(G_MCsamples);
% determine error estimation
errK = DeltaK_MC/sqrt(nsample);
errG = DeltaG_MC/sqrt(nsample);

% save variables
out = fullfile(pwd,path.out_dir,'out.mat');
save(out);