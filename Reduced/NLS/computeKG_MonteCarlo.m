% script to evaluate bulk and shear modulus from Young's modulus and
% Poisson ratio using Monte Carlo method
%
% here, we use normally distributed E and nu for the NLS results according
% to asymptotic normality

clc
clear
close all

% provide identified parameters and uncertainties
E = 2.024650462119315e+05; % [N/mm^2]
nu = 0.276435105374081; % [-]
DeltaE = 1.468123998007590e+03; % [N/mm^2]
Deltanu = 0.004111444870443; % [-]

% define output directory
path.out_dir = 'output_MC_KG_4000';
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

% set default
rng('default')

% assume normally distributed E and nu with mean of fit and uncertainty as
% standard deviation
E_samples = normrnd(E,DeltaE,[nsample 1]);
nu_samples = normrnd(nu,Deltanu,[nsample 1]);

% evaluate bulk and shear modulus from samples
K_MCsamples = zeros(nsample,1);
G_MCsamples = zeros(nsample,1);
for i = 1 : nsample
    K_MCsamples(i,1) = Kfh(E_samples(i,1),nu_samples(i,1));
    G_MCsamples(i,1) = Gfh(E_samples(i,1),nu_samples(i,1));
end

% visualization
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