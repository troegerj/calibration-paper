% script to compute bulk and shear modulus for linear elasticity from
% elastic modulus and poissons ratio based on Gaussian error propagation

clc
clear
close all

% provide identified material parameters and uncertainties
E = 2.024650462119315e+05; % [N/mm^2]
nu = 0.276435105374081; % [-]
deltaE = 1.468123998007590e+03; % [N/mm^2]
deltanu = 0.004111444870443; % [-]

% compute K and G
K = E/(3*(1-2*nu)); % [N/mm^2]
G = E/(2*(1+nu)); % [N/mm^2]

% compute uncertainties with Gaussian error propagation (first-order second
% moment method)
%
% partial derivatives
dKdE = 1/(3*(1-2*nu)); % [-]
dKdnu = 2*E/(3*(1-2*nu)^2); % [N/mm^2]
dGdE = 1/(2*(1+nu)); % [-]
dGdnu = -E/(2*(1+nu)^2); % [N/mm^2]
deltaK = sqrt((dKdE*deltaE)^2+(dKdnu*deltanu)^2); % [N/mm^2]
deltaG = sqrt((dGdE*deltaE)^2+(dGdnu*deltanu)^2); % [N/mm^2]

% compute uncertainty using matrix notation
sensitivityMatrix = [dKdE dKdnu; dGdE dGdnu];
covarEnu = [deltaE^2 0; 0 deltanu^2];
%
covarKG = sensitivityMatrix*covarEnu*sensitivityMatrix';
% perform check
check_dK = sqrt(covarKG(1,1));
check_dG = sqrt(covarKG(2,2));
covarianceKG = sqrt(covarKG(1,2));