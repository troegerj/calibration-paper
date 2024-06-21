% script to perform the uncertainty quantification for two-step NLS
% procedure as described in the publication:
%
%  U. Römer, S. Hartmann, J.-A. Tröger, D. Anton, H. Wessels, M. Flaschel
%  and L. De Lorenzis - Reduced and All-at-Once Approaches for Model 
%  Calibration and Discovery in Computational Solid Mechanics
%
%  (Equations are given in Appendix G of the publication)
%
% some information (residuals and unweighted Jacobian) from the
% NLS-calibration of the plasticity parameters has to be provided, which is
% here realized using .csv files in './input/'

clc
clear
close all

% add path to finite element code
addpath(genpath('../../FEM/'));

%% compute uncertainty of first-step

% compute Jacobian matrix J_e(k_e) for elastic parameter set k_e = (K,G)
%
% define K and G used for plasticity calibration using NLS
K = 1.509914161955137e+05;
G = 7.932095418001128e+04;
% prescribe covariance matrix for K and G
covarKG = zeros(2,2);
covarKG(1,1) = 8.708739281591019e+06;
covarKG(1,2) = -6.632597114877301e+04;
covarKG(2,1) = covarKG(1,2);
covarKG(2,2) = 3.947599915066658e+05;
% elasticity parameters
elasticPara = [K; G];

%% compute uncertainty of second step

% set size of plastic experimental data s_p
m = 50;

% define calibrated plastic material parameters 
plasticPara(1) = 2.826304648430587e+02; % k [N/mm^2]
plasticPara(2) = 3.499777124807194e+03; % c [N/mm^2]
plasticPara(3) = 41.040862524324467; % b [-]

% load plastic calibration data -- provided as .csv in './input/' (adapt
% path or overwrite files in new use cases of this routine)
%   plastic residual r_p -- residualPlastic
residualPlastic = readmatrix('input/plastic_residuals.csv');
%   Jacobian of plastic step J_p(k_p) -- (m x n_p) -- JacobianPlastic
%    1st column [-] ds_p/dk
%    2nd column [-] ds_p/dc
%    3rd column [N/mm^2] ds_p/db
JacobianPlastic = readmatrix('input/plastic_jacobian.csv');

% compute estimation of unknown (but common) variance sig_p^2 from residuals [N^2/mm^4]
sigSquarePlastic = cov(residualPlastic);

% compute J_pe -- (m x n_e)
% contains first partial derivatives of plastic residual w.r.t. elastic parameters (K,G)
%  1st column [-] ds_p/dK
%  2nd column [-] ds_p/dG
J_pe = compJpe(elasticPara,plasticPara',m);

% computation of F_pe = d/dk_e (J_p^T) -- (n_p x m x n_e)
% contains the mixed partial second-derivatives
F_pe = compFpe(elasticPara,plasticPara',m);

% build first term in Z matrix -- sig_p^2 * J_p^T J_p [n_p x n_p]
Z1 = sigSquarePlastic*JacobianPlastic'*JacobianPlastic;

% build second term in Z matrix -- J_p^T*J_pe*Var[k_e]*J_pe^T*J_p [n_p x n_p]
Z2 = JacobianPlastic'*J_pe*covarKG*J_pe'*JacobianPlastic;

% compute third term in Z matrix -- sig_p^2 * G_pe * Var[k_e]
%  here, Var[r_p] is assumed to be sig_p^2* I
Z3 = zeros(size(F_pe,1),size(F_pe,1));
G_pe = zeros(size(F_pe,1),size(F_pe,3),size(F_pe,1),size(F_pe,3));
for j = 1 : size(F_pe,1) % n_p
    for jp = 1 : size(F_pe,1) % n_p
        
        for k = 1 : size(F_pe,3) % n_e
            for kp = 1 : size(F_pe,3) % n_e
                
                for i = 1 : m
                    
                    % compute G contribution
                    G_pe(j,k,jp,kp) = F_pe(j,i,k)*F_pe(jp,i,kp); 
                    
                end
                
                % consideration of Var[k_e](k,kp)
                Z3(j,jp) = Z3(j,jp) + G_pe(j,k,jp,kp).*covarKG(k,kp); 
                
            end
        end
        
    end
end
% consider variance of plastic residuals
Z3 = sigSquarePlastic.*Z3;

% build Z-matrix
Z = (4/m)*(Z1 + Z2 + Z3);

% compute Q matrix
Q = (2/m)*JacobianPlastic'*JacobianPlastic;

% compute uncertainty estimation
uncertPlastic = inv(Q)*Z*inv(Q);

% compute uncertainties in plastic parameters -- take into account number
% of data
uncert_k = sqrt(uncertPlastic(1,1)/m);
uncert_c = sqrt(uncertPlastic(2,2)/m);
uncert_b = sqrt(uncertPlastic(3,3)/m);

% create path to binary MATLAB file
out = fullfile(pwd,strcat('out.mat'));
save(out);