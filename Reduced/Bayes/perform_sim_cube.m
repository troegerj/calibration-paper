function [stress] = perform_sim_cube(matpar)
% function to simulate cube under tensile load with provided finite element
% code
%
% INPUT
%  matpar -- material parameters
%   matpar(-) -- K [N/mm^2] % previously identified
%   matpar(-) -- G [N/mm^2] % previously identified
%   matpar(1) -- k [N/mm^2] 
%   matpar(2) -- b [-]
%   matpar(3) -- c [N/mm^2]
%
% OUTPUT
%  stress -- axial stress response (column vector, ntimesteps x 1)

% check material parameters
if size(matpar,1) ~= 3
    error('Invalid size of material parameter input!')
end

% parameters for scaling
matparTrue = [290 3000 35];

% de-normalize parameters for model evaluation
matpar(1) = 0.4*matparTrue(1).*matpar(1) + 0.8*matparTrue(1); % +- 20 % for yield stress
matpar(2) = 0.6*matparTrue(2).*matpar(2) + 0.7*matparTrue(2); % +- 30 % for hardening parameters
matpar(3) = 0.6*matparTrue(3).*matpar(3) + 0.7*matparTrue(3); % +- 30 % for hardening parameters

% extend with previously calibrated elasticity parameters
K = 1.523056580767234e+05; % [N/mm^2]
G = 7.939244857814073e+04; % [N/mm^2]
matpar = [K; G; matpar];

% perform simulation using finite elements
result = mainFE('cube',matpar);

% extract axial stress response for all time-steps
stressResponse = result.stress(1,1,1,:);
stress = reshape(stressResponse,[size(stressResponse,4),1]);

end