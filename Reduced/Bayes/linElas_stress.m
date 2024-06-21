function [stress] = linelas_stress(matpar)
% function to perform model evaluation for 1D linear elasticity to compute
% stresses for identification of Young's modulus
%
% INPUT
%  matpar -- material parameter
%   matpar(1) -- E [N/mm^2]
%
% OUTPUT
%  stress -- stress response (column vector)

% check matpar
if size(matpar,1) ~= 1
    error('Invalid number of material parameters provided!');
end

% "true" parameter - here the approximated parameter for scaling
matparTrue = 200000;

% de-normalize parameters for "simulation" (+- 20 % range assumed)
matpar = 0.4*matparTrue(:).*matpar(:) + 0.8*matparTrue(:);

% check size of elastic modulus
if matpar(1) <= 0
    error('elastic modulus should not be zero!')
end

% prescribed strain values for evaluation
strains = [2.e-4 4.e-4 6.e-4 8.e-4 1.e-3]';

% linear elasticity in 1D
sigma_fh = @(epsilon)(matpar(1)*epsilon);

% evaluate stresses for given strains
stress = sigma_fh(strains);

end