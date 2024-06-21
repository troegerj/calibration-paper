function [latStrain] = linelas_latStrain(matpar)
% function to perform model evaluation for 1D linear elasticity to compute
% lateral strains for identification of Poisson's ratio
%
% INPUT
%  matpar -- material parameter
%   matpar(1) -- nu [-]
%
% OUTPUT
%  latStrain -- lateral strain response (column vector)

% check matpar
if size(matpar,1) ~= 1
    error('Invalid number of material parameters provided!');
end

% "true" parameter - here the approximated parameter for scaling
matparTrue = 0.275;

% de-normalize parameters for "simulation" (+- 20 % range assumed)
matpar = 0.4*matparTrue(:).*matpar(:) + 0.8*matparTrue(:);

% check Poisson ratio
if matpar(1) <= 0.0 && matpar(1) >= 0.5
    error('Poisson ratio should neither be zero or greater or equal to 0.5');
end

% prescribed strain values for evaluation
strains = [2.e-4 4.e-4 6.e-4 8.e-4 1.e-3]';

% linear elasticity in 1D
latStrain_fh = @(epsilon)(-matpar(1)*epsilon);

% evaluate lateral strains for given strains
latStrain = latStrain_fh(strains);

end