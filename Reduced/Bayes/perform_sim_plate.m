function [nodalDisp] = perform_sim_plate(matpar)
% function to simulate plate with hole (2D) with provided finite element 
% code or parametric PINN
%
% INPUT
%  matpar -- material parameters 
%   matpar(1) -- E [N/mm^2]
%   matpar(2) -- nu [-]
%
% OUTPUT
%  nodalDisp -- nodal displacements on plate's surface [u_x; u_y]

% use true parameters for scaling
matparTrue = [210000, 0.3];

% de-normalize parameters for simulation
matpar = 0.2*matparTrue(:).*matpar(:) + 0.9*matparTrue(:);

% check Poisson's ratio
if matpar(2) < 0.0 || matpar(2) >= 0.5
    error('Invalid value for Poisson ratio: %6.4d!',matpar(2))
end

% perform simulation using finite elements
result = mainFE('plate_linElas',matpar);
% perform simulation using parametric PINN
%%% adapt this line for model evaluation using a pre-trained parametric PINN 

% extract nodal displacements from finite element results (take care of the
% partitioned structure of the displacements)
nodalDisp = result.disp;

end