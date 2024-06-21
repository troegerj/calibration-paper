function plasticStress = exe_sim_plastic(matpar)
% DESCRIPTION
%  function performing the finite element simulation to obtain the model
%  response for the purpose of numerical differentiation
%
% INPUT
%  matpar -- material parameters for elasto-plastic cube
%
% OUTPUT
%  plasticStress -- stress response in the plastic domain

% perform model evaluation using one-element finite element model
resFEM = mainFE('cube',matpar);
% extract plastic stresses
plasticStress = reshape(resFEM.stress(1,1,1,:),[],1);

end