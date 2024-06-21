function F_pe = compFpe(elasticPara,plasticPara,m)
% DECRIPTION
%  function computing the tensor F containing the mixed partial second
%  derivatives of the simulation data w.r.t. elastic and plastic material
%  parameters
% 
% INPUT
%  elasticPara -- elasticity parameters (bulk modulus K, shear modulus G)
%  plasticPara -- plasticity parameters (yield stress k, hardening param. c and b)
%  m -- number of considered data points in plastic domain
%
% OUTPUT
%  F_pe -- tensor containing partial derivatives of plastic model response
%          w.r.t. elasticity and plasticity material parameters

% initialize F_pe -- F_pe is d/dk_e of J_p^T (make sure that transpose is
% considered) -- dimensions are (n_p x m x n_e)
F_pe = zeros(size(plasticPara,1),m,size(elasticPara,1));

% set parameter vector
x = [elasticPara; plasticPara];

% compute F_pe
%
% compute d^2 s_p/dKdk [mm^2/N]
F_pe(1,:,1) = compMixed2ndDeriv(x,1,1)'; % var K and var k
% compute d^2 s_p/dKdc [mm^2/N]
F_pe(2,:,1) = compMixed2ndDeriv(x,1,2)'; % var K and var c
% compute d^2 s_p/dKdb [-]
F_pe(3,:,1) = compMixed2ndDeriv(x,1,3)'; % var K and var b
%
% compute d^2 s_p/dGdk [mm^2/N]
F_pe(1,:,2) = compMixed2ndDeriv(x,2,1)'; % var G and var k
% compute d^2 s_p/dGdc [mm^2/N]
F_pe(2,:,2) = compMixed2ndDeriv(x,2,2)'; % var G and var c
% compute d^2 s_p/dGdb [-]
F_pe(3,:,2) = compMixed2ndDeriv(x,2,3)'; % var G and var b

end

%% subfunction

function mixed2ndDeriv = compMixed2ndDeriv(x,modPar1,modPar2)
% function to compute the second-order mixed partial derivative of the 
% modifiable parameter indices 1 and 2

% set epsdif for second derivatives
epsdif = 1.e-04; % approx. (4*eps)^(1/4)

% compute h
h = epsdif*abs(x);
% perform correction of h
logIdx_h = h < epsdif;
h(logIdx_h) = epsdif;
% create x+h and x-h
xph = x + h;
xpnh = x - h;

% create parameter array for input file modification
para = x;

% extend modPar2 value to consider position of plastic parameter
modPar2 = modPar2 + 2;

% compute first term in nominator +/+
modPara = para;
modPara(modPar1) = xph(modPar1);
modPara(modPar2) = xph(modPar2);
plasticStress1 = exe_sim_plastic(modPara);
% compute second term in nominator +/-
modPara = para; % restore changes
modPara(modPar1) = xph(modPar1);
modPara(modPar2) = xpnh(modPar2);
plasticStress2 = exe_sim_plastic(modPara);
% compute third term in nominator -/+
modPara = para; % restore changes
modPara(modPar1) = xpnh(modPar1);
modPara(modPar2) = xph(modPar2);
plasticStress3 = exe_sim_plastic(modPara);
% compute fourth term in nominator -/-
modPara = para; % restore changes
modPara(modPar1) = xpnh(modPar1);
modPara(modPar2) = xpnh(modPar2);
plasticStress4 = exe_sim_plastic(modPara);

% compute mixed partial second derivative
mixed2ndDeriv = (plasticStress1-plasticStress2-plasticStress3+plasticStress4)/(4*h(modPar1)*h(modPar2));

end