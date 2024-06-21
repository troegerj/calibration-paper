function J_pe = compJpe(elasticPara,plasticPara,m)
% DECRIPTION
%  function computing the matrix J_pe containing the partial derivatives
%  of the plastic simulation data w.r.t. elastic material parameters
% 
% INPUT
%  elasticPara -- elasticity parameters (bulk modulus K, shear modulus G)
%  plasticPara -- plasticity parameters (yield stress k, hardening param. c and b)
%  m -- number of considered data points in plastic domain
%
% OUTPUT
%  J_pe -- matrix containing partial derivatives of plastic model response
%          w.r.t. elasticity parameters

% initialize J_pe
J_pe = zeros(m,size(elasticPara,1));

% set parameter vector
x = [elasticPara; plasticPara];

% compute J_pe
%
% compute ds_p/dK [-]
J_pe(:,1) = comp1stDeriv(x,1); % var K
% compute ds_p/dG [-]
J_pe(:,2) = comp1stDeriv(x,2); % var G

end

%% subfunction

function Partial1stDeriv = comp1stDeriv(x,modPar1)
% function to compute the first-order partial derivative of the modifiable
% parameter index 1

% set epsdif for central differences and first derivatives
epsdif = 1.e-06; % approx. eps^(1/3)

% compute h
h = epsdif*abs(x);
% perform correction of h
logIdx_h = h < epsdif;
h(logIdx_h) = epsdif;
% create x+h and x-h
xph = x + h;
xpnh = x - h;

% create parameter array for computing derivatives
para = x;

% compute first term in nominator x+h
modPara = para;
modPara(modPar1) = xph(modPar1);
plasticStress1 = exe_sim_plastic(modPara);
% compute second term in nominator x-h
modPara = para; % restore changes
modPara(modPar1) = xpnh(modPar1);
plasticStress2 = exe_sim_plastic(modPara);

% compute partial first derivative
Partial1stDeriv = (plasticStress1-plasticStress2)/(2*h(modPar1));

end