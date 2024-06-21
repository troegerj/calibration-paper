function [stress,strain,ct,q,yieldval] = stct_linElasticity_2D(matpar,eps,dt,qn,problemtype2D)
% DESCRIPTION
%  material routine for small-strain linear elasticity in 2D
%
% INPUT
%  matpar -- material parameters
%  eps -- strains at current integration point (stored in vector notation)
%  dt -- time-step size (not used)
%  qn -- internal variable state at tn
%  problemtype2D -- switch for two-dimensional problemtype
%
% OUTPUT
%  stress -- stresses (stored in vector notation)
%  strain -- strains (stored in vector notation)
%  ct -- consistent tangent
%  q -- updated internal variable state at tn+1
%  yieldval -- indicator for plastic yielding

% initialize variables
yieldval = 0;
q        = qn;

% extract material parameters
E = matpar(1); % [N/mm^2]
nu = matpar(2); % [-]
% compute bulk and shear modulus
K = E/(3*(1-2*nu)); % [N/mm^2]
G = E/(2*(1+nu)); % [N/mm^2]

% initalize 3D quantities
eps3D = zeros(6,1);
stress3D = zeros(6,1);
ct3D = zeros(6,6);

% convert 2D strain tensor in 3D strain tensor
switch problemtype2D

    case 'plane_strain'
        idx = [1 2 4];
        eps3D(idx) = eps;

    case 'plane_stress'
        idx = [1 2 4];
        eps3D(idx) = eps;
        % compute strain in z-direction
        eps3D(3) = -nu/(1-nu)*(eps3D(1)+eps3D(2));

    otherwise
        error('%s: two-dimensional problemtype not implemented',mfilename);

end

% compute trace of strains
trE = eps3D(1)+eps3D(2)+eps3D(3);

% compute deviatoric part of strains
epsD = eps3D;
epsD(1:3) = eps3D(1:3)-trE/3;

% compute stresses
stress3D(1:3) = K*trE+2*G*epsD(1:3);
stress3D(4:6) = G*epsD(4:6);

% compute "consistent tangent"
ct3D(1:3,1:3) = K-2/3*G;
for i = 1 : 3
    ct3D(i,i) = ct3D(i,i) + 2*G;
    ct3D(i+3,i+3) = ct3D(i+3,i+3) + G;
end

% reduction to 2D
switch problemtype2D

    case 'plane_strain'

        stress = stress3D(1:4);
        strain = eps;
        ct3D([3,5,6],:)=[];
        ct3D(:,[3,5,6])=[];
        ct = ct3D;

    case 'plane_stress'

        stress = stress3D(idx);

        strain = [eps(1:2,1); eps3D(3); eps(3)];

        % condition for plane stress to account for sigma_zz=0
        ct_plane_stress=-1/ct3D(3,3)*([ct3D(3,1);ct3D(3,2);ct3D(3,4)]*[ct3D(3,1),ct3D(3,2),ct3D(3,4)]);
    
        ct3D([3,5,6],:)=[];
        ct3D(:,[3,5,6])=[];

        ct = ct3D + ct_plane_stress;

    otherwise
        error('%s: two-dimensional problemtype not implemented',mfilename);

end

end