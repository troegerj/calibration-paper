function [stress,strain,ct,q,yieldval] = stct_HainesWilson(matpar,defGrad,dt,qn,problemtype2D)
% DESCRIPTION
%  Material routine for finite strain hyperelasticity using Haines-Wilson
%  ansatz for the isochoric part of the strain energy density:
%  w(IBq,IIBq) = c10(IBq-3) + c01(IIBq-3) + c11(IBq-3)(IIBq-3) + c30(IBq-3)^3
%  Computations are based on the isochoric-volumetric split of the 
%  deformation gradient. All quantities are computed with respect to the 
%  current configuration.
%
% INPUT
%  matpar -- material parameters
%  defGrad -- deformation gradient (NOT stored in vector notation)
%  dt -- time-step size (not used)
%  qn -- internal variable state at tn (not used)
%  problemtype2D -- switch for two-dimensional problemtype
%
% OUTPUT
%  stress -- Cauchy stress (stored in vector notation)
%  strain -- Almansi strain (stored in vector notation)
%  ct -- consistent tangent
%  q -- updated internal variable state at tn+1 (not used)
%  yieldval -- indicator for plastic yielding (not used)

% initialize variables
yieldval = 0;
q        = qn;

% extract material parameters
K = matpar(1); % [N/mm^2]
c10 = matpar(2); % [N/mm^2]
c01 = matpar(3); % [N/mm^2]
c20 = matpar(4); % [N/mm^2]

% compute determinant of deformation gradient
detF = det(defGrad);

% compute unimodular deformation gradient
unimodF = detF^(-1/3)*defGrad;

% compute unimodular left Cauchy-Green tensor
unimodB = unimodF*unimodF';
% rearrange into vector format employing symmetry
Bq = [unimodB(1,1); unimodB(2,2); unimodB(3,3); unimodB(1,2); unimodB(2,3); unimodB(3,1)];

% compute invariants of unimodular left Cauchy-Green tensor
IBq = trace(unimodB);
IIBq = 0.5*(IBq^2-trace(unimodB*unimodB));

% evaluate partial-derivatives of strain energy density w.r.t. first and
% second invariants. Derivatives w.r.t. invariants of left and right
% unimodular Cauchy-Green tensor coincide.
%  Haines-Wilson: c10(IBq-3) + c01(IIBq-3) + c11(IBq-3)(IIBq-3) + c30(IBq-3)^3
w1 = c10 + c11*(IIBq-3) + 3*c30*(IBq-3)^2; % dw/dIBq
w2 = c01 + c11*(IBq-3); % dw/dIIBq
w11 = 6*c30*(IBq-3); % d^2w/dIBq^2
w12 = c11; % d^2w/(dIBq dIIBq)
w22 = 0; % d^2w/dIIBq^2

% evaluate Cauchy stress and consistent tangent operator in relation to
% current configuration
[cauchy3D,ct3D] = hyperelasticity_utility(detF,Bq,K,w1,w2,w11,w12,w22);

% compute Almansi strains: A = 0.5*(1 - F^-T F^-1)
strain3D = 0.5*(eye(3,3) - inv(defGrad')*inv(defGrad));

% reduction to 2D
switch problemtype2D
    case 'plane_strain'
        stress = cauchy3D(1:4);
        strain = [strain3D(1,1) strain3D(2,2) strain3D(1,2)];
        ct3D([3,5,6],:)=[];
        ct3D(:,[3,5,6])=[];
        ct = ct3D;
    otherwise
        error('%s: two-dimensional problemtype not implemented for finite strain hyperelasticity',mfilename);
end

end
