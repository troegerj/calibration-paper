function [kelem,felem,stress,strain,yieldval,q] = elem3D(el,ndim,Xelm,shp,wght,npe,dt,ue,matfun,matpar,qn,varargin) 
% DESCRIPTION
%  element routine for three-dimensional hexahedral elements assuming small
%  strains
%
% INPUT
%  el -- element number
%  ndim -- dimension of problem, should be 3
%  Xelm -- array with node coordinates (npe x ndim)
%  shp -- shape functions and shape function derivatives
%  wght -- weighting factors of spatial integration method
%  npe -- nodes per element
%  dt -- time-step size
%  ue -- element displacements (npe*ndim x 1)
%  matfun -- function handle for material-routine
%  matpar -- material parameters
%  qn -- internal variables at time tn (nq x 1)
%  varargin -- two-dimensional problemtype
%
% OUTPUT
%  kelem -- element stiffness matrix
%  felem -- element force vector
%  stress -- stresses at integration points (stored in vector notation)
%  strain -- strains at integration points (stored in vector notation)
%  yieldval -- indicator whether plastic yielding occurs
%  q -- internal variables at integration points

% number of integration points
nint = size(shp,3);

% initialize quantities
ndf      = 3; % number of degrees of freedom per node
kelem    = zeros(ndf*npe,ndf*npe);
felem    = zeros(ndf*npe,1);
yieldval = false(1,nint);
stress   = zeros(6,nint);
strain   = zeros(6,nint);
B        = zeros(6,ndf*npe);

% loop over all integration points
for ip = 1 : nint

    % compute determinant of Jacobian and shape function derivatives w.r.t.
    % global coordinates
    [detJ,shpd] = inv_jac(ndim,shp(1:ndim,:,ip),Xelm);
        
    % check for non-positive determinant
    if (detJ <= 0)
        error('%s: Jacobian of element %d integration point %d is not invertible',mfilename,el,ip);
    end

    % compute weight factor
    wt = detJ*wght(ip);

    % compute B-matrix (strain-displacement matrix)
    for j = 1 : npe
        idx1 = (j-1)*ndf+1;
        idx2 = j*ndf;
        B(:,idx1:idx2) = [shpd(1,j),0,0; 0,shpd(2,j),0; 0,0,shpd(3,j);...
                    shpd(2,j),shpd(1,j),0; 0,shpd(3,j),shpd(2,j); shpd(3,j),0,shpd(1,j)];
    end
 
    % compute strains at t = tn+1
    eps = B*ue;
    
    % evaluate stress algorithm (local level) - compute internal variables
    % and consistent tangent
    [stress(:,ip),strain(:,ip),ct,q(:,ip),yieldval(:,ip)] = matfun(matpar,eps,dt,qn(:,ip));

    % compute element force vector/right-hand side
    felem = felem - B'*stress(:,ip)*wt;
    
    % compute element stiffness matrix
    ke    = wt*B'*ct*B;
    kelem = kelem + ke;

end

end