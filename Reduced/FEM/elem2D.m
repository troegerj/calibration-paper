function [kelem,felem,stress,strain,yieldval,q] = elem2D(el,ndim,Xelm,shp,wght,npe,dt,ue,matfun,matpar,qn,varargin) 
% DESCRIPTION
%  element routine for two-dimensional quadrilateral elements assuming
%  small strains
%
% INPUT
%  el -- element number
%  ndim -- dimension of problem, should be 2
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

% get two-dimensional problemtype
problemtype2D = varargin{1};

% number of integration points
nint = size(shp,3);

% initialize quantities
ndf      = 2; % number of degrees of freedom per node
kelem    = zeros(ndf*npe,ndf*npe);
felem    = zeros(ndf*npe,1);
yieldval = false(1,nint);
% initialize stress, strain, and strain-displacement matrix
switch problemtype2D
    case 'plane_strain'
        B = zeros(3,ndf*npe);
        stress = zeros(4,nint);
        strain = zeros(3,nint);
    case 'plane_stress'
        B = zeros(3,ndf*npe);
        stress = zeros(3,nint);
        strain = zeros(4,nint);
    otherwise
        error('%s: two-dimensional problemtype not implemented',mfilename);
end

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
        switch problemtype2D
            case 'plane_strain'
                idx1 = (j-1)*ndf+1;
                idx2 = j*ndf;
                B(:,idx1:idx2) = [shpd(1,j) 0; 0 shpd(2,j); shpd(2,j) shpd(1,j)];
            case 'plane_stress' % strains in z-direction will be computed in material routine
                idx1 = (j-1)*ndf+1;
                idx2 = j*ndf;
                B(:,idx1:idx2) = [shpd(1,j) 0; 0 shpd(2,j); shpd(2,j) shpd(1,j)];
            otherwise
                error('%s: two-dimensional problemtype not implemented',mfilename);
        end
    end
    
    % compute strains at t = tn+1
    eps = B*ue;
    
    % evaluate stress algorithm (local level) - compute internal variables
    % and consistent tangent
    % output is already for 2D, i.e. 
    % for plane strain: stress 4x1 and ct 3x3
    % for plane stress: stress 3x1 and ct 3x3
    [stress(:,ip),strain(:,ip),ct,q(:,ip),yieldval(:,ip)] = matfun(matpar,eps,dt,qn(:,ip),problemtype2D);
    
    % compute element force vector/right-hand side
    switch problemtype2D
        case 'plane_strain'
            idx = [1,2,4];
        case 'plane_stress'
            idx = [1,2,3];
        otherwise
            error('%s: two-dimensional problemtype not implemented',mfilename);
    end
    felem = felem - B'*stress(idx,ip)*wt;
    
    % compute element stiffness matrix
    ke    = wt*B'*ct*B;
    kelem = kelem + ke;
    
end
    
end