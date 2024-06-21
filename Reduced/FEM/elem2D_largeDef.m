function [kelem,felem,stress,strain,yieldval,q] = elem2D_largeDef(el,ndim,XelmR,shp,wght,npe,dt,ue,matfun,matpar,qn,varargin) 
% DESCRIPTION
%  Element routine for two-dimensional quadrilateral elements considering
%  large deformations (finite strains). All computations are carried out 
%  with respect to the current configuration.
%  No mixed formulation is used!
%
% INPUT
%  el -- element number
%  ndim -- dimension of problem, should be 2
%  XelmR -- array with node coordinates in reference configuration (npe x ndim)
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
%  stress -- Cauchy stresses at integration points (stored in vector notation)
%  strain -- Almansi strains at integration points (stored in vector notation)
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
% initialize stress and strain
switch problemtype2D
    case 'plane_strain'
        stress = zeros(4,nint);
        strain = zeros(3,nint);
    otherwise
        error('%s: two-dimensional problemtype not implemented for finite strain',mfilename);
end

% compute node coordinates in current configuration
XelmC = XelmR + reshape(ue',[ndim,npe])';

% loop over all integration points
for ip = 1 : nint
    
    % compute determinant of Jacobian and shape function derivatives w.r.t.
    % global coordinates
    [detJR,shpdR] = inv_jac(ndim,shp(1:ndim,:,ip),XelmR); % ref. configuration
    [detJC,shpdC] = inv_jac(ndim,shp(1:ndim,:,ip),XelmC); % cur. configuration
    
    % check for non-positive determinant
    if (detJR <= 0 || detJC <= 0)
        error('%s: Jacobian of element %d integration point %d is not invertible - consider reducing load increment (via time-step size)',...
            mfilename,el,ip);
    end

    % compute deformation gradient
    switch problemtype2D
        case 'plane_strain'
            defGrad = zeros(3,3);
            defGrad(1:2,1:2) = XelmC'*shpdR';
            defGrad(3,3) = 1.0;
        otherwise
            error('%s: two-dimensional problemtype not implemented for finite strain',mfilename);
    end

    % compute weight factor
    wtR = detJR*wght(ip); % reference configuration
    wtC = detJC*wght(ip); % current configuration
    
    % evaluate stress algorithm (local level) - compute internal variables
    % and consistent tangent
    % output is already for 2D, i.e. for plane strain: stress 4x1 and ct 3x3
    [stress(:,ip),strain(:,ip),ct,q(:,ip),yieldval(:,ip)] = matfun(matpar,defGrad,dt,qn(:,ip),problemtype2D);
    
    switch problemtype2D
        case 'plane_strain'
            % setup B-matrix
            for j = 1 : npe
                i1 = (j-1)*ndf+1;
                i2 = j*ndf;
                B(1:3,i1:i2) = reshape([shpdC(1,j) 0.0 shpdC(2,j) 0.0 shpdC(2,j),...
                    shpdC(1,j)],[3 ndf]);
            end

            % reduction stresses
            stressRed = [stress(1,ip); stress(2,ip); stress(4,ip)];

            % setup B-matrix considering geometrical nonlinearity
            for j = 1 : npe
                i1 = (j-1)*ndf+1;
                i2 = j*ndf;
                BNL(:,i1:i2) = reshape([shpdC(1,j) 0.0 shpdC(2,j) 0.0 0.0 shpdC(2,j),...
                    0.0 shpdC(1,j)],[4 ndf]);
            end

            % setup Ms matrix
            Ms = reshape([stress(1,ip) 0.0 stress(4,ip) 0.0 ...
                0.0 stress(2,ip) 0.0 stress(4,ip) ...
                stress(4,ip) 0.0 stress(2,ip) 0.0 ...
                0.0 stress(4,ip) 0.0 stress(1,ip)],[4 4]);
    end


    % compute element force vector/right-hand side
    felem = felem - wtC*B'*stressRed;
    
    % compute element stiffness matrix
    ke    = wtC*B'*ct*B; % physical nonlinearity
    ke    = ke + wtC*BNL'*Ms*BNL; % geometrical nonlinearity
    kelem = kelem + ke;
    
end
    
end