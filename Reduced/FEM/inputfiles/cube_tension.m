function [model,coord,inz,bcond,load,lfun] = cube_tension(matpar)
% DESCRIPTION
%  input file for a unit cube (edge length 1 mm) under tensile load
%
% INPUT
%  matpar -- material parameters
%
% OUTPUT
%  model -- structure containing information about the specified model
%  coord -- node coordinates
%  inz -- incidence matrix
%  bcond -- array with boundary condition information
%  load -- applied nodal loads
%  lfun -- load functions

%% model data

ndim   = 3; % three-dimensional problem
npe    = 8; % nodes per element: linear shape functions
ndf    = 3; % number of degrees of freedom per node
ptype  = ''; % problemtype 2D
elform = 'hex'; % element type: hexahedral

% node coordinates
coord(1,:) = [0 0 0];
coord(2,:) = [1 0 0];
coord(3,:) = [1 1 0];
coord(4,:) = [0 1 0];
coord(5,:) = [0 0 1];
coord(6,:) = [1 0 1];
coord(7,:) = [1 1 1];
coord(8,:) = [0 1 1];
nnode = size(coord,1);

% number of degrees of freedom
ndof = nnode*ndf;

% incidence matrix
inz(1,:) = [1 2 3 4 5 6 7 8];
nelem = size(inz,1);

%% boundary conditions

% boundary condition information stored in array
%
% rows: information of i-th boundary condition
% columns: node number,
%          load function (displacement/force x_1-direction),
%          load function (displacement/force x_2-direction),
%          load function (displacement/force x_3-direction),
%          load factor (x_1-direction),
%          load factor (x_2-direction),
%          load factor (x_3-direction)

% initialize
bcond = zeros(nnode,2*ndf+1);
% create index array
idx = (1:nnode)';

% edge length
L = 1; % [mm]

% set boundary conditions
bcond(1,:) = [1 1 1 1 0 0 0]; % all degrees of freedom are fixed
bcond(2,:) = [2 2 1 1 1 0 0]; % prescribed axial displacement and fixed displacements in other directions
bcond(3,:) = [3 2 0 1 1 0 0]; % prescribed axial displacement and fixed in x_3-direction
bcond(4,:) = [4 1 0 1 0 0 0]; % fixed in x_1- and x_3-direction
bcond(5,:) = [5 1 1 0 0 0 0]; % fixed in x_1- and x_2-direction
bcond(6,:) = [6 2 1 0 1 0 0]; % prescribed axial displacement and fixed in x_2-direction
bcond(7,:) = [7 2 0 0 1 0 0]; % prescribed axial displacement
bcond(8,:) = [8 1 0 0 0 0 0]; % fixed axial displacement

% determine number of nodes with Dirichlet boundary condition
nbc = size(bcond,1);

% initialize -- no nodal loads are applied
load = [];

% determine number of nodes with Neumann boundary condition
nloadbc = size(load,1);

% cell-array for load function
lfun    = cell(2,1);
lfun{1} = @(t)(0); % degrees of freedom are fixed
lfun{2} = @(t)(5.e-02*t); % prescribed axial displacement in x_1-direction

%% element sets

% number of element sets
neset = 1;
% starting element number for set 1
eset(1).start_el = 1;
% number of elements of set 1
eset(1).nelem = nelem;
% number of nodes per element in set 1
eset(1).npe = npe;
% incidence matrix of element set 1
eset(1).incidence = inz;
% element form in element set 1
eset(1).elform = elform;

%% constitutive model for material behavior

% viscoplasticity with isotropic and kinematic hardening (here, used
% without viscous terms and without isotropic hardening)
%  matpar(1) - bulk modulus [N/mm^2]
%  matpar(2) - shear modulus [N/mm^2]
%  matpar(3) - yield stress [N/mm^2]
%  matpar(4) - saturation yield stress (isotropic hardening) [N/mm^2]
%  matpar(5) - exponential parameter (isotropic hardening) [-]
%  matpar(6) - linear kinematic hardening parameter [N/mm^2]
%  matpar(7) - non-linear kinematic hardening parameter [-]
%  matpar(8) - viscosity [mm^2/s]
%  matpar(9) - exponent of viscosity function [-]
model.matpar(1) = matpar(1);
model.matpar(2) = matpar(2);
model.matpar(3) = matpar(3);
model.matpar(4) = 0.0; % no isotropic hardening is used
model.matpar(5) = 0.0; % no isotropic hardening is used
model.matpar(6) = matpar(4);
model.matpar(7) = matpar(5);
model.matpar(8) = 0.0; % no viscous effects are considered
model.matpar(9) = 1.0; % no viscous effects are considered

% set function handle to material routine
model.matfun = @stct_viscoplasticity;

% model has 13 internal variables (6 plastic strains, 6 backstresses, plastic arc length)
model.nq = 13;

% nonlinear geometry switch
model.nlgeom = false;

%% settings for solution of systems of linear equations

% time-integration settings (here, only step-wise application of loads)
t0 = 0; % [s]
tend = 1.0; % [s]
dt = 2.0e-02; % [s] - time-step size
maxsteps = 1000; % maximum number of time-steps

% termination criteria for Newton method 
tolg = 1.e-7; % tolerance in discretized equilibrium equations
toldu = 1.e-7; % tolerance for displacement update
maxni = 25; % maximum number of Newton iterations

%% settings for output

% output quantities
out.disp          = false;
out.stress        = true;
out.strain        = true;
out.yieldval      = true;
out.innerVar      = false;
out.reactionForce = false;

% output time
out.time = linspace(dt,tend,tend/dt)'; % after each time-step

% visualization of nodal quantities
out.visualizeDisp = false;

%% store data in model-structure

model.ndim    = ndim;
model.ptype   = ptype;
model.nnode   = nnode;
model.nbc     = nbc;
model.nloadbc = nloadbc;
model.ndof    = ndof;
model.ndf     = ndf;
model.neset   = neset;
model.eset    = eset;
%
model.t0       = t0;
model.tend     = tend;
model.dt       = dt;
model.maxsteps = maxsteps;
model.tolg     = tolg;
model.toldu    = toldu;
model.maxni    = maxni;
%
model.out = out;

end