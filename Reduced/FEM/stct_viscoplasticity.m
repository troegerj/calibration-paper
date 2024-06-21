function [stress,strain,ct,q,yieldval] = stct_viscoplasticity(matpar,eps,dt,qn)
% DESCRIPTION
%  material routine for generalized viscoplasticity (small-strains)
%  considering isotropic and kinematic hardening. Problem-adapted stress
%  algorithm (see IJNME, Hartmann/Luehrs/Haupt, 1997) and Pegasus-method to
%  solve local non-linear equation. Derivatives computed analytically.
%
% INPUT
%  matpar -- material parameters
%  eps -- strains at current integration point (stored in vector notation)
%  dt -- time-step size (not used)
%  qn -- internal variable state at tn
%
% OUTPUT
%  stress -- stresses (stored in vector notation, 6 x 1)
%  strain -- strains (stored in vector notation, 6 x 1)
%  ct -- consistent tangent
%  q -- updated internal variable state at tn+1
%  yieldval -- indicator for plastic yielding

% assign strains
strain = eps;

% compute trace of strain tensor
trE = eps(1)+eps(2)+eps(3);

% compute deviatoric part of strain tensor
epsD      = eps;
epsD(1:3) = epsD(1:3) - trE/3.d0;

% compute shear angles instead of shear strains
epsD(4:6) = epsD(4:6)*0.5d0;

% compute elastic predictor - deviatoric trial stresses
[yieldval,ttrialDev] = elpre(epsD,qn,matpar);

% evaluate material behavior, yieldval indicates plastic yielding
if yieldval

    % evaluate plastic yielding using Pegasus method
    [zeta] = flowruPegasus(matpar,ttrialDev,qn,dt);

    % compute stresses and update internal variables
    [stress,q] = remaining_state_variables(zeta,matpar,ttrialDev,eps,qn);

    % compute consistent tangent analytically
    ct = compute_ct_analytic(matpar,ttrialDev,zeta,qn,dt);

else

    % elastic material behavior, no change in internal variables
    q = qn;

    % compute stresses
    stress      = ttrialDev;
    % consider volumetric stresses
    stress(1:3) = stress(1:3) + matpar(1)*trE;

    % compute consistent tangent
    ct(1:3,1:3) = matpar(1)-2/3*matpar(2);
    for k = 1 : 3
        ct(k,k)     = ct(k,k) + 2.d0*matpar(2);
        ct(k+3,k+3) = matpar(2);
    end

end

end

%% subfunctions

function [yieldval,ttrial] = elpre(epsD,qn,matpar)

% extract material parameters
G     = matpar(2); % shear modulus
k     = matpar(3); % yield stress
kinf  = matpar(4); % saturation uniaxial yield stress (isotropic hardening)
alpha = matpar(5); % exponential parameter (isotropic hardening)

% save internal variables
epsp = qn(1:6);    % plastic strains at t = tn
xkin = qn(7:12);   % X - backstress tensor
s    = qn(13);     % s - plastic arc length

% compute trial stresses (deviatoric)
ttrial = 2*G*(epsD - epsp);

% calculate yield surface radius, in case of purely kinematic hardening
% giso = k
giso = kfun_isotropic_hardening(1,s,k,kinf,alpha);

% compute difference between deviatoric stresses and backstresses
diffStress = ttrial - xkin;

% compute von Mises yield condition
f = 0.5 * innerp(diffStress,diffStress) - (giso^2)/3;

% check if plastic yielding occurs and select yieldval
if (f > eps)
    yieldval = true;
else
    yieldval = false;
end

end


function giso = kfun_isotropic_hardening(task,s,k0,kinf,alpha)

if (task == 1)
    % determine isotropic hardening
    giso = kinf + (k0-kinf) * exp(-alpha*s);
elseif (task == 2)
    % first derivative of isotropic hardening function
    giso = -alpha * (k0-kinf) * exp(-alpha*s);
elseif (task == 3)
    % second derivative of isotropic hardening function
    giso = alpha^2 * (k0-kinf) * exp(-alpha*s);
else
    error('%s: wrong input for task',mfilename)
end

end


function res = innerp(a,b)
% Calculate inner product of two symmetric second order tensors, which are
% stored in the form 
% A = (a11;a22;a33;a12;a23;a31) and B = (b11;b22;b33;b12;b23;b31)

res =(a(1:3)'*b(1:3)) + 2*(a(4:6)'*b(4:6));

end


function [zeta] = flowruPegasus(matpar,ttrialDev,qn,dt)
% solve non-linear equation using Pegasus method

% termination criteria for Pegasus method
maxit = 25; % maximum number of iterations
tol = 1.0e-10; % tolerance

% evaluate function values for different values of zeta
zeta1       = 0;
[f1,status] = residual_problem_adap_stress_algorithm(zeta1,matpar,ttrialDev,qn,dt);
if (status ~= 0)
    error('%s: stress computation for Pegasus method failed',mfilename);
end
zeta2       = 0.4;
[f2,status] = residual_problem_adap_stress_algorithm(zeta2,matpar,ttrialDev,qn,dt);
if (status ~= 0)
    error('%s: stress computation for Pegasus method failed',mfilename);
end

% check start values, f(a)*f(b) < 0 has to hold
if (f1*f2 >= 0.d0) % same sign
    if (f1*f2 < 1e2*eps) % near zero
        zeta   = zeta1;
    else
        % requirement not fulfilled
        zeta   =  0;
    end
    return
elseif (f1*f2 == 0)
    zeta   =  0;
    return
end

% Pegasus method
for i = 1 : maxit

    % if the absolute value of f2 is lower than abserr, x2 is zero point
    if (abs(f2) <= tol)
        zeta = zeta2;
        break
    elseif (abs(zeta2-zeta1) <= 1e2*eps) % check termination criterion
        zeta = zeta2;

        if (abs(f1) < abs(f2))
            zeta=zeta1;
        end
        break
    else
        % compute secant slope
        s12 = (f2-f1)/(zeta2-zeta1);
        % compute intersection of secant
        zeta3 = zeta2-f2/s12;
        % compute new function value
        [f3,status] = residual_problem_adap_stress_algorithm(zeta3,matpar,ttrialDev,qn,dt);
        if (status ~= 0)
            error('%s: stress computation for Pegasus method failed',mfilename);
        end

        if (f2*f3 <= 0) % different sign
            zeta1 = zeta2;
            f1    = f2;
        else % same sign
            f1 = f1*f2/(f2+f3);
        end
        % overwrite zeta2
        zeta2 = zeta3;
        f2    = f3;
    end
end

if (i == maxit)
    error('%s: Pegasus method did not converge within maximum number of iterations',mfilename);
end

end


function [f,status] = residual_problem_adap_stress_algorithm(zeta,matpar,ttrialStressDev,qn,dt)
% evaluation of a scalar equation within the problem-adapted stress
% algorithm

% initialize status
status = 0;

% get material parameters
G     = matpar(2); % shear modulus
k     = matpar(3); % yield stress
kinf  = matpar(4); % saturation yield stress (isotropic hardening)
alpha = matpar(5); % exponential parameter (isotropic hardening)
ckin  = matpar(6); % linear kin. hardening parameter
bkin  = matpar(7); % non-linear kinematic hardening parameter
eta   = matpar(8); % viscosity
rflow = matpar(9); % exponent of viscosity function

% compute increment in plastic arc length and mue (see IJNME - Hartmann, 
% Luehrs, Haupt 1997)
ds  = zeta * sqrt(2/3);
mue = 1/(1 + bkin * ds);

% integrate plastic arc length
s = qn(13) + ds;

% compute yield stress
gis = kfun_isotropic_hardening(1,s,k,kinf,alpha);
gis = 2/3 * gis*gis;

% compute gamma
gamma = zeta*(2*G + ckin*mue);

% compute norm of xi 
[~,absXiTen] = compute_xi(zeta,matpar,ttrialStressDev,qn);

% compute radicand of phi-term
fac = 2*(eta*zeta/dt)^(1/rflow)+gis;
% check for negative radicand
if (fac < 0.0)
    if (abs(fac) <= eps)
        fac = 0.d0;
    else
        error('%s: negative radicand',mfilename);
    end
end

% evaluate scalar equation
f = absXiTen - gamma - sqrt(fac);

end


function [xid,absXiTen] = compute_xi(zeta,matpar,ttrialDev,qn)

% extract material parameter
bkin = matpar(7); % non-linear kinematic hardening parameter

% extract backstresses
Xkin = qn(7:12);

% compute mue
mue = 1 / (1 + bkin * zeta * sqrt(2/3));

% compute deviatoric part of xi (Eq. 45 IJNME, Hartmann, Luehrs, Haupt
% 1997)
xid = ttrialDev - mue * Xkin;

% compute norm of xi
absXiTen = sqrt(innerp(xid,xid));

end


function [stress,q] = remaining_state_variables(zeta,matpar,ttrialStressDev,eps,qn)

% extract material parameters
K    = matpar(1); % bulk modulus
G    = matpar(2); % shear modulus
ckin = matpar(6); % linear kinematic hardening parameter
bkin = matpar(7); % non-linear kinematic hardening parameter

% compute trace of strains
trE = eps(1)+eps(2)+eps(3);

% compute increment in plastic arc length and mue
ds  = zeta * sqrt(2/3);
mue = 1 / (1 + bkin * zeta * sqrt(2/3));

% compute xi^D
[xitd,absXiTen] = compute_xi(zeta,matpar,ttrialStressDev,qn);
% normalize xi
xn = xitd./absXiTen;

% initialize internal variables at t = tn+1
q           = zeros(length(qn),1);
% plastic strains
q(1:6)      = qn(1:6) + zeta * xn;
% backstresses
q(7:12)     = mue * (qn(7:12)  + zeta * ckin * xn);
% plastic arc length
q(13)       = qn(13) + ds;

% stresses
stress = ttrialStressDev - 2*G*zeta * xn;
% consider volumetric stresses
stress(1:3) = stress(1:3) + K*trE;

end


function ct = compute_ct_analytic(matpar,ttrialStressDev,zeta,qn,dt)

% extract material parameters
K     = matpar(1); % bulk modulus
G     = matpar(2); % shear modulus
bkin  = matpar(7); % non-linear kinematic hardening parameter

% deviator operator
dev = 1/3*[2,-1,-1,0,0,0; -1,2,-1,0,0,0; -1,-1,2,0,0,0; 0,0,0,3,0,0; ....
    0,0,0,0,3,0; 0,0,0,0,0,3];

% compute xi^D
[xid,absXiTen] = compute_xi(zeta,matpar,ttrialStressDev,qn);
% normalize xi
xn = xid/absXiTen;

% compute derivative analytically
dLdzeta = comp_dLdzeta(zeta,matpar,ttrialStressDev,qn,dt);
% revert sign
dLdzeta = -dLdzeta;

% compute mue
mue = 1/ (1 + bkin * sqrt(2/3) * zeta);

% weigth backstresses
xh = sqrt(2/3) * bkin * mue*mue * qn(7:12);

% define dzetaDE
dzetaDE      = (2*G)/dLdzeta*xn;
dzetaDE(4:6) = 2*dzetaDE(4:6);

% Compute derivative dN/dE
%
% compute NxN
NxN         = tenmat(xn,xn);
% dmudzeta dyadic Xn
dmudzeta_Xn = -xh;
% xn dyadic dzetaDE
NxdzetadE   = xn*dzetaDE';
%
dNdE=1/absXiTen*(2*G*dev - dmudzeta_Xn*dzetaDE' - 2*G*NxN + innerp(dmudzeta_Xn,xn)*NxdzetadE);

% compute dEp/dE
dEpdE = NxdzetadE+zeta*dNdE;

% compute tangent
%
% compute inelastic part
ct = -2*G*dEpdE;
%
% Multiply columns 4 to 6 of ct with 0.5 to account for derivative w.r.t.
% shear strains (gamm_{xy}, gamm_{yz}, gamm_{zx})
ct(:,4:6) = 0.5*ct(:,4:6);
%
% add elastic part to tangent
ct(1:3,1:3) = ct(1:3,1:3) + K - 2/3*G;
%
for j = 1 : 3
    ct(j,j)     = ct(j,j) + 2*G;
    ct(j+3,j+3) = ct(j+3,j+3) + G;
end

end


function dLdzeta = comp_dLdzeta(zeta,matpar,ttrialStressDev,qn,dt)
% comput the derivative of the evolution equations with respect to zeta

% extract material parameters
G     = matpar(2); % shear modulus
k     = matpar(3); % yield stress
kinf  = matpar(4); % saturation yield stress (isotropic hardening)
alpha = matpar(5); % exponential parameter (isotropic hardening)
ckin  = matpar(6); % linear kin. hardening parameter
bkin  = matpar(7); % non-linear kinematic hardening parameter
eta   = matpar(8); % viscosity
rflow = matpar(9); % exponent of viscosity function

% compute xi^D
[xitd,absXiTen] = compute_xi(zeta,matpar,ttrialStressDev,qn);
% normalize xi
xn = xitd/absXiTen;

% compute sqrt(2/3)*lambda*dt = sqrt(2/3)*zeta
ds = sqrt(2/3) * zeta;
% compute plastic arclength
s  = qn(13) + ds;

% compute quantities from isotropic hardening
gis1 = kfun_isotropic_hardening(1,s,k,kinf,alpha);
gis2 = kfun_isotropic_hardening(2,s,k,kinf,alpha);

% compute mue
mue = 1 / (1 + bkin * sqrt(2/3) * zeta);

% weight backstresses
xh = sqrt(2/3) * bkin * mue*mue * qn(7:12);
% compute scalar product
h1 = innerp(xh,xn);
% compute root-term
h2 = sqrt(2*(eta*zeta/dt)^(1/rflow)+(2/3)*gis1^2);

% compute derivative
dLdzeta = h1 - (eta/rflow/dt * (eta*zeta/dt)^(1/rflow-1) + (2/3) * sqrt(2/3) * gis1 * gis2) / h2 ...
    - (2*G + mue * ckin * (1 - zeta * sqrt(2/3) * bkin * mue));

end

function c = tenmat(a,b)
% function to compute the tensor product between two symmetric second order 
% tensors stored in Voigt-notation
% A = (a11;a22;a33;a12;a23;a31) and B = (b11;b22;b33;b12;b23;b31)

% initialize
c = zeros(6,6);

for i = 1 : 6
    for j = 1 : 6
        if (j <= 3)
            c(i,j) = a(i) * b(j);
        elseif (j > 3)
            c(i,j) = 2.d0 * a(i) * b(j);            
        end
    end 
end

end