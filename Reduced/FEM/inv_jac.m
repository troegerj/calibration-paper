function [detJ,shpd] = inv_jac(ndim,shpdn,elCoord)
% DESCRIPTION
%  function to compute the determinant of the Jacobian of coordinate
%  transformation and the derivatives of shape functions with respect to
%  global coordinates x,y,z
%
% INPUT
%  ndim -- dimension
%  shpdn -- evaluated shape functions and shape function derivatives 
%  elCoord -- element node coordinates
%
% OUTPUT
%  detJ -- determinant of Jacobian of coordinate transformation
%  shpd -- derivatives of shape functions w.r.t. global coordinates

switch ndim

    case 2

        % compute transpose of Jacobian
        Jac = shpdn*elCoord;

        % compute determinant
        detJ = Jac(1,1)*Jac(2,2)-Jac(1,2)*Jac(2,1);

        % compute inverse of transpose Jacobian
        fac = 1/detJ;
        invjac(1,1)= Jac(2,2)*fac;
        invjac(1,2)=-Jac(1,2)*fac;
        invjac(2,1)=-Jac(2,1)*fac;
        invjac(2,2)= Jac(1,1)*fac;

        % compute global derivatives of shape functions
        shpd = invjac*shpdn;

    case 3

        % compute transpose of Jacobian
        Jac = shpdn*elCoord;

        % compute determinant
        detJ = det(Jac);

        % compute inverse of transpose Jacobian
        fac = 1/detJ;
        invjac(1,1)=( Jac(2,2)*Jac(3,3)-Jac(3,2)*Jac(2,3) ) * fac;
        invjac(1,2)=( Jac(3,2)*Jac(1,3)-Jac(1,2)*Jac(3,3) ) * fac;
        invjac(1,3)=( Jac(1,2)*Jac(2,3)-Jac(2,2)*Jac(1,3) ) * fac;
        invjac(2,1)=( Jac(3,1)*Jac(2,3)-Jac(2,1)*Jac(3,3) ) * fac;
        invjac(2,2)=( Jac(1,1)*Jac(3,3)-Jac(3,1)*Jac(1,3) ) * fac;
        invjac(2,3)=( Jac(2,1)*Jac(1,3)-Jac(1,1)*Jac(2,3) ) * fac;
        invjac(3,1)=( Jac(2,1)*Jac(3,2)-Jac(3,1)*Jac(2,2) ) * fac;
        invjac(3,2)=( Jac(3,1)*Jac(1,2)-Jac(1,1)*Jac(3,2) ) * fac;
        invjac(3,3)=( Jac(1,1)*Jac(2,2)-Jac(2,1)*Jac(1,2) ) * fac;

        % compute global derivatives of shape functions
        shpd = invjac*shpdn;

    otherwise
        error('%s: Invalid dimension given for computation of inverse Jacobian',mfilename)

end

end