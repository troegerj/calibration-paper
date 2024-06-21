function shp = shape(elform,npe,xintp)
% DESCRIPTION
%  function for evaluation of the ansatz functions (shape functions) within
%  the finite elements
%
% INPUT
%  elform -- switch for element type
%  npe -- nodes per element
%  xintp -- integration point coordinates within unit domain [-1 1]
%
% OUTPUT
%  shp -- shape functions and shape function derivatives evaluated at
%         integration points

% set function handle for element form
switch elform

    case 'quad' % quadrilateral element
        shp_fh = @shp_quad;

    case 'hex'
        shp_fh = @shp_hex;

    otherwise
        error('%s: element form not implemented',mfilename)
end

% compute shape function and its derivatives, array with value of shape
% function (N_i) and derivatives with respect to xi (N_i,xi) and eta (N_i,eta)
shp = shp_fh(npe,xintp);

end

%% subfunctions

function shp = shp_quad(npe,xintp)
% shape functions for four node quadrilateral element (bilinear Q4-element)

% number of integration points
nint = length(xintp);

switch npe
    case 4 % shape functions for four node quadrilateral element (bilinear Q4-element)

        % coordinates of nodes in unit square
        xia  = [-1 1 1 -1]';
        etaa = [-1 -1 1 1]';

        % determine value of shape functions and their derivatives at the integration points
        shp = zeros(3,npe,nint);
        for ii = 1:nint
            % derivatives with respect to xi (N_i,xi)
            shp(1,:,ii) = 0.25*xia .* (1 + etaa*xintp(ii,2));

            % derivatives with respect to eta (N_i,eta)
            shp(2,:,ii) = 0.25*etaa .* (1 + xia*xintp(ii,1));

            % value of shape function (N_i)
            shp(3,:,ii) = 0.25*(1 + xia*xintp(ii,1)) .* (1 + etaa*xintp(ii,2));
        end

    case 8 % shape functions for eight node quadrilateral element

        % determine value of shape functions and their derivatives at the integration points
        shp = zeros(3,npe,nint);
        for ii = 1:nint

            % derivatives with respect to xi (N_i,xi)
            shp(1,1,ii) = 1/4 * (1-xintp(ii,2))*(2*xintp(ii,1)+xintp(ii,2));
            shp(1,5,ii) = -(1-xintp(ii,2))*xintp(ii,1);
            shp(1,2,ii) = 1/4 * (1-xintp(ii,2))*(2*xintp(ii,1)-xintp(ii,2));
            shp(1,6,ii) = 2 * 1/4 * (1-xintp(ii,2)*xintp(ii,2));
            shp(1,3,ii) = 1/4 * (1+xintp(ii,2))*(2*xintp(ii,1)+xintp(ii,2));
            shp(1,7,ii) = -(1+xintp(ii,2))*xintp(ii,1);
            shp(1,4,ii) = 1/4 * (1+xintp(ii,2))*(2*xintp(ii,1)-xintp(ii,2));
            shp(1,8,ii) = -2 * 1/4 * (1-xintp(ii,2)*xintp(ii,2));

            % derivatives with respect to eta (N_i,eta)
            shp(2,1,ii) = 1/4 * (1-xintp(ii,1))*(xintp(ii,1)+2*xintp(ii,2));
            shp(2,5,ii) = -2 * 1/4 * (1-xintp(ii,1)*xintp(ii,1));
            shp(2,2,ii) = -1/4*(1+xintp(ii,1))*(xintp(ii,1)-2*xintp(ii,2));
            shp(2,6,ii) = -(1+xintp(ii,1))*xintp(ii,2);
            shp(2,3,ii) = 1/4 * (1+xintp(ii,1))*(xintp(ii,1)+2*xintp(ii,2));
            shp(2,7,ii) = 2 * 1/4 * (1-xintp(ii,1)*xintp(ii,1));
            shp(2,4,ii) = -1/4 * (1-xintp(ii,1))*(xintp(ii,1)-2*xintp(ii,2));
            shp(2,8,ii) = -(1-xintp(ii,1))*xintp(ii,2);

            % value of shape function (N_i)
            shp(3,1,ii) = -1/4 * (1-xintp(ii,1))*(1-xintp(ii,2))*(1+xintp(ii,1)+xintp(ii,2));
            shp(3,5,ii) = 2 * 1/4 * (1-xintp(ii,1)*xintp(ii,1))*(1-xintp(ii,2));
            shp(3,2,ii) = -1/4 * (1+xintp(ii,1))*(1-xintp(ii,2))*(1-xintp(ii,1)+xintp(ii,2));
            shp(3,6,ii) = 2 * 1/4 * (1+xintp(ii,1))*(1-xintp(ii,2)*xintp(ii,2));
            shp(3,3,ii) = -1/4 * (1+xintp(ii,1))*(1+xintp(ii,2))*(1-xintp(ii,1)-xintp(ii,2));
            shp(3,7,ii) = 2 * 1/4 * (1-xintp(ii,1)*xintp(ii,1))*(1+xintp(ii,2));
            shp(3,4,ii) = -1/4 * (1-xintp(ii,1))*(1+xintp(ii,2))*(1+xintp(ii,1)-xintp(ii,2));
            shp(3,8,ii) = 2 * 1/4 * (1-xintp(ii,1))*(1-xintp(ii,2)*xintp(ii,2));
        end

    otherwise
        error('%s: number of nodes per element does not match any implemented quadrilateral element',mfilename)

end

end


function shp = shp_hex(npe,xintp)
% shape functions for hexahedral elements

% number of integration points
nint = length(xintp);

switch npe
    case 8 % shape functions for eight node hexahedral element

        % coordinates of nodes in unit cube
        xia   = [-1, 1, 1,-1,-1, 1, 1,-1]';
        etaa  = [-1,-1, 1, 1,-1,-1, 1, 1]';
        zetaa = [-1,-1,-1,-1, 1, 1, 1, 1]';

        % determine value of shape functions and their derivatives at the integration points
        shp = zeros(4,npe,nint);
        for ii = 1 : nint
            % derivatives with respect to xi (N_i,xi)
            shp(1,:,ii) = 1/8*xia .*(1 + etaa*xintp(ii,2)).*(1 + zetaa*xintp(ii,3));

            % derivatives with respect to eta (N_i,eta)
            shp(2,:,ii) = 1/8*etaa .*(1 + xia*xintp(ii,1)).*(1 + zetaa*xintp(ii,3));

            % derivatives with respect to zeta (N_i,zeta)
            shp(3,:,ii) = 1/8*zetaa .* (1 + xia*xintp(ii,1)).*(1 + etaa*xintp(ii,2));

            % value of shape function (N_i)
            shp(4,:,ii) = 1/8*(1 + xia*xintp(ii,1)) .* (1 + etaa*xintp(ii,2)) .*(1 + zetaa*xintp(ii,3));
        end

    case 20 % shape functions for 20 node hexahedral element

        % initialize
        xia  =zeros(1,npe);
        etaa =zeros(1,npe);
        zetaa=zeros(1,npe);

        % coordinates of corner nodes of the unit cube in xi, eta and zeta direction
        xia(1:8)   = [-1, 1, 1,-1,-1, 1, 1,-1]';
        etaa(1:8)  = [-1,-1, 1, 1,-1,-1, 1, 1]';
        zetaa(1:8) = [-1,-1,-1,-1, 1, 1, 1, 1]';

        % determine value of shape functions and their derivatives at the integration points
        shp = zeros(4,npe,nint);

        for gp = 1 : nint
            % Ni,x for corner nodes
            shp(1,1:8,gp) = 1/8 * xia(1:8).*(1+xintp(gp,2)*etaa(1:8)).*(1+xintp(gp,3)*zetaa(1:8)).*(-1+2*xintp(gp,1)*xia(1:8)+xintp(gp,2)*etaa(1:8)+xintp(gp,3)*zetaa(1:8));
            % Ni,y for corner nodes
            shp(2,1:8,gp) = 1/8 * etaa(1:8).*(1+xintp(gp,1)*xia(1:8)).*(1+xintp(gp,3)*zetaa(1:8)).*(-1+xintp(gp,1)*xia(1:8)+2*xintp(gp,2)*etaa(1:8)+xintp(gp,3)*zetaa(1:8));
            % Ni,z for corner nodes
            shp(3,1:8,gp) = 1/8 * zetaa(1:8).*(1+xintp(gp,1)*xia(1:8)).*(1+xintp(gp,2)*etaa(1:8)).*(-1+xintp(gp,1)*xia(1:8)+xintp(gp,2)*etaa(1:8)+2*xintp(gp,3)*zetaa(1:8));
            % Ni for corner nodes
            shp(4,1:8,gp) = 1/8 * (1+xintp(gp,1)*xia(1:8)) .* (1+xintp(gp,2)*etaa(1:8)) .* (1+xintp(gp,3)*zetaa(1:8)) .*(-2 + xintp(gp,1)*xia(1:8) + xintp(gp,2)*etaa(1:8) + xintp(gp,3)*zetaa(1:8));

            % nodes on the sides parallel to xi-axis
            idx = [9, 11, 17, 19];
            xia(idx)  = 0;
            etaa(idx) = [-1,  1, -1,  1];
            zetaa(idx)= [-1, -1,  1,  1];

            % N_i,xi
            shp(1,idx,gp) = -1/2*xintp(gp,1)*(1+xintp(gp,2)*etaa(idx)).*(1+xintp(gp,3)*zetaa(idx));
            % N_i,eta
            shp(2,idx,gp) =  1/4*etaa(idx).*(1-xintp(gp,1).^2).*(1+xintp(gp,3)*zetaa(idx));
            % N_i,zeta
            shp(3,idx,gp) =  1/4*zetaa(idx).*(1-xintp(gp,1).^2).*(1+xintp(gp,2)*etaa(idx));
            % N_i
            shp(4,idx,gp) =  1/4*(1-xintp(gp,1).^2).*(1+xintp(gp,2)*etaa(idx)).*(1+xintp(gp,3)*zetaa(idx));

            % nodes on the sides parallel to eta-axis
            idx = [10, 12, 18, 20];
            xia(idx)  = [1, -1,  1, -1];
            etaa(idx) = 0;
            zetaa(idx)= [-1, -1,  1,  1];

            % N_i,xi
            shp(1,idx,gp) =  1/4*xia(idx).*(1-xintp(gp,2).^2).*(1+xintp(gp,3)*zetaa(idx));
            % N_i,eta
            shp(2,idx,gp) = -1/2*xintp(gp,2)*(1+xintp(gp,1)*xia(idx)).*(1+xintp(gp,3)*zetaa(idx));
            % N_i,zeta
            shp(3,idx,gp) =  1/4*zetaa(idx).*(1+xintp(gp,1)*xia(idx))*(1-xintp(gp,2).^2);
            % N_i
            shp(4,idx,gp) = 1/4*(1+xintp(gp,1)*xia(idx)).*(1-xintp(gp,2).^2).*(1+xintp(gp,3)*zetaa(idx));

            % nodes on the sides parallel to zeta-axis
            idx = [13, 14, 15, 16];
            xia(idx)   = [-1,  1,  1, -1];
            etaa(idx)  = [-1, -1,  1,  1];
            zetaa(idx) = 0;

            % N_i,xi
            shp(1,idx,gp) =  1/4*xia(idx).*(1+xintp(gp,2)*etaa(idx)).*(1-xintp(gp,3).^2);
            % N_i,eta
            shp(2,idx,gp) =  1/4*etaa(idx).*(1+xintp(gp,1)*xia(idx)).*(1-xintp(gp,3).^2);
            % N_i,zeta
            shp(3,idx,gp) = -1/2*xintp(gp,3)*(1+xintp(gp,1)*xia(idx)).*(1+xintp(gp,2)*etaa(idx));
            % N_i
            shp(4,idx,gp)  =  1/4*(1+xintp(gp,1)*xia(idx)).*(1+xintp(gp,2)*etaa(idx)).*(1-xintp(gp,3).^2);
        end

    otherwise
        error('%s: number of nodes per element does not match any implemented hexahedral element',mfilename)
end

end