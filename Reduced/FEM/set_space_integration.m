function [xintp,weights] = set_space_integration(elform,npe)
% DESCRIPTION
%  function to provide the integration point positions and weighting
%  factors using gauss-integration
%
% INPUT
%  elform -- switch for element type
%  npe -- nodes per element
%
% OUTPUT
%  xintp -- integration point coordinates within unit domain [-1 1]
%  weights -- weighting factors for spatial integration

% initialize counter for integration points
gp = 0;

switch elform

    case 'quad' % quadrilateral element

        % space dimension of choosen element
        ndim = 2;

        switch npe
            case 4  % bilinear Q4-element with four nodes

                % required integration order
                intOrder = 2;

                % number of integration points in one direction
                [num_gp,gp_coordinate,gp_weight] = integration_order(intOrder);

                % initialize variables
                xintp = zeros(num_gp^ndim,ndim);
                weights = zeros(num_gp^ndim,1);

                for i = 1 : num_gp
                    for j = 1 : num_gp
                        gp = gp + 1;
                        weights(gp) = gp_weight(i)*gp_weight(j);
                        xintp(gp,:) = [gp_coordinate(i),gp_coordinate(j)];
                    end
                end
            
            case 8  % quadrilateral element with eight nodes

                % required integration order
                intOrder = 4;

                % number of integration points in one direction
                [num_gp,gp_coordinate,gp_weight] = integration_order(intOrder);

                % initialize variables
                xintp = zeros(num_gp^ndim,ndim);
                weights = zeros(num_gp^ndim,1);

                for i = 1 : num_gp
                    for j = 1 : num_gp
                        gp = gp + 1;
                        weights(gp) = gp_weight(i)*gp_weight(j);
                        xintp(gp,:) = [gp_coordinate(i),gp_coordinate(j)];
                    end
                end

            otherwise
                error('%s: number of nodes per element does not match any implemented quadrilateral element',mfilename);
        end

    case 'hex'

        % space dimension of choosen element
        ndim = 3;

        switch npe

            case 8

                % required integration order
                intOrder = 2;

                % number of integration points in one direction
                [num_gp,gp_coordinate,gp_weight] = integration_order(intOrder);

                % initialize variables
                xintp = zeros(num_gp^ndim,ndim);
                weights = zeros(num_gp^ndim,1);

            case 20

                % required integration order
                intOrder = 4;

                % number of integration points in one direction
                [num_gp,gp_coordinate,gp_weight] = integration_order(intOrder);

                % initialize variables
                xintp = zeros(num_gp^ndim,ndim);
                weights = zeros(num_gp^ndim,1);

            otherwise
                error('%s: number of nodes per element does not match any implemented hexahedral element',mfilename);
        end

        for i = 1 : num_gp
            for j = 1 : num_gp
                for k = 1 : num_gp
                    gp = gp + 1;
                    weights(gp) = gp_weight(i)*gp_weight(j)*gp_weight(k);
                    xintp(gp,:) = [gp_coordinate(i),gp_coordinate(j),gp_coordinate(k)];
                end
            end
        end

    otherwise
        error('%s: element form not implemented',mfilename)
end

end

%% subfunction

function [num_gp,gp_coordinate,gp_weight] = integration_order(order)

switch order
    case 1

        % number of integration points
        num_gp = 1;

        % Gausspoint coordinates and weights in one direction
        gp_coordinate(1) = 0;
        gp_weight(1) = 2;

    case {2,3}

        % number of integration points
        num_gp = 2;

        % Gausspoint coordinates and weights in one direction
        gp_coordinate(1) = -sqrt(1/3);
        gp_coordinate(2) = sqrt(1/3);
        gp_weight(1) = 1;
        gp_weight(2) = 1;

    case {4,5}

        % number of integration points
        num_gp=3;

        % Gausspoint coordinates and weights in one direction
        gp_coordinate(1) = -sqrt(3/5);
        gp_coordinate(2) = 0;
        gp_coordinate(3) = sqrt(3/5);
        gp_weight(1) = 5/9;
        gp_weight(2) = 8/9;
        gp_weight(3) = 5/9;

    otherwise
        error('%s: integration order not implemented!',mfilename)
end

end