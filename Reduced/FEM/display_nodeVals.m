function display_nodeVals(model,coord,ID,nodeField,idx,label)
% DESCRIPTION
%  function to visualize values on nodes in undeformed configuration of
%  two-dimensional spatial discretization
%
% INPUT
%  model -- structure containing information about the specified model
%  coord -- node coordinates
%  ID -- ID-array 
%  nodeField -- node values for visualization
%  idx -- index of degree of freedom for visualization (used in ID)
%  label -- cell-array with axis and colorbar labels
%
% OUTPUT
%  none

% perform check
if model.ndim ~= 2
    error('%s: visualization only implemented for two-dimensional problems',mfilename);
end

% switch for plotting contour of mesh
plotMesh = true;

% auxiliary points for visualization of contour plot
xi = -1:0.5:1;

figure;
for i = 1 : model.eset.nelem
    
    % get element nodes and node coordinates
    elemNodes = model.eset.incidence(i,:)';
    nodeCoord = coord(elemNodes,:);

    nodeVals = nodeField(ID(idx,elemNodes),1);
        
    % evaluate coordinates and node values on auxiliary points for contour
    % fill
    for j = 1 : size(xi,2)
        for k = 1 : size(xi,2)
            
            % evaluate shape functions
            xia  = [-1 1 1 -1]';
            etaa = [-1 -1 1 1]';
            shp = 0.25*(1 + xia*xi(j)) .* (1 + etaa*xi(k));

            X(j,k) = shp'*nodeCoord(:,1);
            Y(j,k) = shp'*nodeCoord(:,2);
            Z(j,k) = shp'*nodeVals(:,1);

        end
    end

    surface(X,Y,Z,'EdgeColor','none');

end

% set labels
xlabel(label{1},'Interpreter','latex','FontSize',16);
ylabel(label{2},'Interpreter','latex','FontSize',16);

if plotMesh
    hold on
    for i = 1 : model.eset.nelem
        nodeCoord = coord(model.eset.incidence(i,:)',:);
        % extend coordinates for closed element circumference
        nodeCoord = [nodeCoord; nodeCoord(1,:)];
        plot3(nodeCoord(:,1),nodeCoord(:,2),1000*ones(size(nodeCoord,1),1),'Color','black');
    end
    hold off
end

axis equal
colormap(jet);
c = colorbar;
c.Label.Interpreter = 'latex';
% c.TickLabelInterpreter = 'latex';
c.Label.FontSize = 16;
c.Label.String = label{3};

end