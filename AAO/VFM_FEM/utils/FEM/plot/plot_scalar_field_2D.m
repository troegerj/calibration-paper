function plot_scalar_field_2D(node,element,scalar_field,varargin)
% plot_scalar_field_2D plots a scalar quantity on a finite element mesh.
%
% ## Comments
% 
% _none_
% 
% ## Input Arguments
%
% `node` (_double_) - nodes of the finite element mesh
%
% `element` (_double_) - element connectivity of the finite element mesh
%
% `scalar_field` (_double_) - nodal values of the plotted quantity
%
% `varargin` - optional arguments
%
% ## Output Arguments
% 
% _none_
% 

% default values
default_newfigure = true;
switch size(element,2)
    case 3
        default_ELEMENT_TYPE = 'T3';
    case 4
        default_ELEMENT_TYPE = 'Q4';
end
default_mesh_overlay = false;
default_labels = {'$x_1$' '$x_2$'};
default_clabel = '';

% input parser
p = inputParser;
addOptional(p,'newfigure',default_newfigure);
addOptional(p,'ELEMENT_TYPE',default_ELEMENT_TYPE);
addOptional(p,'mesh_overlay',default_mesh_overlay);
addOptional(p,'labels',default_labels);
addOptional(p,'clabel',default_clabel);
parse(p,varargin{:});

% plot
if p.Results.newfigure
    figure
end

hold on;

switch p.Results.ELEMENT_TYPE
    
    case 'Q4'
        
    case 'T3'
        
        % triangles -> rectangles
        element_Q4 = zeros(size(element,1)/2,size(element,2)+1);
        for i = 1:size(element_Q4,1)
            element_Q4(i,1:2) = element(2*i-1,1:2);
            element_Q4(i,3:4) = element(2*i,2:3);
        end
        element = element_Q4;
        
end

for ele = 1:size(element,1)
    
    elementnode = element(ele,:);
    coord = node(elementnode,:);
    
    x = coord(:,1);
    y = coord(:,2);
    z = scalar_field(elementnode,1);
    
    if size(element,1) < 20
        xi = -1:0.02:1;
    elseif size(element,1) < 200
        xi = -1:0.2:1;
    else
        xi = -1:0.5:1;
    end
    
    X = zeros(size(xi,2),1);
    Y = zeros(size(xi,2),1);
    Z = zeros(size(xi,2));
    
    for i=1:size(xi,2)
        
        for j=1:size(xi,2)
            
            [N, ~] = lagrange_basis('Q4',[xi(i),xi(j)]);
            
            X(i,j)=N'*x;
            Y(i,j)=N'*y;
            Z(i,j)=N'*z;
            
        end
        
    end
    
    surface(X,Y,Z,'EdgeColor','none');
    
end

xlabel(p.Results.labels(1),'Interpreter','latex')
ylabel(p.Results.labels(2),'Interpreter','latex')

colormap(jet);
c = colorbar;
if ( ~strcmp(p.Results.clabel,'') )
    c.Label.Interpreter = 'latex';
    c.Label.String = p.Results.clabel;
end

if p.Results.mesh_overlay
    plot_mesh(node,element,'newfigure',false,'ELEMENT_TYPE',p.Results.ELEMENT_TYPE)
end

axis equal


