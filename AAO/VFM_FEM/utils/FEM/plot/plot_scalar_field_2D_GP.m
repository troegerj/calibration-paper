function plot_scalar_field_2D_GP(scalar_field,name,ELEMENT_TYPE,element,node,newfigure)
% =========================================================================
% Plots a scalar field for given elements and connectivity.
% For each element the the scalar field values at the Gauss points are
% averaged.
% =========================================================================
% plot_scalar_field_2D_GP plots a scalar quantity on a finite element mesh.
%
% ## Comments
% 
% For each element the the scalar field values at the Gauss points are
% averaged.
%
% ## Input Arguments
%
% `scalar_field` (_double_) - Gauss point values of the plotted quantity
%
% `name` (_char_) - name of the plotted quantity
%
% `ELEMENT_TYPE` (_char_) - element type (e.g. `'Q4'`)
% 
% `element` (_double_) - element connectivity of the finite element mesh
%
% `node` (_double_) - nodes of the finite element mesh
%
% `newfigure` (_logical_) - specifies whether a new figure should be
% created
% 
% ## Output Arguments
% 
% _none_
% 

if nargin == 5
    newfigure = true; 
end

if newfigure
    figure
end
    
hold on

switch ELEMENT_TYPE
    
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

for ele=1:size(element,1)
    
    elementnode= element(ele,:);
    coord=node(elementnode,:);
    
    x = coord(:,1);
    y = coord(:,2);
    n_Gauss_per_element = size(scalar_field,2)*size(scalar_field,3);
    z = sum(sum(scalar_field(ele,:,:)))/n_Gauss_per_element * ones(size(x));
    
    if size(element,1)<20
        xi = -1:0.02:1;
    else
        xi = -1:0.2:1;
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

xlabel('$x_1$');
ylabel('$x_2$');
xlim([min(node(:,1)),max(node(:,1))])
ylim([min(node(:,2)),max(node(:,2))])
colormap(jet);
colorbar;
if ( ~strcmp(name,'') )
    title(name,'fontweight','bold');
end
axis equal
axis square


