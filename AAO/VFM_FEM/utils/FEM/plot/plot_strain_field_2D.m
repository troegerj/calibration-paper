function plot_strain_field_2D(ux,uy,strain_component,name,ELEMENT_TYPE,element,node,newfigure)
% plot_strain_field_2D plots the strain field on a finite element mesh for
% a given displacement field.
%
% ## Comments
% 
% _none_
% 
% ## Input Arguments
%
% `ux` (_double_) - nodal displacement values
%
% `ux` (_double_) - nodal displacement values
%
% `strain_component` (_double_) - strain component to be plotted (1, 2 or 3)
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

hold on;

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

for ele = 1:size(element,1)
    
    elementnode = element(ele,:);
    coord = node(elementnode,:);

    x = coord(:,1);
    y = coord(:,2);
    ux_ele = ux(elementnode,1);
    uy_ele = uy(elementnode,1);
    u_ele = zeros(length(ux_ele)*2,1);
    u_ele(1:2:end) = ux_ele;
    u_ele(2:2:end) = uy_ele;
    
    if size(element,1)<20
        xi = -1:0.01:1;
    elseif size(element,1)<200
        xi = -1:0.05:1;
    else
        xi = -1:0.1:1;
    end
    
    X = zeros(size(xi,2),1);
    Y = zeros(size(xi,2),1);
    Z = zeros(size(xi,2));
    
    for i=1:size(xi,2)
        
        for j=1:size(xi,2)
            
            [N, dNdxi] = lagrange_basis('Q4',[xi(i),xi(j)]);
            J = coord'*dNdxi;
            dNdx = dNdxi / J;
            B = zeros(3,8);
            B(1,1:2:end-1) = dNdx(:,1)';
            B(2,2:2:end) = dNdx(:,2)';
            B(3,1:2:end-1) = dNdx(:,2)';
            B(3,2:2:end) = dNdx(:,1)';

            epsilon = B*u_ele;
            
            X(i,j)=N'*x;
            Y(i,j)=N'*y;
            Z(i,j)=epsilon(strain_component);
            
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


