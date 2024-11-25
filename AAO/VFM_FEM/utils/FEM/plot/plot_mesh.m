function plot_mesh(node,element,varargin)
% plot_mesh plots a finite element mesh.
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
default_labels = {'$x_1$' '$x_2$'};
default_linestyle = 'k-';
default_linewidth = 1.0;
default_z_value = 1000;

% input parser
p = inputParser;
addOptional(p,'newfigure',default_newfigure);
addOptional(p,'ELEMENT_TYPE',default_ELEMENT_TYPE);
addOptional(p,'labels',default_labels);
addOptional(p,'linestyle',default_linestyle);
addOptional(p,'linewidth',default_linewidth);
addOptional(p,'z_value',default_z_value);
parse(p,varargin{:});

% fill z values if needed
if (size(node,2) < 3)
    for c = size(node,2) + 1:3
        node(:,c) = p.Results.z_value*ones(size(node,1),1);
    end
end

% plot
if p.Results.newfigure
    figure
end
holdState = ishold;
hold on
for e=1:size(element,1)

    if ( strcmp(p.Results.ELEMENT_TYPE,'Q9') )      % 9-node quad element
        ord=[1,5,2,6,3,7,4,8,1];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'Q8') )  % 8-node quad element
        ord=[1,5,2,6,3,7,4,8,1];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'T3') )  % 3-node triangle element
        ord=[1,2,3,1];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'T6') )  % 6-node triangle element
        ord=[1,4,2,5,3,6,1];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'Q4') )  % 4-node quadrilateral element
        ord=[1,2,3,4,1];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'L2') )  % 2-node line element
        ord=[1,2];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'L3') )  % 3-node line element
        ord=[1,3,2];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'H4') )  % 4-node tet element
        ord=[1,2,4,1,3,4,2,3];
    elseif ( strcmp(p.Results.ELEMENT_TYPE,'B8') )  % 8-node brick element
        ord=[1,5,6,2,3,7,8,4,1,2,3,4,8,5,6,7];
    end

    for n=1:size(ord,2)
        xpt(n)=node(element(e,ord(n)),1);
        ypt(n)=node(element(e,ord(n)),2);
        zpt(n)=node(element(e,ord(n)),3);
    end
    plot3(xpt,ypt,zpt,p.Results.linestyle,'linewidth',p.Results.linewidth)
end

xlabel(p.Results.labels(1),'Interpreter','latex')
ylabel(p.Results.labels(2),'Interpreter','latex')

axis equal

if ( ~holdState )
    hold off
end


