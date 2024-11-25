function [mesh_FOV,ux_FOV,uy_FOV] = get_FOV(FOV_bounds,node_given,element_given,x1_given,x2_given,ux_given,uy_given)
% get_FOV gets the field of view of a given finite element mesh.

% tolerance
TOL = 1e-9;

% compute element center
element_center = zeros(size(element_given,1),size(node_given,2));
for idx_node = 1:size(element_given,2)
    element_center = element_center + node_given(element_given(:,idx_node),:) / size(element_given,2);
end

% bounds of the field of view
x1_min = FOV_bounds(1,1);
x1_max = FOV_bounds(1,2);
x2_min = FOV_bounds(2,1);
x2_max = FOV_bounds(2,2);

% find elements in the field of view
FOV_elements = logical(...
    (element_center(:,1) >= x1_min) ...
    .* (element_center(:,1) <= x1_max) ...
    .* (element_center(:,2) >= x2_min) ...
    .* (element_center(:,2) <= x2_max) );
mesh_FOV.element = element_given(FOV_elements,:);

% note: the connectivity array mesh_FOV.element contains the node numbers
% of the given mesh
% we need to define a new mesh and map the node numbers of the given mesh
% to the node numbers of the new mesh
FOV_n_node = length(unique(squeeze(mesh_FOV.element)));
mesh_FOV.node = zeros(FOV_n_node,2);
map = zeros(FOV_n_node,1);
counter = 1; % counter over all nodes in the new mesh
% loop over all elements in the field of view
for idx_ele = 1:size(mesh_FOV.element,1)
    % loop over all nodes of each element in the field of view
    for idx_node = 1:size(mesh_FOV.element,2)
        % get node number of the given mesh
        node_number = mesh_FOV.element(idx_ele,idx_node);
        % check if the the node has been already assigned
        if sum(node_number == map) ~= 1
            mesh_FOV.node(counter,:) = node_given(node_number,:);
            map(counter) = node_number;
            counter = counter + 1;
        end
        % assign node number of the new mesh
        mesh_FOV.element(idx_ele,idx_node) = find(map==node_number);
    end
end

% find the displacements in the field of view
ux_FOV = zeros(size(ux_given));
uy_FOV = zeros(size(uy_given));
% loop over all nodes in the field of view
for idx_node = 1:size(mesh_FOV.node,1)
    node = mesh_FOV.node(idx_node,:);
    node_pos = find((abs(x1_given - node(1)) < TOL) .* (abs(x2_given - node(2)) < TOL));
    ux_FOV(idx_node) = ux_given(node_pos);
    uy_FOV(idx_node) = uy_given(node_pos);
end

end