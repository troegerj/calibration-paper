function mesh = bc_plate_w_hole_tension(mesh)

%% Nodes
TOL = 1e-9;
x_min = min(mesh.node(:,1));
x_max = max(mesh.node(:,1));
y_min = min(mesh.node(:,2));
y_max = max(mesh.node(:,2));
mesh.nodes_left = find(abs(mesh.node(:,1) - x_min) < TOL);
mesh.nodes_right = find(abs(mesh.node(:,1) - x_max) < TOL);
mesh.nodes_top = find(abs(mesh.node(:,2) - y_max) < TOL);
mesh.nodes_bottom = find(abs(mesh.node(:,2) - y_min) < TOL);

mesh.dof_left = sort([mesh.nodes_left*2-1; mesh.nodes_left*2]);
mesh.dof_left_x = sort(mesh.nodes_left*2-1);
mesh.dof_left_y = sort(mesh.nodes_left*2);
mesh.dof_right = sort([mesh.nodes_right*2-1; mesh.nodes_right*2]);
mesh.dof_right_x = sort(mesh.nodes_right*2-1);
mesh.dof_right_y = sort(mesh.nodes_right*2);
mesh.dof_top = sort([mesh.nodes_top*2-1; mesh.nodes_top*2]);
mesh.dof_bottom = sort([mesh.nodes_bottom*2-1; mesh.nodes_bottom*2]);
mesh.dof_bottom_x = sort(mesh.nodes_bottom*2-1);
mesh.dof_bottom_y = sort(mesh.nodes_bottom*2);

%nodes_fix = mesh.nodes_right'; % fixed nodes
%nodes_displacement = mesh.nodes_left'; % displaced nodes

%% Degrees of Freedom
mesh.dof_free = mesh.dof;
mesh.dof_free = setdiff(mesh.dof_free,mesh.dof_left_x);
mesh.dof_free = setdiff(mesh.dof_free,mesh.dof_right_x);
mesh.dof_free = setdiff(mesh.dof_free,mesh.dof_bottom_y);
mesh.n_reaction = 1;
mesh.dof_reaction = (mesh.dof_left_x)';
% the displacement is assumed to be known at the boundaries where the
% forces are unknown
mesh.dof_known = [mesh.dof_left_x', mesh.dof_bottom_y', mesh.dof_right_x'];
% mesh.dof_known = [ ]; % no displacement values are assumed to be known
mesh.dof_unknown = mesh.dof; mesh.dof_unknown(mesh.dof_known) = [];
