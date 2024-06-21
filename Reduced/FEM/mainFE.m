function [result] = mainFE(inputfile,matpar)
% DESCRIPTION
%  main routine
%
% INPUT
%  inputfile -- selected inputfile storing model specifications
%  matpar -- material parameters
%
% OUTPUT
%  result -- results structure

% control output of convergence information
outConv = false;

% read information from inputfile
if strcmp(inputfile,'cube')
    [model,coord,inz,bcond,load,lfun] = cube_tension(matpar);
elseif strcmp(inputfile,'plate_linElas')
    [model,coord,inz,bcond,load,lfun] = linElas_plate_with_hole(matpar);
elseif strcmp(inputfile,'plate_hyperelas_NH') % Neo-Hooke
    [model,coord,inz,bcond,load,lfun] = hyperelas_plate_with_hole_NH(matpar);
elseif strcmp(inputfile,'plate_hyperelas_IH') % Isihara
    [model,coord,inz,bcond,load,lfun] = hyperelas_plate_with_hole_NH(matpar);
elseif strcmp(inputfile,'plate_hyperelas_HW') % Haines-Wilson
    [model,coord,inz,bcond,load,lfun] = hyperelas_plate_with_hole_NH(matpar);
else 
    error('%s: wrong inputfile',mfilename);
end

% create ID- and LM-array (see Hughes - The finite element method)
[ID,LM,neq,ncs] = set_ID_LM(model,bcond,inz);

% store model data for partitioning system of linear equations
model.neq  = neq;
model.ncs  = ncs;
model.LM   = LM;

% determine location of integration points in reference domain and
% weighting factors
[xintp,weights] = set_space_integration(model.eset(1).elform,model.eset(1).npe);

% number of integration points per element
nintp = size(xintp,1);

% total number of integration points
ngp = model.eset(1).nelem*nintp;

% compute shape functions and derivatives w.r.t. local coordinates
shp = shape(model.eset(1).elform,model.eset(1).npe,xintp);

% store data in model-structure
model.ngp              = ngp;
model.eset(1).nintp    = nintp;
model.eset(1).start_gp = 1;
model.eset(1).shp      = shp;
model.eset(1).weights  = weights;

% perform time integration - for the case of constitutive models without
% internal variables, the time integration corresponds to a step-wise
% application of the loads
result = integrate(model,bcond,load,lfun,coord,inz,ID,outConv);

% visualization of displacement fields
if model.out.visualizeDisp
    % set labels
    label{1} = '$x_1$ in [mm]';
    label{2} = '$x_2$ in [mm]';
    % visualize axial displacements
    label{3} = '$u_1$ in [mm]';
    display_nodeVals(model,coord,ID,result.disp,1,label);
    % visualize lateral displacements
    label{3} = '$u_2$ in [mm]';
    display_nodeVals(model,coord,ID,result.disp,2,label);
end

% re-order displacements in result-structure for comparison with
% experimental data in calibration-setting + sum up horizontal and vertical
% reaction forces
nodes = (1:1:model.nnode)';
for i = 1 : size(model.out.time,1)
    if model.out.disp
        result.dispCalib{i}(1,:) = result.disp(ID(1,nodes),i);
        result.dispCalib{i}(2,:) = result.disp(ID(2,nodes),i);
    end
    % 
    if model.out.reactionForce
        % settings apply here for hyperelasticity example, to be adapted
        % for other cases
        result.reactionCalib(i,1) = sum(result.reactionForce(ID(1,nodes(abs(coord(:,1)-1)<=10*eps))-neq,i));
        result.reactionCalib(i,2) = sum(result.reactionForce(ID(2,nodes(abs(coord(:,2)-1)<=10*eps))-neq,i));
    end
end

end