clear
close all

mesh_size = 0.125;
tol_rel_err = 1e-5;

% Define the geometry
model = createpde('structural','static-planestress');
gdm = [3, 4, -6, 18, 18, -6, -6, -6, 18, 18;
       1, 0, 0, 2, 0, 0, 0, 0, 0, 0;
       1, 12, 0, 1, 0, 0, 0, 0, 0, 0;
       1, 0, 12, 1, 0, 0, 0, 0, 0, 0;
       1, 12, 12, 3, 0, 0, 0, 0, 0, 0]'; % Geometry description matrix
ns = char('S1', 'C1', 'C2', 'C3', 'C4');
g = decsg(gdm,'S1-C1-C2-C3-C4', ns'); % Create geometry
geometryFromEdges(model,g);
pdegplot(g,"EdgeLabels","on","FaceLabels","on")

% Material properties
structuralProperties(model,'YoungsModulus',1e3,...
                     'PoissonsRatio',0.3);

% Boundary conditions
% Bottom edge fixed in the y-direction only
structuralBC(model,'Edge',4,'Constraint', 'roller');

% Left edge fixed in the x-direction only
structuralBC(model,'Edge',3,'Constraint', 'roller');


% Traction boundary condition at the top edge, applying a vertical load
structuralBoundaryLoad(model,'Edge',2,'SurfaceTraction',[0; 10]);


% Mesh the geometry
generateMesh(model,'Hmax', mesh_size); % Adjust 'Hmax' for finer mesh
figure
pdemesh(model)

% Solve the PDE
result = solve(model);

% Extract the node coordinates and corresponding displacement and stress values
node_coords = model.Mesh.Nodes';
ux_ref = result.Displacement.ux;
uy_ref = result.Displacement.uy;
sigma_xx_ref = result.Stress.sxx;
sigma_yy_ref = result.Stress.syy;
sigma_xy_ref = result.Stress.sxy;

% Save the data
save('elast2d_Plate_4_holes.mat', 'node_coords', 'ux_ref', 'uy_ref', 'sigma_xx_ref', 'sigma_yy_ref', 'sigma_xy_ref');

% Post-processing: Visualize the displacement and stress
figure;
pdeplot(model,'XYData',result.Displacement.ux, 'ColorMap', 'jet');
title('Horizontal Displacement (ux)');
xlabel('X');
ylabel('Y');
axis equal;

figure;
pdeplot(model,'XYData',result.Displacement.uy, 'ColorMap', 'jet');
title('Vertical Displacement (uy)');
xlabel('X');
ylabel('Y');
axis equal;

figure;
pdeplot(model,'XYData',result.Stress.sxx, 'ColorMap', 'jet');
title('xx Stress');
xlabel('X');
ylabel('Y');
axis equal;

figure;
pdeplot(model,'XYData',result.Stress.syy, 'ColorMap', 'jet');
title('yy Stress');
xlabel('X');
ylabel('Y');
axis equal;

figure;
pdeplot(model,'XYData',result.Stress.sxy, 'ColorMap', 'jet');
title('xy Stress');
xlabel('X');
ylabel('Y');
axis equal;

figure;
pdeplot(model,'XYData',result.VonMisesStress, 'ColorMap', 'jet');
title('Von Mises Stress');
xlabel('X');
ylabel('Y');
axis equal;
