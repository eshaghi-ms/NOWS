clear
close all

%mesh_size = 0.125;
mesh_size = 0.5;

% Define the geometry for a plate with a single hole
model = createpde('structural','static-planestress');

% Geometry matrix for the outer plate (rectangle) and one circular hole
% S1: Plate defined as a rectangle with vertices (-6,-6), (18,-6), (18,18), (-6,18)
% C1: Hole defined as a circle centered at (6,6) with radius 3
gdm = [3, 4, -6, 18, 18, -6, -6, -6, 18, 18;
       1, 6, 6, 3, 0, 0, 0, 0, 0, 0]';
ns = char('S1','C1');
g = decsg(gdm, 'S1-C1', ns');
geometryFromEdges(model, g);
pdegplot(model, "EdgeLabels", "on", "FaceLabels", "on")
axis equal

% Material properties
structuralProperties(model,'YoungsModulus',1e3,...
                     'PoissonsRatio',0.3);

% Boundary conditions
% Bottom edge fixed in the y-direction only (roller BC)
structuralBC(model,'Edge',4,'Constraint', 'roller');

% Left edge fixed in the x-direction only (roller BC)
structuralBC(model,'Edge',3,'Constraint', 'roller');

% Apply a vertical traction load on the top edge
structuralBoundaryLoad(model,'Edge',2,'SurfaceTraction',[0; 10]);

% Mesh the geometry
generateMesh(model,'Hmax', mesh_size);
figure
pdemesh(model)

% Solve the PDE
result = solve(model);

% Extract the node coordinates, displacements, and stresses
node_coords = model.Mesh.Nodes';
ux_ref = result.Displacement.ux;
uy_ref = result.Displacement.uy;
sigma_xx_ref = result.Stress.sxx;
sigma_yy_ref = result.Stress.syy;
sigma_xy_ref = result.Stress.sxy;

% Save the results
save('elast2d_Plate_1_hole.mat', 'node_coords', 'ux_ref', 'uy_ref', 'sigma_xx_ref', 'sigma_yy_ref', 'sigma_xy_ref');

% Post-processing: Visualize the results
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
