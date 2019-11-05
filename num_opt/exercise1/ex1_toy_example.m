clc;                             % clear command window/console
clear;                           % clear workspace                                      
close all;                       % close all windows (plots)

import casadi.*

% create empty optimization problem
opti = casadi.Opti();

% define variables
x = opti.variable();
% y = ...

% define objective
f = x^2 - 2*x ;

% hand objective to casadi, no minimization done yet
opti.minimize(f)                        % opti.minimize(x^2 - 2*x) would also work

% define constraints. To include several constraints, just call this
% function several times
% opti.subject_to(   x >= 1.5);
% opti.subject_to( x < 8)

% define solver
opti.solver('ipopt')                    % Use IPOPT as solver

% solve optimization problem
sol = opti.solve();

% read and print solution
xopt = sol.value(x);

disp(' ')
if strcmp(sol.stats.return_status, 'Solve_Succeeded')      % if casadi returns sucessfull
    disp(['Optimal solution found: x = ' num2str(xopt)]);
else
    disp('Problem failed')
end
