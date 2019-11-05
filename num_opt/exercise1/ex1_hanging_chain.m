clc;                             % clear command window/console
clear;                           % clear workspace                                      
close all;                       % close all windows (plots)

import casadi.*

% create empty optimization problem
opti = casadi.Opti();

N = 40;                     % number of masses

% TODO: complete the definition of variables HERE
% HINT: opti.variable(n) creates a column vector of size n
%       opti.variable(n,m) creates a n-by-m matrix
%       so opti.variable(n,1) is the same as opti.variable(n)
y = ... ;
z = ... ;

m = 4/N;                    % mass
Di = (70/40)*N;             % spring constant
g0 = 9.81;                  % gravity

Vchain = 0;
for i = 1:N
    % TODO: complete the objective function (i.e. potential energy) HERE
    Vchain = Vchain + ...;
end

% pass objective to opti
opti.minimize(Vchain)

% TODO: complete the (equality) constraints HERE
opti.subject_to( ... )
...
   
% Setting solver and solving the problem:
opti.solver('ipopt')
sol = opti.solve();

% get solution and plot results
Y = sol.value(y);
Z = sol.value(z);

figure;
plot(Y,Z,'--or'); hold on;
plot(-2,1,'xg','MarkerSize',10);
plot(2,1,'xg','MarkerSize',10);
xlabel('y'); ylabel('z');
title('Optimal solution hanging chain (without extra constraints)')
