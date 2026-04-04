% EVERYTHING IN SI UNITS

close all
clearvars
clc

%% Hand Constants
m_h=0*11.6;         %mass of the hand in kg
B_h=0*17.26;        %damping of the hand N.s/m
k_h=0*243.2;        %stiffness of the hand N/m

%% General
R=287;              % specific gas constant J/(kg.K)
rho0 = 1.204;       % kg/m^3
T = 273.15+20;      % K
P_s = 3e5;          % Pa
P_atm = 1.013e5;    % Pa

%% Tube Constants
D_t = 4e-3;         % diameter of tube in m
A_t=pi*D_t^2/4;     % area of the inlet m^2
L_t=10;             % length of the line in m
mui=1.813e-5;       % dynamic viscosity in Ns/m^2

%% Cylinder Constants
m_p=0.25;           % kg
beta=0.33*35;          % N.s/m
l_cyl=0.275;        % m
A_p=4.2072e-4;      % m^2
V_md=2.4797*10^-5;    % m^3 Calculated from data sheet 2.4797*10^-5
V_sd=V_md;
P_md = R*T*rho0;

%% Valve Constants
C_v = 4.5e-9;       % m^3/Pa.s
b_v = 0.21;         % []
omega_v = 150 %400*2*pi; % rad/s
K_v = 1/5;          % 1/V
zeta_v = 0.7;       % []

%% Environment impedance

% No environment
% B_e=0; K_e=0; m_e = 0;

% Spring environment 
%B_e=0.1212; K_e=1.2784; m_e=0;

% Stress Ball environment (Leen - in SI)
% K_e = 2124.4; B_e = 0406.9;

% Journal Spring Environment
%B_e = 0; K_e = 0*0.400; m_e = 0;

% Baayoun's Skin Ennvironment (SI)
B_e = 3*10^-3 ; K_e = 331; m_e=0;

% Baayoun's Fat Environment (SI)
%B_e = 10^-3 ; K_e = 83;

% Baayoun's Muscle Environment (SI)
%K_e = 497; B_e = 3e-3;

%% Sampling time
Ts=0.001;

%% Reference Model
omega_mod = pi; %pi
zeta_mod = 0.707; %0.707
Am = [1 2*zeta_mod*omega_mod omega_mod^2];
Bm = (omega_mod)^2;
am0 = Am(1); am1=Am(2); am2=Am(3);

%% Q AND P
A0 = [1 10];
a01 = A0(1);
P1 = [am0 am1 am2];
P2 = A0;
Q = [1 a01+am1 am1*a01+am2 am2*a01];
P = [1 a01+am1 am1*a01+am2 am2*a01];
A0Am = [1 a01+am1 am1*a01+am2 am2*a01];

%% Linearizaton

x_mE = l_cyl/2;
P_m1E = -(P_s*b_v - P_atm*b_v + (P_atm^2*b_v^2 - 2*P_atm^2*b_v + 2*P_atm^2 - 2*P_atm*P_s*b_v^2 + 5*P_s^2*b_v^2 - 6*P_s^2*b_v + 2*P_s^2)^(1/2))/(2*(b_v - 1));
c1 = R*T/(V_md+x_mE*A_p);
c2 = P_m1E*A_p/(V_md+x_mE*A_p);
P_m2E = P_m1E;

c3 = R*T/(V_md+(l_cyl-x_mE)*A_p);
c4 = P_m2E*A_p/(V_md+(l_cyl-x_mE)*A_p);
x_sE = l_cyl/2;
P_s1E = P_m1E;
c5 = R*T/(V_sd+x_sE*A_p);
c6 = P_s1E*A_p/(V_sd+x_sE*A_p);
P_s2E = P_m1E;
c7 = R*T/(V_sd+(l_cyl-x_sE)*A_p);
c8 = P_s2E*A_p/(V_sd+(l_cyl-x_sE)*A_p);
c9 = C_v*rho0*P_s*sqrt(1-((P_m1E/P_s-b_v)/(1-b_v))^2);
c10 = C_v*rho0*P_s*sqrt(1-((P_m2E/P_s-b_v)/(1-b_v))^2);
c11 = A_t/L_t;
c12 = R*T/A_t/L_t;
c13 = k_h/m_h;
c14 = B_h/m_h;
c15 = 1/m_h;
c16 = k_h/m_p;
c17 = B_h/m_p;
c18 = (beta+B_h)/m_p;
c19 = (A_p)/m_p;
c20 = beta/m_p;
c21 = 1/m_p;
c22 = 32*mui/rho0/D_t/D_t;
c23 = 1/(1+c5/c12);
c24 = 1/(1+c7/c12);

A = [0,      1,       0,       0,       0,       0,       0;...
     -(K_e)/(m_p+m_e),    -(beta+B_e)/(m_p+m_e), -1/(m_p+m_e)*A_p,    0,    0,   0,   0;...
     0,      (c6+c8), 0,       (c7+c5), 0,      0,       0;...
     0,      0,       -c11,    -c22,  0,       0,       c11;...
     0,      0,       0,       0,       0,       1,       0;...
     0,      0,       0,       0,       -1/(m_p+m_h)*k_h,   -1/(m_p+m_h)*(beta),  1/(m_p+m_h)*A_p;...
     0 ,     0,       0,       -c12/(c12+c1)*(c1+c3),   0,  -c12/(c12+c1)*(c2+c4),   -K_v*c12/(c12+c1)*(c1*c9+c3*c10);];

B = [0                    0                      ;...
        0                    0                      ;...
        0                    0                      ;...
        0                    0                      ;...
        0                    0                      ;...
        1/(m_p+m_h)       0                      ;...
        0         K_v*c12/(c12+c1)*(c1*c9+c3*c10)]  ;

C = eye(7,7);

D = zeros(7,2);

sys=ss(A,B,C,D);

