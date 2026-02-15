"""
Configuration for bilateral pneumatic teleoperation with RL control.

Exact dynamics from:
  "Enhanced MRAC Design Using Filtered Regressors for
   Bilateral Pneumatic Teleoperation Control"
  Aya Abed & Naseem Daher, IC2AI 2025  (DOI: 10.1109/IC2AI62984.2025.10932214)

The RL agent outputs u_v (servo-valve voltage) to make
the slave piston track the master.
"""

import numpy as np

# ================================================================== #
#  PNEUMATIC PLANT PARAMETERS                                        #
# ================================================================== #
AP        = 4.2072e-4    # Piston area [m^2]
AT        = 1.257e-5     # Tube cross-section area [m^2]
B_CRIT    = 0.21         # Critical pressure ratio (ISO 6358) [-]
C_SONIC   = 4.5e-9       # Sonic conductance [m^3/(Pa*s)]
DT_TUBE   = 4e-3         # Tube diameter [m]
L_TUBE    = 10.0         # Pneumatic line length [m]
L_CYL     = 0.275        # Total cylinder stroke [m]
MP        = 0.25         # Piston mass [kg]
VMD       = 2e-6         # Dead volume [m^3]
BETA      = 11.6         # Viscous friction coeff [Ns/m]
NU        = 1.57e-5      # Kinematic viscosity [m^2/s]
MU        = 1.813e-5     # Dynamic viscosity [Ns/m^2]
R_GAS     = 287.0        # Ideal gas constant (air) [J/(kg*K)]
RHO0      = 1.204        # Reference air density [kg/m^3]

# ================================================================== #
#  ADDITIONAL PARAMETERS                                             #
# ================================================================== #
T_AIR     = 293.0        # Air temperature [K]
T0_REF    = 293.0        # Reference temperature [K]
P_ATM     = 101_325.0    # Atmospheric pressure [Pa]
P_SUPPLY  = 600_000.0    # Supply pressure [Pa]

OMEGA_V   = 100.0        # Valve natural frequency [rad/s]
ZETA_V    = 0.7          # Valve damping ratio [-]
KV        = 0.1          # Valve gain [1/V]

# ================================================================== #
#  ENVIRONMENT MODELS                                                #
# ================================================================== #
SKIN_BE   = 3e-3         # Skin damping [Ns/m]
SKIN_KE   = 331.0        # Skin stiffness [N/m]
FAT_BE    = 1e-3         # Fat damping [Ns/m]
FAT_KE    = 83.0         # Fat stiffness [N/m]
FREE_BE   = 0.0          # Free motion [Ns/m]
FREE_KE   = 0.0          # Free motion [N/m]

# ================================================================== #
#  HUMAN OPERATOR FORCE MODEL                                        #
# ================================================================== #
FH_AMP    = 10.0         # Force amplitude [N]
FH_FREQ   = 0.5          # Force frequency [Hz]

# ================================================================== #
#  SIMULATION                                                        #
# ================================================================== #
DT               = 0.0005
SUB_STEPS        = 40
EPISODE_DURATION = 10.0
RL_DT            = DT * SUB_STEPS
MAX_STEPS        = int(EPISODE_DURATION / RL_DT)

# ================================================================== #
#  ENVIRONMENT MODES                                                 #
# ================================================================== #
ENV_MODE_CONSTANT = "constant_skin"
ENV_MODE_CHANGING = "changing_skin_fat"
ENV_SWITCH_TIME   = EPISODE_DURATION / 2.0
ENV_LABELS        = ("skin", "fat")
N_ENV_CONTEXTS    = len(ENV_LABELS)

# ================================================================== #
#  ACTION SPACE                                                      #
# ================================================================== #
V_LEVELS  = np.array([-10.0, -7.5, -5.0, -2.5, 0.0,
                       2.5,   5.0,  7.5, 10.0])
N_ACTIONS = len(V_LEVELS)

# ================================================================== #
#  STATE DISCRETISATION (tabular Q-learning)                         #
# ================================================================== #
# Core tracking features
POS_ERROR_BINS = np.array([
    -0.24, -0.18, -0.14, -0.10, -0.07, -0.05, -0.03, -0.015,
     0.0,
     0.015, 0.03, 0.05, 0.07, 0.10, 0.14, 0.18, 0.24
])
VEL_ERROR_BINS = np.array([
    -2.20, -1.60, -1.20, -0.90, -0.60, -0.35, -0.15,
     0.15,  0.35,  0.60,  0.90,  1.20,  1.60,  2.20
])
FH_BINS        = np.array([
    -14.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
      1.0,   3.0,  5.0,  7.0,  9.0, 11.0, 14.0
])
FE_BINS        = np.array([
    -45.0, -36.0, -28.0, -21.0, -15.0, -10.0, -6.0, -2.0,
      2.0,   6.0,  10.0,  15.0,  21.0,  28.0, 36.0, 45.0
])

# Added internal plant features for RL state:
# pressure differentials, tube flows, and spool position.
PM_DIFF_BINS = np.array([
    -450_000.0, -300_000.0, -200_000.0, -120_000.0, -60_000.0, -20_000.0,
      20_000.0,   60_000.0,  120_000.0,  200_000.0,  300_000.0, 450_000.0
])
PS_DIFF_BINS = PM_DIFF_BINS.copy()

FLOW_BINS = np.array([
    -0.0040, -0.0025, -0.0015, -0.0007,
     0.0000,
     0.0007,  0.0015,  0.0025,  0.0040
])

SPOOL_POS_BINS = np.array([
    -0.80, -0.50, -0.25, -0.10,
     0.10,  0.25,  0.50,  0.80
])

# ================================================================== #
#  TERMINATION + REWARD SCALING                                      #
# ================================================================== #
POS_ERROR_FAIL_THRESHOLD = 0.24

MAX_POSITION_ERROR = POS_ERROR_FAIL_THRESHOLD
POS_ERR_NORM_CLIP = 1.0

# Power normalization from model parameters:
# |F_e * v_m - F_h * v_s| <= |F_e|*|v_m| + |F_h|*|v_s|
# with |F_e| <= SKIN_KE*(L_CYL/2) + SKIN_BE*V_MAX_GEOM,
# |F_h| <= FH_AMP, V_MAX_GEOM = L_CYL/RL_DT.
V_MAX_GEOM = L_CYL / RL_DT
F_E_MAX_THEORETICAL = SKIN_KE * (L_CYL / 2.0) + SKIN_BE * V_MAX_GEOM
MAX_POWER_ERROR_THEORETICAL = V_MAX_GEOM * (F_E_MAX_THEORETICAL + FH_AMP)
# Practical scale for learning (about 95th percentile under random policy).
MAX_POWER_ERROR = 10.0

# Reward weights
ALPHA_TRACKING = 40.0
GAMMA_EFFORT   = 0.01
REWARD_CLIP    = 50.0
BETA_TRANSPARENCY = 5.0

# ================================================================== #
#  MRAC DEFAULT PARAMETERS                                           #
# ================================================================== #
MRAC_WN       = np.pi
MRAC_ZETA     = 0.7
MRAC_A0       = 8.0
MRAC_GAMMA1   = 6.0
MRAC_GAMMA2   = 0.8
MRAC_THETA0   = np.array([-6.0, 1.8, 6.6, 0.0, 0.0], dtype=np.float64)
MRAC_U_CLIP   = 10.0

# ================================================================== #
#  Q-LEARNING HYPER-PARAMETERS                                       #
# ================================================================== #
LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.05
NUM_EPISODES    = 10_000
EPSILON_DECAY   = (EPSILON_END / EPSILON_START) ** (1.0 / NUM_EPISODES)
EVAL_EVERY      = 500
EVAL_EPISODES   = 50

# ================================================================== #
#  RESULT FOLDER LAYOUT                                              #
# ================================================================== #
RESULTS_ROOT_DIR  = "results"
RL_CONSTANT_DIR   = "rl_constant"
RL_CHANGING_DIR   = "rl_changing"
MRAC_RESULTS_DIR  = "mrac"
COMPARE_RESULTS_DIR = "comparisons"
