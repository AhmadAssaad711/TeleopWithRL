"""
Configuration for bilateral pneumatic teleoperation with RL control.

Exact dynamics from:
  "Enhanced MRAC Design Using Filtered Regressors for
   Bilateral Pneumatic Teleoperation Control"
  Aya Abed & Naseem Daher, IC2AI 2025  (DOI: 10.1109/IC2AI62984.2025.10932214)

The RL agent outputs u_v — the servo-valve voltage — to make
the slave piston track the master.  Transparency is achieved
through passive pneumatic tubes connecting master ↔ slave.
There is NO underlying PD or MRAC controller.
"""

import numpy as np

# ================================================================== #
#  PNEUMATIC PLANT PARAMETERS  (Table I from paper)                    #
# ================================================================== #
AP        = 4.2072e-4    # Piston area  [m²]
AT        = 1.257e-5     # Tube cross-section area  [m²]
B_CRIT    = 0.21         # Critical pressure ratio (ISO 6358)  [-]
C_SONIC   = 4.5e-9       # Sonic conductance  [m³/(Pa·s)]
DT_TUBE   = 4e-3         # Tube diameter  [m]
L_TUBE    = 10.0         # Pneumatic line length  [m]
L_CYL     = 0.275        # Total cylinder stroke  [m]
MP        = 0.25         # Piston mass  [kg]
VMD       = 2e-6         # Dead volume  [m³]
BETA      = 11.6         # Viscous friction coeff  [Ns/m]
NU        = 1.57e-5      # Kinematic viscosity  [m²/s]
MU        = 1.813e-5     # Dynamic viscosity  [Ns/m²]
R_GAS     = 287.0        # Ideal gas constant (air)  [J/(kg·K)]
RHO0      = 1.204        # Reference air density  [kg/m³]

# ================================================================== #
#  ADDITIONAL PARAMETERS  (not in Table I — standard values)           #
# ================================================================== #
T_AIR     = 293.0        # Air / operating temperature  [K]  (20 °C)
T0_REF    = 293.0        # Reference temperature  [K]
P_ATM     = 101_325.0    # Atmospheric pressure  [Pa]
P_SUPPLY  = 600_000.0    # Compressed-air supply pressure  [Pa]  (6 bar)

# Servo-valve parameters  (typical for pneumatic proportional valves)
# Spool dynamics:  ẍ_v + 2·ζ_v·ω·ẋ_v + ω²·x_v  =  K_v·ω²·u_v   (Eq 8)
OMEGA_V   = 100.0        # Valve natural frequency  [rad/s]
ZETA_V    = 0.7          # Valve damping ratio  [-]
KV        = 0.1          # Electro-mechanical valve gain  [1/V]
                          #   u_v = ±10 V  →  x_v = ±1 (normalised)

# ================================================================== #
#  ENVIRONMENT MODELS  (from paper Section V)                          #
# ================================================================== #
# Modelled as spring-damper:  F_e = K_e·δx_s + B_e·ẋ_s
SKIN_BE   = 3e-3         # Skin damping  [Ns/m]
SKIN_KE   = 331.0        # Skin stiffness  [N/m]
FAT_BE    = 1e-3         # Fat damping  [Ns/m]
FAT_KE    = 83.0         # Fat stiffness  [N/m]
FREE_BE   = 0.0          # Free motion  [Ns/m]
FREE_KE   = 0.0          # Free motion  [N/m]

# ================================================================== #
#  HUMAN OPERATOR FORCE MODEL                                          #
# ================================================================== #
# Paper uses 10 N sinusoidal force on the master actuator (Sec V).
FH_AMP    = 10.0         # Force amplitude  [N]
FH_FREQ   = 0.5          # Force frequency  [Hz]

# ================================================================== #
#  SIMULATION                                                          #
# ================================================================== #
DT               = 0.0005        # physics time-step  [s]  (2 kHz)
SUB_STEPS        = 40            # physics steps per RL decision
                                  # → RL at 50 Hz  (40 × 0.5 ms = 20 ms)
EPISODE_DURATION = 10.0           # seconds per episode (paper: 30 s, reduced for training speed)
RL_DT            = DT * SUB_STEPS                   # 0.02 s
MAX_STEPS        = int(EPISODE_DURATION / RL_DT)     # 500 RL steps

# ================================================================== #
#  RL ACTION SPACE  (discrete servo-valve voltage)                     #
# ================================================================== #
V_LEVELS  = np.array([-10.0, -7.5, -5.0, -2.5, 0.0,
                         2.5,  5.0,  7.5, 10.0])    # 9 levels  [V]
N_ACTIONS = len(V_LEVELS)                             # 9

# ================================================================== #
#  STATE DISCRETISATION  (for Q-table)                                 #
# ================================================================== #
#  State tuple:  (pos_error_bin, vel_error_bin, F_h_bin, F_e_bin)
POS_ERROR_BINS = np.array([-0.03, -0.02, -0.01, -0.005,
                            0.0,   0.005,  0.01,  0.02, 0.03])  # 10 rgns
VEL_ERROR_BINS = np.array([-0.05, -0.02, -0.005,
                            0.005,  0.02,  0.05])                # 7 rgns
FH_BINS        = np.array([-8.0, -4.0, -1.0, 1.0, 4.0, 8.0])   # 7 rgns
FE_BINS        = np.array([-5.0, -2.0, -0.5, 0.5, 2.0, 5.0])   # 7 rgns
# Total states:  10 × 7 × 7 × 7 = 3 430
# Q-table size:  3 430 × 9 = 30 870 entries

# ================================================================== #
#  REWARD WEIGHTS                                                      #
# ================================================================== #
#  r = −α·(x_m − x_s)²  − γ·u_v²
ALPHA_TRACKING = 100.0
GAMMA_EFFORT   = 0.01
REWARD_CLIP    = 10.0

# ================================================================== #
#  Q-LEARNING HYPER-PARAMETERS                                         #
# ================================================================== #
LEARNING_RATE  = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9997
NUM_EPISODES   = 10_000
EVAL_EVERY     = 500
EVAL_EPISODES  = 3
