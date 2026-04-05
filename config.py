"""
Configuration for F_h-driven bilateral pneumatic teleoperation with RL control.

Exact dynamics from:
  "Enhanced MRAC Design Using Filtered Regressors for
   Bilateral Pneumatic Teleoperation Control"
  Aya Abed & Naseem Daher, IC2AI 2025  (DOI: 10.1109/IC2AI62984.2025.10932214)

This top-level TeleopWithRL package is the active working copy for the
F_h master-input experiments. The archived reference-tracking baseline
is preserved separately under reference_tracking_RL/TeleopWithRL.
"""

import os

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
VMD       = 2.4797e-5    # Dead volume [m^3] (matches MATLAB ParmsOriginal.m)
BETA      = 0.33 * 35.0  # Viscous friction coeff [Ns/m] (matches SI_NonLinear.m)
NU        = 1.57e-5      # Kinematic viscosity [m^2/s]
MU        = 1.813e-5     # Dynamic viscosity [Ns/m^2]
R_GAS     = 287.0        # Ideal gas constant (air) [J/(kg*K)]
RHO0      = 1.204        # Reference air density [kg/m^3]

# ================================================================== #
#  ADDITIONAL PARAMETERS                                             #
# ================================================================== #
T_AIR     = 273.15 + 20.0  # Air temperature [K] (matches SI_NonLinear.m)
T0_REF    = 273.15 + 20.0  # Reference temperature [K]
P_ATM     = 1.013e5        # Atmospheric pressure [Pa]
P_SUPPLY  = 3e5            # Supply pressure [Pa]

OMEGA_V   = 400.0 * 2.0 * np.pi  # Valve natural frequency [rad/s] (matches MATLAB GUI)
ZETA_V    = 0.7          # Valve damping ratio [-]
KV        = 1.0 / 5.0    # Valve gain [1/V]

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
#  MASTER REFERENCE TRAJECTORY                                       #
# ================================================================== #
# Kept for compatibility and side-by-side comparison with the archived
# reference-tracking setup.
REF_POS_AMP   = 0.06     # Reference position amplitude [m]
REF_POS_FREQ  = 0.5      # Reference frequency [Hz]
REF_POS_PHASE = 0.0      # Reference phase [rad]

# Legacy aliases kept for older scripts that used fh_* as a reference alias.
FH_AMP   = REF_POS_AMP
FH_FREQ  = REF_POS_FREQ
FH_PHASE = REF_POS_PHASE

# ================================================================== #
#  MASTER FORCE INPUT                                                #
# ================================================================== #
# Paper-faithful option: the master actuator is driven by an
# externally applied sinusoidal force F_h(t).
FORCE_INPUT_AMP   = 10.0   # Force amplitude [N]
FORCE_INPUT_FREQ  = 0.5    # Force frequency [Hz]
FORCE_INPUT_PHASE = 0.0    # Force phase [rad]

# Saved top-level Simulink reference input:
#   F_h(t) = 5 + 10*sin(0.5*t)
# The Simulink Sine Wave block is time-based, so its frequency parameter is
# interpreted directly in the block's native form; the Python env expects Hz.
MATLAB_REFERENCE_FORCE_BIAS = 5.0
MATLAB_REFERENCE_SINE_FREQ_RAD = 0.5
MATLAB_REFERENCE_FORCE_FREQ_HZ = MATLAB_REFERENCE_SINE_FREQ_RAD / (2.0 * np.pi)

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
#  MASTER INPUT MODES                                                #
# ================================================================== #
MASTER_INPUT_REFERENCE = "reference"
MASTER_INPUT_FORCE     = "force"
DEFAULT_MASTER_INPUT_MODE = MASTER_INPUT_FORCE

# ================================================================== #
#  ACTION SPACE                                                      #
# ================================================================== #
# Discrete valve-voltage actions over the range [-5, 5] V.
V_LEVELS  = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
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

# Added internal plant features for RL state.
FLOW_BINS = np.array([
    -0.0040, -0.0025, -0.0015, -0.0007,
     0.0000,
     0.0007,  0.0015,  0.0025,  0.0040
])

SPOOL_POS_BINS = np.array([
    -0.80, -0.50, -0.25, -0.10,
     0.10,  0.25,  0.50,  0.80
])

# Observation/discretization bins aligned with the environment observation:
# [slave_pos_err, master_pos_err, P_s1, P_s2, P_m1, P_m2, mdot_L1, mdot_L2]
SLAVE_POS_ERROR_BINS = POS_ERROR_BINS.copy()
MASTER_POS_ERROR_BINS = POS_ERROR_BINS.copy()
PRESSURE_BINS = np.array([
    50_000.0, 75_000.0, 100_000.0, 125_000.0,
    150_000.0, 200_000.0, 300_000.0, 450_000.0, 600_000.0
], dtype=np.float64)
SLAVE_P1_BINS = PRESSURE_BINS.copy()
SLAVE_P2_BINS = PRESSURE_BINS.copy()
MASTER_P1_BINS = PRESSURE_BINS.copy()
MASTER_P2_BINS = PRESSURE_BINS.copy()
MASS_FLOW1_BINS = FLOW_BINS.copy()
MASS_FLOW2_BINS = FLOW_BINS.copy()

# ================================================================== #
#  STATE DISCRETISATION (reduced 4-D tabular Q-learning)             #
# ================================================================== #
# State: (tracking_error, velocity_error, slave_pdiff, master_pdiff)
REDUCED_TRACKING_ERROR_BINS = np.array([
    -0.06, -0.04, -0.025, -0.015, -0.008, -0.003,
     0.003,  0.008,  0.015,  0.025,  0.04,  0.06
])
REDUCED_VELOCITY_ERROR_BINS = np.array([
    -0.40, -0.25, -0.15, -0.08, -0.03,
     0.03,  0.08,  0.15,  0.25,  0.40
])
REDUCED_SLAVE_PRESSURE_DIFF_BINS = np.array([
    -200_000, -120_000, -60_000, -25_000, -8_000,
       8_000,   25_000,  60_000, 120_000, 200_000
], dtype=np.float64)
REDUCED_MASTER_PRESSURE_DIFF_BINS = np.array([
    -200_000, -120_000, -60_000, -25_000, -8_000,
       8_000,   25_000,  60_000, 120_000, 200_000
], dtype=np.float64)

# ================================================================== #
#  OBSERVATION NORMALIZATION (for DQN / neural-net agents)           #
# ================================================================== #
# Each feature is divided by its scale so the observation lives in
# approximately [-1, 1]. For the active F_h setup, velocity scaling uses
# the larger of the reference-based estimate and the damping-limited
# master-force estimate.
OBS_SCALE_POS      = L_CYL / 2.0              # ~0.1375 m  (half-stroke)
REF_VEL_SCALE      = 2 * np.pi * REF_POS_FREQ * REF_POS_AMP
FORCE_VEL_SCALE    = FORCE_INPUT_AMP / max(BETA, 1e-9)
OBS_SCALE_VEL      = max(REF_VEL_SCALE, FORCE_VEL_SCALE)
OBS_SCALE_PRESSURE = P_SUPPLY                  # 600 kPa
OBS_SCALE_FLOW     = 0.004                     # kg/s  (matches bin extremes)

# ================================================================== #
#  TERMINATION + REWARD SCALING                                      #
# ================================================================== #
# For the active F_h experiments, use geometry-based limits instead of
# reference-trajectory amplitude.
POS_ERROR_FAIL_THRESHOLD = 0.75 * L_CYL
MAX_POSITION_ERROR = 0.50 * L_CYL
POS_ERR_NORM_CLIP = 1.0

# Power normalization heuristics for reward scaling.
V_MAX_GEOM = L_CYL / RL_DT
F_E_MAX_THEORETICAL = SKIN_KE * (L_CYL / 2.0) + SKIN_BE * V_MAX_GEOM
# Conservative reflected-force estimate from Eq. (2): pressure + inertia + damping.
F_H_SCALE_EST = max(
    FORCE_INPUT_AMP,
    F_E_MAX_THEORETICAL + MP * (2 * np.pi * REF_POS_FREQ) ** 2 * REF_POS_AMP + BETA * V_MAX_GEOM,
)
MAX_POWER_ERROR_THEORETICAL = V_MAX_GEOM * (F_E_MAX_THEORETICAL + F_H_SCALE_EST)
# Practical scale: based on observed range (~17 W peak with random actions).
MAX_POWER_ERROR = 20.0

# Reward weights
ALPHA_TRACKING = 40.0
GAMMA_EFFORT   = 0.01
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
MRAC_G1_GAIN  = 1.0

# ================================================================== #
#  PAPER REPLICA PROFILE (IC2AI 2025)                               #
# ================================================================== #
PAPER_MASTER_INPUT_MODE = MASTER_INPUT_FORCE
PAPER_EPISODE_DURATION = 60.0
PAPER_ENV_SWITCH_TIME  = 30.0
# Paper states a sinusoidal force input of amplitude 10 N is applied
# to the master actuator.
PAPER_FORCE_AMP        = 10.0
PAPER_FORCE_FREQ       = 0.5
PAPER_FORCE_PHASE      = 0.0
PAPER_RESULTS_DIR      = "paper_replica"

# ================================================================== #
#  Q-LEARNING HYPER-PARAMETERS                                       #
# ================================================================== #
LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.05
NUM_EPISODES    = 10_000
_EPSILON_HORIZON = 8_000
EPSILON_DECAY   = (EPSILON_END / EPSILON_START) ** (1.0 / _EPSILON_HORIZON)
EVAL_EVERY      = 100
EVAL_EPISODES   = 10

# ================================================================== #
#  DQN HYPER-PARAMETERS                                               #
# ================================================================== #
DQN_LEARNING_RATE          = 3e-4
DQN_DISCOUNT_FACTOR        = 0.99
DQN_REPLAY_BUFFER_SIZE     = 500_000
DQN_BATCH_SIZE             = 128
DQN_TARGET_UPDATE_FREQ     = 5_000     # gradient steps between target-net syncs
DQN_HIDDEN_SIZES           = (256, 256)
DQN_EPSILON_START          = 1.0
DQN_EPSILON_END            = 0.05
DQN_NUM_EPISODES           = 20_000
DQN_EPSILON_DECAY_EPISODES = 15_000
DQN_EVAL_EVERY             = 500
DQN_EVAL_EPISODES          = 10
DQN_MIN_REPLAY_SIZE        = 20_000
DQN_GRAD_CLIP              = 1.0

# ================================================================== #
#  RESULT FOLDER LAYOUT                                              #
# ================================================================== #
RESULTS_ROOT_DIR        = os.environ.get("TELEOP_RESULTS_ROOT_DIR", "results_fh")
DQN_CONSTANT_DIR        = "dqn_constant"
DQN_CHANGING_DIR        = "dqn_changing"
Q_LEARNING_CONSTANT_DIR = "q_learning_constant"
Q_LEARNING_CHANGING_DIR = "q_learning_changing"
MRAC_RESULTS_DIR        = "mrac"
COMPARE_RESULTS_DIR     = "comparisons"
