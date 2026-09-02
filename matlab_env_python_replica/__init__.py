"""Python replica of the SimuOriginal teleoperation environment.

The public Gymnasium environment lives in :mod:`.environment`. Training,
evaluation, and command-line entry points are grouped by algorithm under the
``dqn``, ``ql``, and ``policy_gradient`` packages.
"""

from .environment.simuoriginal_env import SimuOriginalReplicaEnv

__all__ = ["SimuOriginalReplicaEnv"]
