# Configuration

`config.py` is the shared source of constants for the replica, including plant
parameters, RL timing, action levels, observation scales, reward weights, and
tabular discretization bins.

Import it as a module so the names remain easy to identify:

```python
from matlab_env_python_replica.config import config as cfg
```

Algorithm modules and notebook runners should read configuration from this
package rather than maintaining a second copy of the constants.
