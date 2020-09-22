"""
This module implements the Needleman-Wunsch Algorithm.

The backtracing step is modified to start with maximum score and extend
to the top-left and down to bottom-right. This follows one path that
contains the maximal matching sequence.

This algorithm also implements an X-Drop termination condition.
"""

from .seq    import Sequence
from .score  import Scorer
from .matrix import DPMatrix