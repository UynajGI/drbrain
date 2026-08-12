"""Static domain resources for DrBrain.

Word lists live here as data so that linguistic rules (concept normalization,
noise filtering, chemical-formula checks) stay in sync across callers instead of
being re-inlined per module. See :mod:`drbrain.resources.materials` for the
materials-science vocabulary.
"""

from __future__ import annotations
