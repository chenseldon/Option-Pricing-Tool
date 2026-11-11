"""Checks whose output is evidence, not a price.

Nothing in this package feeds a valuation. The futures-versus-parity forward
check is the clearest case: the moment its gap were used to imply a rate, the
independence that makes it evidence would be gone.
"""

from __future__ import annotations
