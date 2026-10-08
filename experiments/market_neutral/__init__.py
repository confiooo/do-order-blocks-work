"""Market-neutral study: relative-value pairs and funding/basis.

Independent of the order-block code: nothing here imports ``order_blocks``,
``strategies`` or ``family_a``.
"""

from experiments.market_neutral.calendar import UNIVERSE_BIAS_NOTE

__all__ = ["UNIVERSE_BIAS_NOTE"]
