"""
Backtest option layer — synthesize a deep-ITM contract and reprice it over its life,
built on the shared Black-Scholes core (agent.pricing). Because the strategy buys DEEP
in-the-money (price ~ intrinsic), this reconstructs option value from the underlying alone
— no historical option-price data needed. See validate.py for the accuracy check.
"""
from backtest.pricing.synth import (
    SynthContract, hv_sigma, select_contract, reprice, pick_expiration,
)

__all__ = ["SynthContract", "hv_sigma", "select_contract", "reprice", "pick_expiration"]
