Derivatives Analytics
=====================

``ferro-ta`` includes a Rust-backed derivatives layer for analytics, research,
and simulation workflows. The implementation is analytics-only: there is no
broker connectivity, order routing, or execution engine in this package.

What Is Included
----------------

Options analytics
~~~~~~~~~~~~~~~~~

- Rolling IV helpers: ``iv_rank``, ``iv_percentile``, ``iv_zscore``
- Black-Scholes-Merton pricing
- Black-76 pricing
- Greeks: delta, gamma, vega, theta, rho
- Implied volatility inversion
- Smile metrics: ATM IV, 25-delta risk reversal, butterfly, skew slope, convexity
- Chain helpers: moneyness labels and strike selection by offset or delta

Futures analytics
~~~~~~~~~~~~~~~~~

- Synthetic forwards and parity diagnostics
- Basis, annualized basis, implied carry, carry spread
- Continuous contract stitching: weighted, back-adjusted, ratio-adjusted
- Curve analytics: calendar spreads, slope, contango/backwardation summary

Strategy and payoff helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Typed strategy schemas for expiry selectors, strike selectors, leg presets,
  risk controls, and simulation limits
- Multi-leg payoff aggregation
- Greeks aggregation across option and futures legs

Conventions
-----------

- ``model="bsm"`` expects spot as the underlying input.
- ``model="black76"`` expects forward as the underlying input.
- Volatility uses decimal annualized units: ``0.20`` means 20%.
- Rates and carry use decimal annualized units: ``0.05`` means 5%.
- ``time_to_expiry`` is expressed in years.

Options Example
---------------

.. code-block:: python

   from ferro_ta.analysis.options import greeks, implied_volatility, option_price

   price = option_price(
       100.0,
       100.0,
       0.05,
       1.0,
       0.20,
       option_type="call",
       model="bsm",
   )
   iv = implied_volatility(
       price,
       100.0,
       100.0,
       0.05,
       1.0,
       option_type="call",
       model="bsm",
   )
   g = greeks(
       100.0,
       100.0,
       0.05,
       1.0,
       0.20,
       option_type="call",
       model="bsm",
   )

Futures Example
---------------

.. code-block:: python

   from ferro_ta.analysis.futures import basis, curve_summary, synthetic_forward

   front_basis = basis(100.0, 103.0)
   synthetic = synthetic_forward(8.0, 5.0, 100.0, 0.02, 0.5)
   curve = curve_summary(100.0, [0.1, 0.5, 1.0], [101.0, 102.0, 104.0])

Strategy and Payoff Example
---------------------------

.. code-block:: python

   from ferro_ta.analysis.derivatives_payoff import PayoffLeg, aggregate_greeks, strategy_payoff

   legs = [
       PayoffLeg(
           instrument="option",
           side="long",
           option_type="call",
           strike=100.0,
           premium=5.0,
           volatility=0.20,
           time_to_expiry=0.5,
       ),
       PayoffLeg(
           instrument="future",
           side="long",
           entry_price=100.0,
       ),
   ]

   payoff = strategy_payoff([90.0, 100.0, 110.0], legs=legs)
   portfolio_greeks = aggregate_greeks(100.0, legs=legs)

Related Modules
---------------

- :mod:`ferro_ta.analysis.options`
- :mod:`ferro_ta.analysis.futures`
- :mod:`ferro_ta.analysis.options_strategy`
- :mod:`ferro_ta.analysis.derivatives_payoff`
