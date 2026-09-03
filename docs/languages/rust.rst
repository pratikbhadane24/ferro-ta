Rust
====

``ferro_ta_core`` is the compute engine. Use it directly from any Rust project
— no PyO3, NumPy, or JavaScript runtime.

Installation
------------

.. code-block:: toml

   [dependencies]
   ferro_ta_core = "1.2.0"

Or:

.. code-block:: bash

   cargo add ferro_ta_core

Docs.rs: https://docs.rs/ferro_ta_core

Basic Usage
-----------

Functions take borrowed ``&[f64]`` slices and return owned ``Vec<f64>`` (or
tuples of vectors). Leading warmup bars are ``NaN``.

.. code-block:: rust

   use ferro_ta_core::{momentum, overlap};

   fn main() {
       let close: Vec<f64> = (0..40).map(|i| 44.0 + i as f64 * 0.1).collect();

       let sma = overlap::sma(&close, 5);
       let rsi = momentum::rsi(&close, 14);
       let (macd, signal, hist) = overlap::macd(&close, 12, 26, 9);
       let (upper, middle, lower) = overlap::bbands(&close, 5, 2.0, 2.0);

       assert!(sma[0].is_nan());
       assert!(rsi[13].is_nan());
   }

API map
-------

Module layout matches the crate README. Names are ``snake_case``.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Module
     - Count
     - Highlights
   * - ``overlap``
     - 20
     - ``sma``, ``ema``, ``bbands``, ``macd``, ``macdfix``, ``macdext``, ``sar``
   * - ``momentum``
     - 26
     - ``rsi``, ``stoch``, ``adx``, ``plus_di``, ``minus_di``, ``trix``
   * - ``volatility``
     - 3
     - ``atr``, ``natr``, ``trange``
   * - ``volume``
     - 4
     - ``obv``, ``mfi``, ``ad``, ``adosc``
   * - ``pattern``
     - 61
     - ``cdldoji`` … ``cdlxsidegap3methods``
   * - ``statistic``
     - 9
     - ``stddev``, ``var``, ``linearreg``, ``beta``, ``correl``
   * - ``math`` / ``math_ops``
     - 24+
     - rolling ``sum``/``max``/``min``, element-wise ``add``/``sin``/…
   * - ``price_transform``
     - 4
     - ``avgprice``, ``medprice``, ``typprice``, ``wclprice``
   * - ``cycle``
     - 7
     - Hilbert ``ht_trendline``, ``ht_dcperiod``, …
   * - ``extended``
     - 10
     - VWAP, Supertrend, Ichimoku, Donchian, …
   * - ``streaming``
     - 9
     - Stateful ``StreamingSMA``, ``StreamingRSI``, …
   * - ``batch``
     - 8
     - Multi-column ``batch_sma`` / ``batch_ema`` / …

Other modules (``backtest``, ``options``, ``futures``, ``portfolio``, …) are
available from the same crate. See
`crates/ferro_ta_core/README.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/crates/ferro_ta_core/README.md>`_
and :doc:`coverage`.
