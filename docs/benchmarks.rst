Benchmarks
==========

The authoritative benchmark workflow is in ``benchmarks/``:

- Cross-library speed suite: ``benchmarks/test_speed.py``
- Cross-library accuracy suite: ``benchmarks/test_accuracy.py``
- TA-Lib head-to-head speed script: ``benchmarks/bench_vs_talib.py``
- Table generation from benchmark JSON: ``benchmarks/benchmark_table.py``

Run the cross-library speed suite on 100,000 bars:

.. code-block:: bash

   uv run pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v

Selected results on a modern CPU (100,000 bars):

.. list-table::
   :header-rows: 1

   * - Indicator
     - Throughput
   * - ``ADD``
     - 1.9 G bars/s
   * - ``CDLENGULFING``
     - 454 M bars/s
   * - ``EMA``
     - 444 M bars/s
   * - ``SMA``
     - 259 M bars/s
   * - ``RSI``
     - 145 M bars/s
   * - ``ATR``
     - 70 M bars/s
   * - ``MACD``
     - 104 M bars/s
   * - ``STOCH``
     - 33 M bars/s

Multi-size and JSON output
--------------------------

To build the markdown comparison table from the JSON output:

.. code-block:: bash

   uv run python benchmarks/benchmark_table.py

Comparison with TA-Lib
----------------------

To measure speedup vs TA-Lib on the same data and parameters, run:

.. code-block:: bash

   pip install ta-lib
   python benchmarks/bench_vs_talib.py --sizes 10000 100000 --json benchmark_vs_talib.json

See the README “Performance vs TA-Lib” section for methodology and a
representative comparison table. The script prints a table of median times and
speedup (TA-Lib time / ferro_ta time); use ``--json out.json`` to save results.
