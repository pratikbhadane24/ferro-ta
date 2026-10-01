C / C++ (C ABI)
===============

``crates/ferro_ta_ffi`` is a stable C ABI over ``ferro_ta_core``. It is the
foundation for the Go module and for future C-ABI bindings (C#, JVM, Ruby,
...), and it is usable directly from C and C++.

Getting the library
-------------------

Each GitHub Release has one archive per platform,
``ferro_ta-<version>-<platform>.tar.gz``, containing ``include/ferro_ta.h``,
the static library and the shared library (``.so`` / ``.dylib``; on Windows an
MSVC ``.dll`` with its import ``.lib``).

To build from source:

.. code-block:: bash

   cargo build -p ferro_ta_ffi --release
   # header:  crates/ferro_ta_ffi/include/ferro_ta.h
   # library: target/release/libferro_ta_ffi.{a,so,dylib}

Example
-------

.. code-block:: c

   #include <stdio.h>
   #include "ferro_ta.h"

   int main(void) {
       const double close[] = {1, 2, 3, 4, 5};
       double out[5];
       int32_t status = ft_sma(close, 5, 3, out);
       if (status != FT_OK) {
           fprintf(stderr, "ft_sma: %s\n", ft_status_message(status));
           return 1;
       }
       printf("%f\n", out[4]); /* 4.0 */
       return 0;
   }

.. code-block:: bash

   cc -std=c99 -Iinclude example.c lib/libferro_ta_ffi.a -lm   # add -lpthread -ldl on Linux

ABI conventions
---------------

- Every function returns ``int32_t``: ``FT_OK`` (0) or an ``FT_ERR_*`` code;
  ``ft_status_message`` describes it.
- Array inputs are ``const double *`` sharing one ``size_t len``; outputs are
  caller-allocated arrays of ``len`` elements, written only on success. An
  output may alias an input. Any array may be ``NULL`` when ``len == 0``.
- Periods are ``int64_t`` in ``[minimum, 2^24]`` (negative values are
  rejected, not wrapped; the cap keeps absurd values from exhausting memory),
  MA types ``int32_t`` in 0–8, enums ``int32_t`` (``FT_OPTION_CALL``, ...),
  other scalars ``double`` (must be finite).
- Scalar functions write each result through an output pointer
  (``ft_black_scholes_greeks`` → ``out_delta``, ``out_gamma``, ...).
- Streaming indicators are opaque handles: ``ft_stream_<x>_new`` /
  ``_update`` / ``_reset`` / ``_free``. A handle is not thread-safe; free it
  exactly once (``free(NULL)`` is a no-op).
- A panic inside the core is caught and returned as ``FT_ERR_PANIC``; it never
  unwinds into the caller.

Every exported function is described in
``crates/ferro_ta_ffi/ffi_spec.json`` (inputs, parameter kinds and minimums,
outputs), which is what language wrappers are generated from.
