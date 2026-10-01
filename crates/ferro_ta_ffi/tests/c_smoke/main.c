/*
 * C smoke test: compiles against the generated ferro_ta.h with strict
 * warnings and links the static library, proving the header is usable from
 * plain C. Run via `make c-smoke` (CI runs it on Linux, macOS and Windows).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ferro_ta.h"

static int failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,     \
                    __LINE__, #cond);                                  \
            failures++;                                                \
        }                                                              \
    } while (0)

int main(void) {
    const double close[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const size_t n = sizeof close / sizeof close[0];
    double out[5];

    CHECK(strcmp(ft_version(), FT_VERSION) == 0);

    CHECK(ft_sma(close, n, 3, out) == FT_OK);
    CHECK(isnan(out[0]) && isnan(out[1]));
    CHECK(out[2] == 2.0 && out[3] == 3.0 && out[4] == 4.0);

    CHECK(ft_sma(close, n, 0, out) == FT_ERR_INVALID_PARAM);
    CHECK(strcmp(ft_status_message(FT_ERR_INVALID_PARAM), "invalid parameter") == 0);
    CHECK(ft_sma(NULL, n, 3, out) == FT_ERR_NULL_PTR);
    CHECK(ft_sma(NULL, 0, 3, NULL) == FT_OK);

    double up[5], mid[5], lo[5];
    CHECK(ft_bbands(close, n, 3, 2.0, 2.0, 0, up, mid, lo) == FT_OK);
    CHECK(mid[4] == 4.0 && up[4] > mid[4] && lo[4] < mid[4]);

    const double open[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double high[] = {1.5, 2.5, 3.5, 4.5, 5.5};
    const double low[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    int32_t pattern[5];
    CHECK(ft_cdldoji(open, high, low, close, n, pattern) == FT_OK);

    FtStreamSma *stream = NULL;
    CHECK(ft_stream_sma_new(3, &stream) == FT_OK && stream != NULL);
    double v = 0.0;
    for (size_t i = 0; i < n; i++) {
        CHECK(ft_stream_sma_update(stream, close[i], &v) == FT_OK);
    }
    CHECK(v == 4.0);
    CHECK(ft_stream_sma_reset(stream) == FT_OK);
    ft_stream_sma_free(stream);
    ft_stream_sma_free(NULL);

    FtStreamMacd *bad = NULL;
    CHECK(ft_stream_macd_new(26, 12, 9, &bad) == FT_ERR_INVALID_PARAM && bad == NULL);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("c smoke test passed (ferro_ta %s)\n", ft_version());
    return 0;
}
