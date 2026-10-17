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

    /* Kernels: derived size, row-major matrices, enum-indexed metrics. */
    int64_t n_comb = 0;
    CHECK(ft_pbo_n_combinations(4, &n_comb) == FT_OK && n_comb == 6);
    CHECK(ft_pbo_n_combinations(5, &n_comb) == FT_ERR_INVALID_PARAM);

    double prices[60];
    for (size_t i = 0; i < 60; i++) {
        prices[i] = 100.0 + 5.0 * sin((double)i * 0.3) + (double)i * 0.05;
    }
    const double combos[] = {3.0, 10.0, 12.0, 6.0}; /* (fast, slow); 2nd invalid */
    int64_t n_params = 0;
    CHECK(ft_sweep_strategy_n_params(FT_STRATEGY_SMA_CROSSOVER, &n_params) == FT_OK);
    CHECK(n_params == 2);
    int64_t n_metrics = 0;
    CHECK(ft_sweep_n_metrics(&n_metrics) == FT_OK);
    CHECK(n_metrics == FT_METRIC_N_POSITION_CHANGES + 1);
    int8_t valid[2];
    double sweep_returns[60 * 2];
    double sweep_metrics[2 * (FT_METRIC_N_POSITION_CHANGES + 1)];
    CHECK(ft_sweep_builtin(prices, 60, FT_STRATEGY_SMA_CROSSOVER, combos, 2, 2,
                           0.0, 0.0, 100000.0, 252.0, 0.0, (size_t)n_metrics,
                           valid, sweep_returns, sweep_metrics) == FT_OK);
    CHECK(valid[0] == 1 && valid[1] == 0);
    CHECK(!isnan(sweep_metrics[FT_METRIC_SHARPE]));
    CHECK(isnan(sweep_returns[59 * 2 + 1]));

    int64_t folds[3 * 4];
    CHECK(ft_purged_kfold_indices(30, 3, 1, 1, folds) == FT_OK);
    CHECK(folds[4] == 9 && folds[5] == 10 && folds[6] == 20 && folds[7] == 21);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("c smoke test passed (ferro_ta %s)\n", ft_version());
    return 0;
}
