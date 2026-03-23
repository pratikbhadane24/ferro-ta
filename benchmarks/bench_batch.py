import time
import numpy as np
import ferro_ta

def _time_fn(fn, *args, **kwargs):
    times = []
    # Warmup
    fn(*args, **kwargs)
    for _ in range(5):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)

def main():
    n_samples = 100_000
    n_series = 100
    print(f"Batch Benchmark: {n_samples} bars, {n_series} series (Total: {n_samples*n_series/1e6:.1f} M bars)")
    
    np.random.seed(42)
    # contiguous array in row-major
    close2d = np.random.uniform(100.0, 200.0, (n_samples, n_series))
    h2d = close2d + np.random.uniform(0.1, 2.0, (n_samples, n_series))
    l2d = close2d - np.random.uniform(0.1, 2.0, (n_samples, n_series))
    
    print("-" * 50)
    print(f"{'Indicator':<15} {'Batch (ms)':>12} {'Loop (ms)':>12} {'Speedup':>10}")
    print("-" * 50)

    # 1. SMA
    kwargs = {"timeperiod": 14}
    def loop_sma(arr):
        for j in range(arr.shape[1]):
            ferro_ta.SMA(arr[:, j], **kwargs)
            
    t_batch_sma = _time_fn(ferro_ta.batch.batch_sma, close2d, **kwargs)
    t_loop_sma = _time_fn(loop_sma, close2d)
    print(f"SMA             {t_batch_sma*1000:12.1f} {t_loop_sma*1000:12.1f} {t_loop_sma/t_batch_sma:9.1f}x")

    # 2. RSI
    def loop_rsi(arr):
        for j in range(arr.shape[1]):
            ferro_ta.RSI(arr[:, j], **kwargs)
    t_batch_rsi = _time_fn(ferro_ta.batch.batch_rsi, close2d, **kwargs)
    t_loop_rsi = _time_fn(loop_rsi, close2d)
    print(f"RSI             {t_batch_rsi*1000:12.1f} {t_loop_rsi*1000:12.1f} {t_loop_rsi/t_batch_rsi:9.1f}x")

    # 3. ATR
    def loop_atr(h, l, c):
        for j in range(h.shape[1]):
            ferro_ta.ATR(h[:, j], l[:, j], c[:, j], **kwargs)
    t_batch_atr = _time_fn(ferro_ta.batch.batch_atr, h2d, l2d, close2d, **kwargs)
    t_loop_atr = _time_fn(loop_atr, h2d, l2d, close2d)
    print(f"ATR             {t_batch_atr*1000:12.1f} {t_loop_atr*1000:12.1f} {t_loop_atr/t_batch_atr:9.1f}x")

    # 4. ADX
    def loop_adx(h, l, c):
        for j in range(h.shape[1]):
            ferro_ta.ADX(h[:, j], l[:, j], c[:, j], **kwargs)
    t_batch_adx = _time_fn(ferro_ta.batch.batch_adx, h2d, l2d, close2d, **kwargs)
    t_loop_adx = _time_fn(loop_adx, h2d, l2d, close2d)
    print(f"ADX             {t_batch_adx*1000:12.1f} {t_loop_adx*1000:12.1f} {t_loop_adx/t_batch_adx:9.1f}x")

if __name__ == '__main__':
    main()
