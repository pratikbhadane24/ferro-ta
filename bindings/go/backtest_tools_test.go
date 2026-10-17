package ferrota

import (
	"errors"
	"math"
	"testing"
)

func TestCrossoverSignalsAndCloseBacktest(t *testing.T) {
	close_ := walk(200)
	sma, err := SmaCrossoverSignals(close_, 5, 20)
	if err != nil || !math.IsNaN(sma[0]) || math.Abs(sma[199]) != 1 {
		t.Fatalf("sma signals=%v err=%v", sma[195:], err)
	}
	if _, err := SmaCrossoverSignals(close_, 20, 5); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("fast>=slow: err=%v", err)
	}
	macd, err := MacdCrossoverSignals(close_, 6, 13, 4)
	if err != nil || math.Abs(macd[199]) != 1 {
		t.Fatalf("macd=%v err=%v", macd[195:], err)
	}
	settings, _ := BacktestDefaultSettings()
	settings[BtSlippageBps] = 2
	pos, _, rets, eq, err := BacktestClose(close_, sma, settings)
	if err != nil || pos[0] != 0 || eq[0] != 1 {
		t.Fatalf("close backtest err=%v", err)
	}
	metrics, bench, err := PerformanceMetrics(rets, eq, rets, pos, 252, 0)
	if err != nil || math.IsNaN(metrics[MetricSharpe]) || math.Abs(bench[BenchBeta]-1) > 1e-9 {
		t.Fatalf("self-benchmark beta=%v err=%v", bench[BenchBeta], err)
	}
}

func TestMonteCarloAndWalkForward(t *testing.T) {
	r := lcgNoise(100, 3)
	a, err := MonteCarloBootstrap(r, 4, 7, 5)
	b, _ := MonteCarloBootstrap(r, 4, 7, 5)
	if err != nil || len(a) != 400 {
		t.Fatalf("len=%d err=%v", len(a), err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed must give identical paths")
		}
	}
	folds, err := WalkForwardIndices(100, 40, 20, 0, 0)
	if err != nil || len(folds) != 12 || folds[4] != 20 || folds[7] != 80 {
		t.Fatalf("folds=%v err=%v", folds, err)
	}
	if _, err := WalkForwardIndices(100, 40, 20, 2, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("anchored=2: err=%v", err)
	}
}

func TestExtractTradesFromOhlcvRun(t *testing.T) {
	open, high, low, close_ := ohlcBars(60)
	signals := make([]float64, len(close_))
	for i := range signals {
		signals[i] = 1
		if (i/8)%2 == 1 {
			signals[i] = -1
		}
	}
	settings, _ := BacktestDefaultSettings()
	pos, fills, _, _, _, err := BacktestOhlcv(open, high, low, close_, nil, signals, nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	n, entry, exit, dir, _, _, pnl, dur, _, _, err := ExtractTrades(pos, fills, high, low)
	if err != nil || n < 3 {
		t.Fatalf("n=%d err=%v", n, err)
	}
	for k := 0; k < int(n); k++ {
		if exit[k] < entry[k] || dur[k] < 1 || math.Abs(dir[k]) != 1 || math.IsNaN(pnl[k]) {
			t.Fatalf("bad trade %d: entry=%d exit=%d dur=%d dir=%v", k, entry[k], exit[k], dur[k], dir[k])
		}
	}
	if entry[n] != -1 || !math.IsNaN(pnl[n]) {
		t.Fatal("rows past n_trades must be padded")
	}
}

func TestBacktestMultiAsset(t *testing.T) {
	const nBars, nAssets = 50, 2
	a, b := walk(nBars), walk(nBars + 7)[7:]
	close_ := make([]float64, nBars*nAssets)
	weights := make([]float64, nBars*nAssets)
	for i := 0; i < nBars; i++ {
		close_[i*nAssets], close_[i*nAssets+1] = a[i], b[i]
		weights[i*nAssets], weights[i*nAssets+1] = 0.6, 0.4
	}
	assets, port, eq, err := BacktestMultiAsset(close_, weights, nAssets, 0, 0, 1, 0, 0)
	if err != nil || len(assets) != nBars*nAssets || len(port) != nBars || eq[0] != 1 {
		t.Fatalf("err=%v", err)
	}
	for i := 1; i < nBars; i++ {
		if math.Abs(port[i]-(assets[i*nAssets]+assets[i*nAssets+1])) > 1e-12 {
			t.Fatalf("bar %d: portfolio != sum of asset returns", i)
		}
	}
	if _, _, _, err := BacktestMultiAsset(close_[:5], weights[:5], nAssets, 0, 0, 1, 0, 0); !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("ragged matrix: err=%v", err)
	}
}

func TestStreamBacktest(t *testing.T) {
	s, err := NewStreamBacktest(0.001, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var trades int64
	var equity float64
	for i, c := range walk(40) {
		sig := 1.0
		if (i/10)%2 == 1 {
			sig = -1
		}
		_, _, eq, n, _, _, _, _, _, err := s.Update(c, sig)
		if err != nil {
			t.Fatal(err)
		}
		trades, equity = n, eq
	}
	if trades < 3 || equity <= 0 {
		t.Fatalf("trades=%d equity=%v", trades, equity)
	}
}

func TestMonteCarloAcceptsLargeSeeds(t *testing.T) {
	r := lcgNoise(30, 5)
	for _, seed := range []uint64{1_700_000_000, math.MaxUint64} {
		if _, err := MonteCarloBootstrap(r, 2, seed, 4); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
	}
}
