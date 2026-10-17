package ferrota

import (
	"errors"
	"math"
	"testing"
)

func ohlcBars(n int) (open, high, low, close_ []float64) {
	close_ = walk(n)
	open = make([]float64, n)
	high = make([]float64, n)
	low = make([]float64, n)
	for i := range close_ {
		open[i] = close_[i]
		if i > 0 {
			open[i] = close_[i-1]
		}
		high[i] = math.Max(open[i], close_[i]) + 0.5
		low[i] = math.Min(open[i], close_[i]) - 0.5
	}
	return
}

func TestBacktestOhlcvDefaultsAndStops(t *testing.T) {
	open, high, low, close_ := ohlcBars(120)
	signals := make([]float64, len(close_))
	for i := range signals {
		signals[i] = 1
	}
	settings, err := BacktestDefaultSettings()
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := BacktestNSettings(); int(n) != len(settings) {
		t.Fatalf("BacktestNSettings=%d, defaults len %d", n, len(settings))
	}
	if settings[BtInitialCapital] != 100_000 || settings[BtLotSize] != 1 {
		t.Fatalf("unexpected defaults %v", settings)
	}
	pos, _, _, sr, eq, err := BacktestOhlcv(open, high, low, close_, nil, signals, nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	if pos[0] != 0 || pos[1] != 1 || eq[0] != 1 {
		t.Fatalf("pos=%v eq0=%v", pos[:3], eq[0])
	}
	metrics, bench, err := PerformanceMetrics(sr, eq, nil, pos, 252, 0)
	if err != nil || len(metrics) == 0 || math.IsNaN(metrics[MetricSharpe]) {
		t.Fatalf("metrics=%v err=%v", metrics, err)
	}
	if !math.IsNaN(bench[BenchBeta]) {
		t.Fatalf("no benchmark given, beta=%v want NaN", bench[BenchBeta])
	}

	// A tight stop must flatten at least once versus the always-long baseline.
	settings[BtStopLossPct] = 0.002
	posStop, _, _, _, _, err := BacktestOhlcv(open, high, low, close_, nil, signals, nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	flat := 0
	for _, p := range posStop[1:] {
		if p == 0 {
			flat++
		}
	}
	if flat == 0 {
		t.Fatal("stop loss never triggered")
	}
}

func TestBacktestOhlcvPartialFillsAndErrors(t *testing.T) {
	open, high, low, close_ := ohlcBars(40)
	signals := make([]float64, len(close_))
	volume := make([]float64, len(close_))
	for i := range signals {
		signals[i] = 1
		volume[i] = 300 // cap = 0.1*300*~100/100000 ≈ 0.03 weight per bar
	}
	settings, _ := BacktestDefaultSettings()
	settings[BtMaxParticipation] = 0.1
	if _, _, _, _, _, err := BacktestOhlcv(open, high, low, close_, nil, signals, nil, settings); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("participation without volume: err=%v, want ErrInvalidParam", err)
	}
	pos, _, _, _, _, err := BacktestOhlcv(open, high, low, close_, volume, signals, nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	if !(pos[1] > 0 && pos[1] < pos[5] && pos[5] < 1) {
		t.Fatalf("expected a gradual ramp, got %v", pos[:6])
	}
	if _, _, _, _, _, err := BacktestOhlcv(open, high, low, close_, volume[:3], signals, nil, settings); !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("short volume: err=%v, want ErrLengthMismatch", err)
	}
	settings[BtFillMode] = 7
	if _, _, _, _, _, err := BacktestOhlcv(open, high, low, close_, volume, signals, nil, settings); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("bad fill mode: err=%v, want ErrInvalidParam", err)
	}
}
