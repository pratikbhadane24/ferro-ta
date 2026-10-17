package ferrota

import (
	"errors"
	"math"
	"testing"
)

func lcgNoise(n int, seed uint64) []float64 {
	out := make([]float64, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1
		out[i] = (float64(s>>11)/float64(uint64(1)<<53) - 0.5) * 0.02
	}
	return out
}

func walk(n int) []float64 {
	p, out := 100.0, make([]float64, n)
	for i, r := range lcgNoise(n, 9) {
		p *= 1 + r
		out[i] = p
	}
	return out
}

func TestPsrHalfAtOwnSharpe(t *testing.T) {
	r := lcgNoise(500, 7)
	mean, sq := 0.0, 0.0
	for _, x := range r {
		mean += x
	}
	mean /= float64(len(r))
	for _, x := range r {
		sq += (x - mean) * (x - mean)
	}
	sr := mean / math.Sqrt(sq/float64(len(r)-1))
	psr, err := ProbabilisticSharpeRatio(r, sr)
	if err != nil || math.Abs(psr-0.5) > 1e-7 {
		t.Fatalf("psr=%v err=%v, want 0.5", psr, err)
	}
	if _, err := ProbabilisticSharpeRatio([]float64{1, 1, 1}, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("constant returns: err=%v, want ErrInvalidParam", err)
	}
}

func TestPboDominantStrategyNeverOverfits(t *testing.T) {
	const nBars, nStrat = 400, 8
	m := make([]float64, nBars*nStrat)
	for j := 0; j < nStrat; j++ {
		col := lcgNoise(nBars, uint64(100+j))
		for i := range col {
			m[i*nStrat+j] = col[i]
			if j == 5 {
				m[i*nStrat+j] += 0.01
			}
		}
	}
	pbo, logits, _, _, best, err := PboCscv(m, nStrat, 8)
	if err != nil {
		t.Fatal(err)
	}
	if pbo != 0 || len(logits) != 70 {
		t.Fatalf("pbo=%v combos=%d, want 0 and C(8,4)=70", pbo, len(logits))
	}
	for _, b := range best {
		if b != 5 {
			t.Fatalf("best index %d, want 5", b)
		}
	}
}

func TestPboShapeErrors(t *testing.T) {
	if _, _, _, _, _, err := PboCscv(make([]float64, 10), 3, 4); !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("ragged matrix: err=%v, want ErrLengthMismatch", err)
	}
	if _, _, _, _, _, err := PboCscv(make([]float64, 40), 2, 3); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("odd n_splits: err=%v, want ErrInvalidParam", err)
	}
	if _, _, _, _, _, err := PboCscv(make([]float64, 40), -2, 4); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("negative size: err=%v, want ErrInvalidParam", err)
	}
}

func TestSweepBuiltinMatchesSignalSweep(t *testing.T) {
	close_ := walk(300)
	// RSI (timeperiod, oversold, overbought); the middle combo is invalid.
	combos := []float64{14, 30, 70, 14, 80, 20, 7, 25, 75}
	valid, returns, metrics, err := SweepBuiltin(close_, StrategyRsiThreshold, combos, 0, 0, 1e5, 252, 0)
	if err != nil {
		t.Fatal(err)
	}
	if valid[0] != 1 || valid[1] != 0 || valid[2] != 1 {
		t.Fatalf("valid=%v, want [1 0 1]", valid)
	}
	cols := len(metrics) / len(valid)
	if n, err := SweepNMetrics(); err != nil || int(n) != cols {
		t.Fatalf("SweepNMetrics()=%d err=%v, want stride %d", n, err, cols)
	}
	if !math.IsNaN(metrics[1*cols+int(MetricSharpe)]) {
		t.Fatal("skipped combo must have NaN metrics")
	}
	// The same combos through RsiThresholdSignals + SweepSignals.
	n := len(close_)
	signals := make([]float64, n*2)
	for k, c := range [][3]float64{{14, 30, 70}, {7, 25, 75}} {
		s, err := RsiThresholdSignals(close_, int(c[0]), c[1], c[2])
		if err != nil {
			t.Fatal(err)
		}
		for i := range s {
			signals[i*2+k] = s[i]
		}
	}
	r2, m2, err := SweepSignals(close_, signals, 2, 0, 0, 1e5, 252, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if returns[i*3] != r2[i*2] || returns[i*3+2] != r2[i*2+1] {
			t.Fatalf("bar %d: builtin and signal sweeps differ", i)
		}
	}
	if metrics[2*cols+int(MetricSharpe)] != m2[1*cols+int(MetricSharpe)] {
		t.Fatal("sharpe differs between builtin and signal sweeps")
	}
}

func TestSweepBuiltinRejectsWrongParamCount(t *testing.T) {
	_, _, _, err := SweepBuiltin(walk(50), StrategySmaCrossover, []float64{5, 20, 1}, 0, 0, 1e5, 252, 0)
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("err=%v, want ErrLengthMismatch (combos not a multiple of 2 params)", err)
	}
}

func TestPlateauAndKfold(t *testing.T) {
	scores, err := SweepPlateauScores([]float64{10, -5, 3, 3, 3, 0}, []int64{0, 1, 2, 3, 4, 5}, 1, 1)
	if err != nil || scores[0] != 2.5 || scores[3] != 3 {
		t.Fatalf("scores=%v err=%v", scores, err)
	}
	folds, err := PurgedKfoldIndices(100, 4, 3, 2)
	if err != nil || len(folds) != 16 || folds[4] != 22 || folds[7] != 52 {
		t.Fatalf("folds=%v err=%v", folds, err)
	}
	if n, err := PboNCombinations(16); err != nil || n != 12870 {
		t.Fatalf("PboNCombinations(16)=%d err=%v", n, err)
	}
}

func TestKernelWrappersRejectAbsurdSizesWithoutPanicking(t *testing.T) {
	for _, nFolds := range []int{-1, 1, 1 << 40} {
		if _, err := PurgedKfoldIndices(100, nFolds, 0, 0); !errors.Is(err, ErrInvalidParam) {
			t.Fatalf("nFolds=%d: err=%v, want ErrInvalidParam", nFolds, err)
		}
	}
	if _, _, err := SweepSignals(nil, nil, 1<<62, 0, 0, 1e5, 252, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("huge nCombos: err=%v, want ErrInvalidParam", err)
	}
	if _, err := SweepPlateauScores([]float64{1}, []int64{0}, -1, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("negative nAxes: err=%v, want ErrInvalidParam", err)
	}
}
