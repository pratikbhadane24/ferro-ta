package ferrota

import (
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
)

func ramp(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = float64(i + 1)
	}
	return s
}

func TestSmaKnownValues(t *testing.T) {
	got, err := Sma([]float64{1, 2, 3, 4, 5}, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.NaN(), math.NaN(), 2, 3, 4}
	for i := range want {
		if !closeEnough(got[i], want[i]) {
			t.Fatalf("Sma[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestInvalidPeriod(t *testing.T) {
	for _, period := range []int{0, -1, math.MinInt} {
		_, err := Sma(ramp(10), period)
		if !errors.Is(err, ErrInvalidParam) {
			t.Fatalf("period %d: got %v, want ErrInvalidParam", period, err)
		}
	}
}

func TestPeriodMinimumAboveOne(t *testing.T) {
	// Rwi requires timeperiod >= 2.
	h, l, c := ramp(20), ramp(20), ramp(20)
	if _, _, err := Rwi(h, l, c, 1); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
	if _, _, err := Rwi(h, l, c, 2); err != nil {
		t.Fatal(err)
	}
}

func TestInvalidMAType(t *testing.T) {
	if _, err := Ma(ramp(20), 5, MAType(9)); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
	if _, err := Ma(ramp(20), 5, MATypeT3); err != nil {
		t.Fatal(err)
	}
}

func TestNonFiniteFloatParam(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, _, _, err := Bbands(ramp(30), 5, bad, 2, MATypeSMA); !errors.Is(err, ErrInvalidParam) {
			t.Fatalf("nbdevup=%v: got %v, want ErrInvalidParam", bad, err)
		}
	}
}

func TestLengthMismatch(t *testing.T) {
	_, err := Atr(ramp(10), ramp(10), ramp(9), 3)
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("got %v, want ErrLengthMismatch", err)
	}
	if !strings.Contains(err.Error(), "close=9") {
		t.Fatalf("error should name the offending input: %v", err)
	}
}

func TestEmptyInput(t *testing.T) {
	out, err := Sma(nil, 3)
	if err != nil || len(out) != 0 {
		t.Fatalf("got %v, %v; want empty, nil", out, err)
	}
	if _, err := Sma(nil, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("empty input must still validate params, got %v", err)
	}
}

func TestErrorFormatting(t *testing.T) {
	_, err := Sma(ramp(5), 0)
	var e *Error
	if !errors.As(err, &e) || e.Op != "Sma" {
		t.Fatalf("got %#v", err)
	}
	if got := err.Error(); got != "ferrota: Sma: invalid parameter" {
		t.Fatalf("Error() = %q", got)
	}
	if errors.Is(err, ErrLengthMismatch) {
		t.Fatal("must not match a different sentinel")
	}
}

func TestInputNotModified(t *testing.T) {
	in := ramp(50)
	before := append([]float64(nil), in...)
	if _, err := Rsi(in, 14); err != nil {
		t.Fatal(err)
	}
	for i := range in {
		if in[i] != before[i] {
			t.Fatalf("input modified at %d", i)
		}
	}
}

func TestConcurrentCalls(t *testing.T) {
	in := ramp(500)
	want, err := Ema(in, 10)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < 50; k++ {
				got, err := Ema(in, 10)
				if err != nil {
					t.Error(err)
					return
				}
				compareSeries(t, "Ema", got, want)
			}
		}()
	}
	wg.Wait()
}

func TestVersion(t *testing.T) {
	if Version == "" || Version != nativeVersion() {
		t.Fatalf("Go Version %q != native %q", Version, nativeVersion())
	}
}

func TestCrossParamRule(t *testing.T) {
	s := ramp(64)
	if _, err := RollingVarianceBreak(s, 10, 10, 2); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("long_window == short_window: got %v, want ErrInvalidParam", err)
	}
	if _, err := RollingVarianceBreak(s, 5, 20, 2); err != nil {
		t.Fatal(err)
	}
}

func TestFastPeriodMustBeBelowSlow(t *testing.T) {
	// Previously returned all-NaN with a nil error, indistinguishable from warm-up.
	if _, _, _, err := Macd(ramp(64), 26, 12, 9); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
	if _, err := Apo(ramp(64), 12, 12, MATypeEMA); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("equal periods: got %v, want ErrInvalidParam", err)
	}
}

func TestOversizedPeriodIsRejected(t *testing.T) {
	if _, err := Sma(ramp(8), 1<<40); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
}
