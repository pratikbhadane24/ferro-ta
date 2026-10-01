package ferrota

import (
	"errors"
	"math"
	"testing"
)

func TestOptionKindChangesPrice(t *testing.T) {
	call, err := BlackScholesPrice(100, 100, 0.03, 0, 1, 0.2, OptionCall)
	if err != nil {
		t.Fatal(err)
	}
	put, err := BlackScholesPrice(100, 100, 0.03, 0, 1, 0.2, OptionPut)
	if err != nil {
		t.Fatal(err)
	}
	// Put-call parity: C - P = S - K e^{-rT}.
	if parity := call - put - (100 - 100*math.Exp(-0.03)); math.Abs(parity) > 1e-9 {
		t.Fatalf("parity residual %v", parity)
	}
}

func TestInvalidEnumValue(t *testing.T) {
	if _, err := BlackScholesPrice(100, 100, 0, 0, 1, 0.2, OptionKind(5)); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
}

func TestImpliedVolatilityRoundTrip(t *testing.T) {
	price, err := BlackScholesPrice(100, 110, 0.02, 0.02, 0.75, 0.3, OptionPut)
	if err != nil {
		t.Fatal(err)
	}
	iv, err := ImpliedVolatility(price, ModelBlackScholes, 100, 110, 0.02, 0.02, 0.75, OptionPut, 0.2, 1e-10, 200)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(iv-0.3) > 1e-6 {
		t.Fatalf("iv = %v, want 0.3", iv)
	}
}

func TestKellyRejectsInvalidWinRate(t *testing.T) {
	if _, err := KellyFraction(1.5, 2, 1); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v, want ErrInvalidParam", err)
	}
	f, err := HalfKellyFraction(0.6, 2, 1)
	if err != nil || f <= 0 {
		t.Fatalf("got %v, %v", f, err)
	}
}
