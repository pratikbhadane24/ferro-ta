package ferrota

import (
	"errors"
	"math"
	"testing"
)

func condorLeg(qty float64, right OptionKind, strike, premium float64) Leg {
	return Leg{Kind: LegOption, Right: right, Quantity: qty, Multiplier: 1, Strike: strike,
		Premium: premium, ExpiryYears: 0.25, Volatility: 0.2}
}

var ironCondor = []Leg{
	condorLeg(1, OptionPut, 80, 1),
	condorLeg(-1, OptionPut, 90, 3),
	condorLeg(-1, OptionCall, 110, 3),
	condorLeg(1, OptionCall, 120, 1),
}

func TestStrategyIronCondor(t *testing.T) {
	be, err := StrategyBreakevens(ironCondor)
	if err != nil || len(be) != 2 || be[0] != 86 || be[1] != 114 {
		t.Fatalf("breakevens %v %v", be, err)
	}
	mp, _ := StrategyMaxProfit(ironCondor)
	ml, _ := StrategyMaxLoss(ironCondor)
	if mp.Unlimited || mp.Value != 4 || ml.Unlimited || ml.Value != -6 {
		t.Fatalf("extremes %+v %+v", mp, ml)
	}
	pop, err := StrategyProbabilityOfProfit(ironCondor, 100, 0.2, 0.25, 0, 0)
	if err != nil || pop <= 0.5 || pop >= 1 {
		t.Fatalf("pop %v %v", pop, err)
	}
	grid, err := StrategyScenarioGrid(ironCondor, []float64{95, 105}, []float64{0, 0.1}, 0, 0, 0)
	if err != nil || len(grid) != 2 || len(grid[0]) != 2 || grid[1][0] >= grid[0][0] {
		t.Fatalf("grid %v %v", grid, err)
	}
	pnl, _ := StrategyPayoff(ironCondor, []float64{70, 100})
	if pnl[0] != -6 || pnl[1] != 4 {
		t.Fatalf("payoff %v", pnl)
	}
}

func TestStrategyUnlimitedAndErrors(t *testing.T) {
	naked := []Leg{condorLeg(-1, OptionCall, 100, 5)}
	ml, err := StrategyMaxLoss(naked)
	if err != nil || !ml.Unlimited || !math.IsNaN(ml.Value) {
		t.Fatalf("naked call loss %+v %v", ml, err)
	}
	bad := []Leg{condorLeg(0, OptionCall, 100, 5)}
	if _, err := StrategyBreakevens(bad); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("zero quantity: %v", err)
	}
	if _, err := StrategyBreakevens(nil); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("empty: %v", err)
	}
	if _, _, _, _, _, err := StrategyGreeks(ironCondor, 100, 0, 0); err != nil {
		t.Fatal(err)
	}
}
