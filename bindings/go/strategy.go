package ferrota

/*
#include "ferro_ta.h"
*/
import "C"

import (
	"math"
	"unsafe"
)

// LegKind selects what a Leg holds.
type LegKind int32

// LegKind values.
const (
	LegOption LegKind = C.FT_LEG_OPTION
	LegFuture LegKind = C.FT_LEG_FUTURE
	LegStock  LegKind = C.FT_LEG_STOCK
)

// Leg is one position in an option strategy. Quantity is signed (+ long,
// - short). Option fields are ignored for futures/stock and EntryPrice for
// options.
type Leg struct {
	Kind        LegKind
	Right       OptionKind
	Quantity    float64
	Multiplier  float64
	Strike      float64
	Premium     float64
	ExpiryYears float64
	Volatility  float64
	EntryPrice  float64
}

// Extreme is a strategy's max profit or max loss at expiry. When Unlimited
// is true, Value and Spot are NaN.
type Extreme struct {
	Value     float64
	Spot      float64
	Unlimited bool
}

func cLegs(legs []Leg) (*C.FtLeg, C.size_t) {
	if len(legs) == 0 {
		return nil, 0
	}
	out := make([]C.FtLeg, len(legs))
	for i, l := range legs {
		out[i] = C.FtLeg{
			kind:         C.int32_t(l.Kind),
			right:        C.int32_t(l.Right),
			quantity:     C.double(l.Quantity),
			multiplier:   C.double(l.Multiplier),
			strike:       C.double(l.Strike),
			premium:      C.double(l.Premium),
			expiry_years: C.double(l.ExpiryYears),
			volatility:   C.double(l.Volatility),
			entry_price:  C.double(l.EntryPrice),
		}
	}
	return &out[0], C.size_t(len(out))
}

func cDoubles(xs []float64) *C.double {
	if len(xs) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&xs[0]))
}

// StrategyPayoff returns the expiry P&L at each spot. Option legs must share
// one expiry.
func StrategyPayoff(legs []Leg, spots []float64) ([]float64, error) {
	l, n := cLegs(legs)
	out := make([]float64, len(spots))
	rc := C.ft_strategy_payoff(l, n, cDoubles(spots), C.size_t(len(spots)), cDoubles(out))
	if err := statusError("StrategyPayoff", rc); err != nil {
		return nil, err
	}
	return out, nil
}

// StrategyBreakevens returns the spots where expiry P&L crosses zero,
// ascending.
func StrategyBreakevens(legs []Leg) ([]float64, error) {
	l, n := cLegs(legs)
	buf := make([]float64, 2*len(legs)+2)
	var count C.size_t
	rc := C.ft_strategy_breakevens(l, n, cDoubles(buf), C.size_t(len(buf)), &count)
	if err := statusError("StrategyBreakevens", rc); err != nil {
		return nil, err
	}
	return buf[:int(count)], nil
}

func strategyExtreme(op string, legs []Leg, max bool) (Extreme, error) {
	l, n := cLegs(legs)
	var value, spot C.double
	var unlimited C.int32_t
	var rc C.int32_t
	if max {
		rc = C.ft_strategy_max_profit(l, n, &value, &spot, &unlimited)
	} else {
		rc = C.ft_strategy_max_loss(l, n, &value, &spot, &unlimited)
	}
	if err := statusError(op, rc); err != nil {
		return Extreme{Value: math.NaN(), Spot: math.NaN()}, err
	}
	return Extreme{Value: float64(value), Spot: float64(spot), Unlimited: unlimited != 0}, nil
}

// StrategyMaxProfit returns the best expiry P&L over spot >= 0.
func StrategyMaxProfit(legs []Leg) (Extreme, error) {
	return strategyExtreme("StrategyMaxProfit", legs, true)
}

// StrategyMaxLoss returns the worst expiry P&L over spot >= 0 (usually
// negative).
func StrategyMaxLoss(legs []Leg) (Extreme, error) {
	return strategyExtreme("StrategyMaxLoss", legs, false)
}

// StrategyProbabilityOfProfit returns the risk-neutral probability that the
// expiry P&L is positive, with a lognormal underlying.
func StrategyProbabilityOfProfit(legs []Leg, spot, volatility, timeYears, rate, carry float64) (float64, error) {
	l, n := cLegs(legs)
	var out C.double
	rc := C.ft_strategy_probability_of_profit(l, n, C.double(spot), C.double(volatility),
		C.double(timeYears), C.double(rate), C.double(carry), &out)
	if err := statusError("StrategyProbabilityOfProfit", rc); err != nil {
		return math.NaN(), err
	}
	return float64(out), nil
}

// StrategyGreeks returns Black-Scholes greeks summed over legs.
func StrategyGreeks(legs []Leg, spot, rate, carry float64) (delta, gamma, vega, theta, rho float64, err error) {
	l, n := cLegs(legs)
	var d, g, v, t, r C.double
	rc := C.ft_strategy_greeks(l, n, C.double(spot), C.double(rate), C.double(carry), &d, &g, &v, &t, &r)
	if err = statusError("StrategyGreeks", rc); err != nil {
		return
	}
	return float64(d), float64(g), float64(v), float64(t), float64(r), nil
}

// StrategyScenarioGrid returns mark-to-model P&L as grid[shift][spot],
// daysForward days ahead with each option's volatility shifted.
func StrategyScenarioGrid(legs []Leg, spots, volShifts []float64, daysForward, rate, carry float64) ([][]float64, error) {
	l, n := cLegs(legs)
	flat := make([]float64, len(spots)*len(volShifts))
	rc := C.ft_strategy_scenario_grid(l, n, cDoubles(spots), C.size_t(len(spots)),
		cDoubles(volShifts), C.size_t(len(volShifts)), C.double(daysForward), C.double(rate),
		C.double(carry), cDoubles(flat))
	if err := statusError("StrategyScenarioGrid", rc); err != nil {
		return nil, err
	}
	grid := make([][]float64, len(volShifts))
	for i := range grid {
		grid[i] = flat[i*len(spots) : (i+1)*len(spots)]
	}
	return grid, nil
}
