package ferrota

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// goldenPath is the shared, language-neutral fixture produced by
// scripts/build_golden_fixtures.py. It lives outside this module, so the
// golden tests skip when the module is used from the Go module cache.
var goldenPath = filepath.Join("..", "..", "tests", "fixtures", "golden", "ffi_golden.json")

const goldenRelTol = 1e-9

type goldenStepper struct {
	step  func(bar map[string]float64) (map[string]any, error)
	close func() error
}

type goldenCase struct {
	Fn      string                       `json:"fn"`
	Stream  string                       `json:"stream"`
	Inputs  map[string]string            `json:"inputs"`
	Params  map[string]float64           `json:"params"`
	Outputs map[string][]json.RawMessage `json:"outputs"`
}

type goldenScalarCase struct {
	Fn      string                     `json:"fn"`
	Params  map[string]float64         `json:"params"`
	Outputs map[string]json.RawMessage `json:"outputs"`
}

type goldenFile struct {
	Columns   map[string][]json.RawMessage `json:"columns"`
	Functions []goldenCase                 `json:"functions"`
	Scalars   []goldenScalarCase           `json:"scalars"`
	Streams   []goldenCase                 `json:"streams"`
}

func loadGolden(t *testing.T) (goldenFile, map[string][]float64) {
	t.Helper()
	raw, err := os.ReadFile(goldenPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("golden fixtures not available outside the ferro-ta repository")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse %s: %v", goldenPath, err)
	}
	cols := make(map[string][]float64, len(g.Columns))
	for name, vals := range g.Columns {
		cols[name] = decodeFloats(t, vals)
	}
	return g, cols
}

// decodeFloats maps the fixture encoding back to float64: null is NaN and
// "inf"/"-inf" are infinities (JSON has neither).
func decodeFloats(t *testing.T, vals []json.RawMessage) []float64 {
	t.Helper()
	out := make([]float64, len(vals))
	for i, v := range vals {
		switch string(v) {
		case "null":
			out[i] = math.NaN()
		case `"inf"`:
			out[i] = math.Inf(1)
		case `"-inf"`:
			out[i] = math.Inf(-1)
		default:
			if err := json.Unmarshal(v, &out[i]); err != nil {
				t.Fatalf("decode %s: %v", v, err)
			}
		}
	}
	return out
}

func toFloats(v any) []float64 {
	switch s := v.(type) {
	case []float64:
		return s
	case []int32:
		return convert(s)
	case []int64:
		return convert(s)
	case []int8:
		return convert(s)
	}
	panic(fmt.Sprintf("unexpected output type %T", v))
}

func convert[T int32 | int64 | int8](s []T) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = float64(v)
	}
	return out
}

func closeEnough(got, want float64) bool {
	if math.IsNaN(want) || math.IsInf(want, 0) {
		return math.IsNaN(got) && math.IsNaN(want) || got == want
	}
	return math.Abs(got-want) <= goldenRelTol*math.Max(1, math.Abs(want))
}

func compareSeries(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if !closeEnough(got[i], want[i]) {
			t.Fatalf("%s[%d] = %v, want %v", label, i, got[i], want[i])
		}
	}
}

func TestGoldenFunctions(t *testing.T) {
	g, cols := loadGolden(t)
	if len(g.Functions) != len(goldenFunctions) {
		t.Errorf("fixture has %d functions, Go dispatch has %d: regenerate fixtures and bindings",
			len(g.Functions), len(goldenFunctions))
	}
	for _, c := range g.Functions {
		t.Run(c.Fn, func(t *testing.T) {
			call, ok := goldenFunctions[c.Fn]
			if !ok {
				t.Fatalf("no Go wrapper for %s", c.Fn)
			}
			in := make(map[string][]float64, len(c.Inputs))
			for arg, col := range c.Inputs {
				in[arg] = cols[col]
			}
			got, err := call(in, c.Params)
			if err != nil {
				t.Fatal(err)
			}
			for name, want := range c.Outputs {
				compareSeries(t, name, toFloats(got[name]), decodeFloats(t, want))
			}
		})
	}
}

func TestGoldenStreams(t *testing.T) {
	g, cols := loadGolden(t)
	if len(g.Streams) != len(goldenStreams) {
		t.Errorf("fixture has %d streams, Go dispatch has %d", len(g.Streams), len(goldenStreams))
	}
	for _, c := range g.Streams {
		t.Run(c.Stream, func(t *testing.T) {
			build, ok := goldenStreams[c.Stream]
			if !ok {
				t.Fatalf("no Go wrapper for stream %s", c.Stream)
			}
			s, err := build(c.Params)
			if err != nil {
				t.Fatal(err)
			}
			defer s.close()
			got := map[string][]float64{}
			for i := range cols["close"] {
				bar := make(map[string]float64, len(c.Inputs))
				for arg, col := range c.Inputs {
					bar[arg] = cols[col][i]
				}
				outs, err := s.step(bar)
				if err != nil {
					t.Fatalf("bar %d: %v", i, err)
				}
				for name, v := range outs {
					got[name] = append(got[name], scalarFloat(v))
				}
			}
			for name, want := range c.Outputs {
				compareSeries(t, name, got[name], decodeFloats(t, want))
			}
		})
	}
}

func scalarFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int8:
		return float64(x)
	}
	panic(fmt.Sprintf("unexpected scalar type %T", v))
}

func TestGoldenScalars(t *testing.T) {
	g, _ := loadGolden(t)
	covered := map[string]bool{}
	for i, c := range g.Scalars {
		covered[c.Fn] = true
		t.Run(fmt.Sprintf("%s/%d", c.Fn, i), func(t *testing.T) {
			call, ok := goldenScalars[c.Fn]
			if !ok {
				t.Fatalf("no Go wrapper for %s", c.Fn)
			}
			got, err := call(c.Params)
			if err != nil {
				t.Fatal(err)
			}
			for name, raw := range c.Outputs {
				want := decodeFloats(t, []json.RawMessage{raw})[0]
				if v := scalarFloat(got[name]); !closeEnough(v, want) {
					t.Fatalf("%s = %v, want %v", name, v, want)
				}
			}
		})
	}
	if len(covered) != len(goldenScalars) {
		t.Errorf("fixture covers %d scalar functions, Go dispatch has %d", len(covered), len(goldenScalars))
	}
}
