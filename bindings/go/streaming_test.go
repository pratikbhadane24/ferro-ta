package ferrota

import (
	"errors"
	"math"
	"runtime"
	"sync"
	"testing"
)

func TestStreamMatchesBatch(t *testing.T) {
	in := ramp(60)
	for i := range in {
		in[i] += math.Sin(float64(i)) * 3
	}
	batch, err := Rsi(in, 14)
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewStreamRsi(14)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for i, x := range in {
		got, err := s.Update(x)
		if err != nil {
			t.Fatal(err)
		}
		if !closeEnough(got, batch[i]) {
			t.Fatalf("bar %d: stream %v, batch %v", i, got, batch[i])
		}
	}
}

func TestStreamReset(t *testing.T) {
	s, err := NewStreamSma(3)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, x := range []float64{1, 2, 3} {
		if _, err := s.Update(x); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if v, _ := s.Update(9); !math.IsNaN(v) {
		t.Fatalf("after Reset want warm-up NaN, got %v", v)
	}
}

func TestStreamInvalidParams(t *testing.T) {
	if _, err := NewStreamSma(0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v", err)
	}
	// Rejected by the core itself: fast must be < slow.
	if _, err := NewStreamMacd(26, 12, 9); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v", err)
	}
}

func TestStreamUseAfterClose(t *testing.T) {
	s, err := NewStreamBbands(5, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close must be a no-op, got %v", err)
	}
	if _, _, _, err := s.Update(1); !errors.Is(err, ErrClosed) {
		t.Fatalf("got %v, want ErrClosed", err)
	}
	if err := s.Reset(); !errors.Is(err, ErrClosed) {
		t.Fatalf("got %v, want ErrClosed", err)
	}
}

func TestStreamConcurrentUpdates(t *testing.T) {
	s, err := NewStreamVwap()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < 200; k++ {
				if _, err := s.Update(11, 9, 10, 100); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	// Every bar is identical, so the cumulative VWAP is exactly the typical price.
	if v, _ := s.Update(11, 9, 10, 100); v != 10 {
		t.Fatalf("VWAP = %v, want 10", v)
	}
}

func TestStreamFinalizerReleasesHandle(t *testing.T) {
	for i := 0; i < 1000; i++ {
		if _, err := NewStreamEma(10); err != nil {
			t.Fatal(err)
		}
	}
	runtime.GC()
	runtime.GC()
}
