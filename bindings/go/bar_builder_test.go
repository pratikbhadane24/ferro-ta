package ferrota

import (
	"errors"
	"testing"
)

// istMs is UTC epoch ms for an IST wall-clock time on a fixed day.
func istMs(h, m int64) int64 {
	return 20_377*86_400_000 + (h*60+m)*60_000 - 19_800_000
}

func TestBarBuilderClosesOnBoundaryAndTime(t *testing.T) {
	b, err := NewNSEBarBuilder(300)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if st, closed, err := b.OnTick(istMs(9, 16), 100, 1); err != nil || st != TickAccepted || closed != nil {
		t.Fatalf("first tick: %v %v %v", st, closed, err)
	}
	st, closed, err := b.OnTick(istMs(9, 20), 101, 1)
	if err != nil || st != TickAccepted || closed == nil {
		t.Fatalf("boundary tick: %v %v %v", st, closed, err)
	}
	if closed.StartMs != istMs(9, 15) || closed.EndMs != istMs(9, 20) || closed.Close != 100 {
		t.Fatalf("closed bar %+v", *closed)
	}
	if st, _, _ := b.OnTick(istMs(9, 10), 99, 1); st != TickOutOfSession {
		t.Fatalf("want out of session, got %v", st)
	}
	bar, err := b.OnTime(istMs(9, 25))
	if err != nil || bar == nil || bar.StartMs != istMs(9, 20) {
		t.Fatalf("OnTime: %+v %v", bar, err)
	}
	if f, _ := b.Forming(); f != nil {
		t.Fatalf("forming should be nil, got %+v", *f)
	}
}

func TestBarBuilderInvalidAndClosed(t *testing.T) {
	if _, err := NewBarBuilder(0, 0, 100, 0); !errors.Is(err, ErrInvalidParam) {
		t.Fatalf("got %v", err)
	}
	b, _ := NewNSEBarBuilder(60)
	b.Close()
	if _, _, err := b.OnTick(0, 1, 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("got %v", err)
	}
}
