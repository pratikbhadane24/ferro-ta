package ferrota

/*
#include "ferro_ta.h"
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

// Bar is one OHLCV bar covering [StartMs, EndMs) in UTC epoch milliseconds.
type Bar struct {
	StartMs int64
	EndMs   int64
	Open    float64
	High    float64
	Low     float64
	Close   float64
	Volume  float64
	Ticks   uint32
}

// TickStatus reports what BarBuilder.OnTick did with a tick.
type TickStatus int32

// TickStatus values; only TickAccepted changes builder state.
const (
	TickAccepted     TickStatus = C.FT_TICK_ACCEPTED
	TickLate         TickStatus = C.FT_TICK_LATE
	TickDuplicate    TickStatus = C.FT_TICK_DUPLICATE
	TickOutOfSession TickStatus = C.FT_TICK_OUT_OF_SESSION
	TickInvalid      TickStatus = C.FT_TICK_INVALID
)

func barFromC(b C.FtBar) Bar {
	return Bar{
		StartMs: int64(b.start_ms),
		EndMs:   int64(b.end_ms),
		Open:    float64(b.open),
		High:    float64(b.high),
		Low:     float64(b.low),
		Close:   float64(b.close),
		Volume:  float64(b.volume),
		Ticks:   uint32(b.ticks),
	}
}

// BarBuilder builds session-aligned OHLCV bars from ticks: bars start at
// sessionOpen + k*interval (the last bar is cut at sessionClose) and gaps
// produce no bars. Session times are seconds after local midnight; local
// time is UTC + utcOffsetSecs, a fixed offset with no DST rules (create a new
// builder when a market's offset changes).
//
// Methods are safe for concurrent use. Call Close when done; a finalizer
// releases the handle if Close is forgotten.
type BarBuilder struct {
	mu sync.Mutex
	h  *C.FtBarBuilder
}

// NewBarBuilder creates a BarBuilder for an arbitrary session.
func NewBarBuilder(intervalSecs, sessionOpenSecs, sessionCloseSecs, utcOffsetSecs int64) (*BarBuilder, error) {
	var h *C.FtBarBuilder
	status := C.ft_bar_builder_new(C.int64_t(intervalSecs), C.int64_t(sessionOpenSecs),
		C.int64_t(sessionCloseSecs), C.int64_t(utcOffsetSecs), &h)
	if err := statusError("NewBarBuilder", status); err != nil {
		return nil, err
	}
	b := &BarBuilder{h: h}
	runtime.SetFinalizer(b, (*BarBuilder).Close)
	return b, nil
}

// NewNSEBarBuilder creates a BarBuilder for NSE hours (09:15-15:30 IST).
func NewNSEBarBuilder(intervalSecs int64) (*BarBuilder, error) {
	return NewBarBuilder(intervalSecs, 9*3600+15*60, 15*3600+30*60, 5*3600+30*60)
}

// OnTick feeds one tick. closed is non-nil when this tick closed a bar.
func (b *BarBuilder) OnTick(tsMs int64, price, qty float64) (status TickStatus, closed *Bar, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h == nil {
		return TickInvalid, nil, closedError("BarBuilder.OnTick")
	}
	var cStatus, cClosed C.int32_t
	var cBar C.FtBar
	rc := C.ft_bar_builder_on_tick(b.h, C.int64_t(tsMs), C.double(price), C.double(qty), &cStatus, &cClosed, &cBar)
	if err := statusError("BarBuilder.OnTick", rc); err != nil {
		return TickInvalid, nil, err
	}
	if cClosed != 0 {
		bar := barFromC(cBar)
		closed = &bar
	}
	return TickStatus(cStatus), closed, nil
}

// OnTime closes the forming bar once nowMs reaches its end, with no tick
// needed; it returns the closed bar or nil.
func (b *BarBuilder) OnTime(nowMs int64) (*Bar, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h == nil {
		return nil, closedError("BarBuilder.OnTime")
	}
	var cClosed C.int32_t
	var cBar C.FtBar
	if err := statusError("BarBuilder.OnTime", C.ft_bar_builder_on_time(b.h, C.int64_t(nowMs), &cClosed, &cBar)); err != nil {
		return nil, err
	}
	if cClosed == 0 {
		return nil, nil
	}
	bar := barFromC(cBar)
	return &bar, nil
}

// Forming returns the bar being built, or nil.
func (b *BarBuilder) Forming() (*Bar, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h == nil {
		return nil, closedError("BarBuilder.Forming")
	}
	var cHas C.int32_t
	var cBar C.FtBar
	if err := statusError("BarBuilder.Forming", C.ft_bar_builder_forming(b.h, &cHas, &cBar)); err != nil {
		return nil, err
	}
	if cHas == 0 {
		return nil, nil
	}
	bar := barFromC(cBar)
	return &bar, nil
}

// Reset drops the forming bar and tick history.
func (b *BarBuilder) Reset() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h == nil {
		return closedError("BarBuilder.Reset")
	}
	return statusError("BarBuilder.Reset", C.ft_bar_builder_reset(b.h))
}

// Save returns a versioned JSON snapshot, including the forming bar; restore
// it with LoadBarBuilder.
func (b *BarBuilder) Save() ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h == nil {
		return nil, closedError("BarBuilder.Save")
	}
	var n C.size_t
	if rc := C.ft_bar_builder_save(b.h, nil, 0, &n); rc != C.FT_ERR_LENGTH_MISMATCH {
		return nil, statusError("BarBuilder.Save", rc)
	}
	buf := make([]byte, int(n))
	rc := C.ft_bar_builder_save(b.h, (*C.char)(unsafe.Pointer(&buf[0])), n, &n)
	if err := statusError("BarBuilder.Save", rc); err != nil {
		return nil, err
	}
	return buf[:int(n)], nil
}

// LoadBarBuilder restores a BarBuilder from a snapshot made by Save.
func LoadBarBuilder(snapshot []byte) (*BarBuilder, error) {
	var h *C.FtBarBuilder
	var p *C.char
	if len(snapshot) > 0 {
		p = (*C.char)(unsafe.Pointer(&snapshot[0]))
	}
	if err := statusError("LoadBarBuilder", C.ft_bar_builder_load(p, C.size_t(len(snapshot)), &h)); err != nil {
		return nil, err
	}
	b := &BarBuilder{h: h}
	runtime.SetFinalizer(b, (*BarBuilder).Close)
	return b, nil
}

// Close releases the underlying handle. It is safe to call more than once.
func (b *BarBuilder) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.h != nil {
		C.ft_bar_builder_free(b.h)
		b.h = nil
		runtime.SetFinalizer(b, nil)
	}
	return nil
}
