package ferrota

import (
	"math"
	"testing"
)

const benchBars = 1_000_000

func benchSeries() []float64 {
	s := make([]float64, benchBars)
	for i := range s {
		s[i] = 100 + math.Sin(float64(i)*0.01)*10
	}
	return s
}

func BenchmarkSma(b *testing.B) {
	in := benchSeries()
	b.SetBytes(int64(len(in) * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Sma(in, 20); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRsi(b *testing.B) {
	in := benchSeries()
	b.SetBytes(int64(len(in) * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Rsi(in, 14); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBbands(b *testing.B) {
	in := benchSeries()
	b.SetBytes(int64(len(in) * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := Bbands(in, 20, 2, 2, MATypeSMA); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStreamSmaUpdate(b *testing.B) {
	s, err := NewStreamSma(20)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	for i := 0; i < b.N; i++ {
		if _, err := s.Update(100); err != nil {
			b.Fatal(err)
		}
	}
}
