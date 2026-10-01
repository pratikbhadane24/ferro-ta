package ferrota_test

import (
	"fmt"

	ferrota "github.com/pratikbhadane24/ferro-ta/bindings/go"
)

func ExampleSma() {
	sma, err := ferrota.Sma([]float64{1, 2, 3, 4, 5}, 3)
	if err != nil {
		panic(err)
	}
	fmt.Println(sma)
	// Output: [NaN NaN 2 3 4]
}

func ExampleBbands() {
	closes := []float64{10, 11, 12, 11, 10, 11, 12}
	upper, middle, lower, err := ferrota.Bbands(closes, 5, 2, 2, ferrota.MATypeSMA)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.4f %.4f %.4f\n", upper[6], middle[6], lower[6])
	// Output: 12.6967 11.2000 9.7033
}

func ExampleNewStreamSma() {
	s, err := ferrota.NewStreamSma(3)
	if err != nil {
		panic(err)
	}
	defer s.Close()
	for _, price := range []float64{1, 2, 3, 4} {
		v, _ := s.Update(price)
		fmt.Println(v)
	}
	// Output:
	// NaN
	// NaN
	// 2
	// 3
}
