package test

import (
	"digit-recognizer/utils"
	"reflect"
	"testing"
)

func TestMatmul(t *testing.T) {
	tests := []struct {
		name string
		matA [][]float64
		matB [][]float64
		want [][]float64
	}{
		{
			name: "2x2 times 2x2",
			matA: [][]float64{{1, 2}, {3, 4}},
			matB: [][]float64{{5, 6}, {7, 8}},
			// [1*5+2*7, 1*6+2*8] = [19, 22]
			// [3*5+4*7, 3*6+4*8] = [43, 50]
			want: [][]float64{{19, 22}, {43, 50}},
		},
		{
			name: "identity matrix leaves input unchanged",
			matA: [][]float64{{1, 2}, {3, 4}},
			matB: [][]float64{{1, 0}, {0, 1}},
			want: [][]float64{{1, 2}, {3, 4}},
		},
		{
			name: "non-square 2x3 times 3x2",
			matA: [][]float64{{1, 2, 3}, {4, 5, 6}},
			matB: [][]float64{{7, 8}, {9, 10}, {11, 12}},
			// row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
			// row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
			want: [][]float64{{58, 64}, {139, 154}},
		},
		{
			name: "row vector times column vector yields 1x1",
			matA: [][]float64{{1, 2, 3}},
			matB: [][]float64{{4}, {5}, {6}},
			want: [][]float64{{32}}, // 1*4 + 2*5 + 3*6
		},
		{
			name: "zero matrix",
			matA: [][]float64{{0, 0}, {0, 0}},
			matB: [][]float64{{1, 2}, {3, 4}},
			want: [][]float64{{0, 0}, {0, 0}},
		},
		{
			name: "mismatched dimensions returns nil",
			matA: [][]float64{{1, 2}, {3, 4}},
			matB: [][]float64{{1, 2, 3}},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := utils.Matmul(tt.matA, tt.matB)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("matmul(%v, %v) = %v, want %v", tt.matA, tt.matB, got, tt.want)
			}
		})
	}
}
