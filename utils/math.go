package utils

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SoftMaxVec(logits []float64) []float64 {
	if len(logits) == 0 { return nil }

	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	probabilities := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		probabilities[i] = math.Exp(v - maxLogit)
		sum += probabilities[i]
	}
	for i := range probabilities {
		probabilities[i] /= sum
	}
	return probabilities
}