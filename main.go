package main

import (
	"digit-recognizer/utils"
	"fmt"
	"math/rand"
)

type Dataset struct {
	Images [][]float64 // 28x28 float64 values in range [0, 1]
	Labels []int32 // 0-9
}

type Layer struct {
	Weight [][]float64
	Bias []float64
}

type Network struct {
	Layers []Layer
}

func generateRandomWeights(row int, col int) [][]float64 {
	mat := make([][]float64, row)
	for i := range mat {
		mat[i] = make([]float64, col)
		for j := range mat[i] {
			mat[i][j] = rand.Float64()
		}
	}
	return mat
}

func generateRandomBiases(row int) []float64 {
	vec := make([]float64, row)
	for i := range row {
		vec[i] = rand.Float64()
	}
	return vec
}

func initLayer(row int, col int) Layer {
	weightMatrix := generateRandomWeights(row, col)
	biasVector := generateRandomBiases(row)
	layer := Layer{Weight: weightMatrix, Bias: biasVector}
	return layer
}

func main() {
	images, err := utils.ReadIDXFile("data/train-images-idx3-ubyte")
	if err != nil {
		fmt.Printf("Error reading images: %v\n", err)
		return
	}
	labels, err := utils.ReadLabels("data/train-labels-idx1-ubyte")
	if err != nil {
		fmt.Printf("Error reading labels: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d images and %d labels\n", len(images), len(labels))

	// --------------------
}