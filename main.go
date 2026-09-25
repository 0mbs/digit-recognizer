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

func initLayer(from int, to int) Layer {
	weightMatrix := generateRandomWeights(from, to)
	biasVector := generateRandomBiases(from)
	layer := Layer{Weight: weightMatrix, Bias: biasVector}
	return layer
}

func forwardPass(layer Layer, inputVector [][]float64) []float64 {
	multiplied := utils.Matmul(inputVector, layer.Weight)
	fmt.Printf("multiplied dimensions: %dx%d\n", len(multiplied), len(multiplied[0]))
	return multiplied[0]
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
	// initialize first layer with Weight [784 x 16] & bias [16] to transform
	// 784 activations to 16.
	firstActivations := images[0]
	input := make([][]float64, 1)
	input[0] = firstActivations
	layer1 := initLayer(784,16)

	// fmt.Printf("forward pass with %s input", input)
	secondActivations := forwardPass(layer1, input)
	fmt.Printf("output: %v", secondActivations)

	fmt.Printf("activation before: %d, after: %d", len(firstActivations), len(secondActivations))
}