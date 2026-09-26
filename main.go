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

func initLayer(inputSize int, outputSize int) Layer {
	weightMatrix := generateRandomWeights(inputSize, outputSize)
	biasVector := generateRandomBiases(outputSize)
	layer := Layer{Weight: weightMatrix, Bias: biasVector}
	return layer
}

// return simple 3 layer neural net with 2 hidden layers and 1 output layer
func createNetwork() Network {
	nn := Network{make([]Layer, 3)}
	nn.Layers[0] = initLayer(784,16)
	nn.Layers[1] = initLayer(16, 16)
	nn.Layers[2] = initLayer(16, 10)
	return nn
}

func forwardPass(layer Layer, input [][]float64) []float64 {
	multiplied := utils.Matmul(input, layer.Weight)
	transformed := multiplied[0]
	outputVector := make([]float64, len(transformed))

	for i, value := range transformed {
		outputVector[i] = utils.SoftMax(value + layer.Bias[i])
	} 
	fmt.Printf("multiplied dimensions: %dx%d\n", len(multiplied), len(multiplied[0]))
	return outputVector
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
	// extract the first set of activations (784 nodes) and pass them into
	// feedforward neural net which consists of 2 hidden layers (784 -> 16 -> 16)
	firstActivations := images[0]
	mat := make([][]float64, 1)
	mat[0] = firstActivations
	network := createNetwork()

	for i := range network.Layers {
		matrix := mat
		mat[0] = forwardPass(network.Layers[i], matrix)
	}

	output := mat[0]
	fmt.Printf("activation before: %d, after: %d", len(firstActivations), len(output))
	fmt.Printf("output: %v", output)
}