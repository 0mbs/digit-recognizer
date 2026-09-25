package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
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

func readIDXFile(filepath string) ([][]float64, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Metadata headers
	var magic, numImages, rows, cols int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &numImages)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	if magic != 2051 {
		return nil, fmt.Errorf("invalid image magic number: %d", magic)
	}

	imageSize := int(rows * cols) // 28 * 28 = 784
	images := make([][]float64, numImages)

	// Temporal buffer to read 1 image at a time
	buf := make([]byte, imageSize)

	for i := 0; i < int(numImages); i++ {
		_, err := io.ReadFull(file, buf)
		if err != nil {
			return nil, err
		}

		images[i] = make([]float64, imageSize)
		for j, pixel := range buf {
			// Normalize pixel values directly to [0.0, 1.0]
			images[i][j] = float64(pixel) / 255.0
		}
	}
	return images, nil
}

func ReadLabels(filename string) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, numItems int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &numItems)

	if magic != 2049 {
		return nil, fmt.Errorf("invalid label magic number: %d", magic)
	}

	labels := make([]int, numItems)
	buf := make([]byte, numItems)

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, err
	}

	for i, val := range buf {
		labels[i] = int(val)
	}

	return labels, nil
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
	images, err := readIDXFile("data/train-images-idx3-ubyte")
	if err != nil {
		fmt.Printf("Error reading images: %v\n", err)
		return
	}
	labels, err := ReadLabels("data/train-labels-idx1-ubyte")
	if err != nil {
		fmt.Printf("Error reading labels: %v\n", err)
		return
	}

	fmt.Printf("Loaded %d images and %d labels\n", len(images), len(labels))
}