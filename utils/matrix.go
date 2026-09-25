package utils 

func Matmul(matA, matB [][]float64) [][]float64 {
	rowsA := len(matA)
	colsA := len(matA[0])
	rowsB := len(matB)
	colsB := len(matB[0])
	if colsA != rowsB {
		return nil
	}

	result := make([][]float64, rowsA)

	for i := range rowsA {
		result[i] = make([]float64, colsB)
		for j := range colsB {
			for k := range colsA {
				result[i][j] += matA[i][k] * matB[k][j]
			}
		}
	}
	return result
}
