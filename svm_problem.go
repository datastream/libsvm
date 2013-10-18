package libsvm

// SVMProblem define svm problem
type SVMProblem struct {
	L int
	Y []float64
	X [][]SVMNode
}
