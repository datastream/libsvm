package libsvm

type SVM_Problem struct {
	l int
	y []float64
	x [][]SVM_Node
}
