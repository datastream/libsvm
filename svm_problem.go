package libsvm

type SVM_Problem struct {
	L int
	Y []float64
	X [][]SVM_Node
}
