package libsvm

type SVM_Node struct {
	Index int
	Value float64
}

func (this *SVM_Node) clone() *SVM_Node {
	rst := new(SVM_Node)
	*rst = *this
	return rst
}
