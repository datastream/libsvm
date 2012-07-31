package libsvm

type SVM_Node struct {
	index int
	value float64
}
func (this *SVM_Node)clone() *SVM_Node {
	rst := new(SVM_Node)
	*rst = *this
	return rst
}
