package libsvm

// SVMNode define node of svm
type SVMNode struct {
	Index int
	Value float64
}

func (s *SVMNode) clone() *SVMNode {
	rst := new(SVMNode)
	rst.Index = s.Index
	rst.Value = s.Value
	return rst
}
