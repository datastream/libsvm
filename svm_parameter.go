package libsvm

const (
	/* svmType */
	CSVC       = 0
	NUSVC      = 1
	ONECLASS   = 2
	EPSILONSVR = 3
	NUSVR      = 4

	/* kernelType */

	LINEAR      = 0
	POLY        = 1
	RBF         = 2
	SIGMOID     = 3
	PRECOMPUTED = 4
)

// SVMParameter define param for svm
type SVMParameter struct {
	SvmType    int
	KernelType int
	Degree     int     // for poly
	Gamma      float64 // for poly/rbf/sigmoid
	Coef0      float64 // for poly/sigmoid

	// these are for training only
	CacheSize   float64   // in MB
	Eps         float64   // stopping criteria
	C           float64   // for CSVC, EPSILONSVR and NUSVR
	NrWeight    int       // for CSVC
	WeightLabel []int     // for CSVC
	Weight      []float64 // for CSVC
	Nu          float64   // for NUSVC, ONECLASS, and NUSVR
	P           float64   // for EPSILONSVR
	Shrinking   int       // use the shrinking heuristics
	Probability int       // do probability estimates

}

// Clone SVMParameter
func (s *SVMParameter) Clone() *SVMParameter {
	rst := new(SVMParameter)
	rst.SvmType = s.SvmType
	rst.KernelType = s.KernelType
	rst.Degree = s.Degree
	rst.Gamma = s.Gamma
	rst.Coef0 = s.Coef0
	rst.CacheSize = s.CacheSize
	rst.Eps = s.Eps
	rst.C = s.C
	rst.NrWeight = s.NrWeight
	copy(rst.WeightLabel, s.WeightLabel)
	copy(rst.Weight, s.Weight)
	rst.Nu = s.Nu
	rst.P = s.P
	rst.Shrinking = s.Shrinking
	rst.Probability = s.Probability
	return rst
}
