package libsvm

const (
	/* svm_type */
	C_SVC       = 0
	NU_SVC      = 1
	ONE_CLASS   = 2
	EPSILON_SVR = 3
	NU_SVR      = 4

	/* kernel_type */
	LINEAR      = 0
	POLY        = 1
	RBF         = 2
	SIGMOID     = 3
	PRECOMPUTED = 4
)

type SVM_Parameter struct {
	Svm_type    int
	Kernel_type int
	Degree      int     // for poly
	Gamma       float64 // for poly/rbf/sigmoid
	Coef0       float64 // for poly/sigmoid

	// these are for training only
	Cache_size   float64   // in MB
	Eps          float64   // stopping criteria
	C            float64   // for C_SVC, EPSILON_SVR and NU_SVR
	Nr_weight    int       // for C_SVC
	Weight_label []int     // for C_SVC
	Weight       []float64 // for C_SVC
	Nu           float64   // for NU_SVC, ONE_CLASS, and NU_SVR
	P            float64   // for EPSILON_SVR
	Shrinking    int       // use the shrinking heuristics
	Probability  int       // do probability estimates

}

func (this *SVM_Parameter) Clone() *SVM_Parameter {
	rst := new(SVM_Parameter)
	return rst
}
