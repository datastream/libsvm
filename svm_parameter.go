package libsvm

const (
	/* svm_type */
	C_SVC = 0
	NU_SVC = 1
	ONE_CLASS = 2
	EPSILON_SVR = 3
	NU_SVR = 4

	/* kernel_type */
	LINEAR = 0
	POLY = 1
	RBF = 2
	SIGMOID = 3
	PRECOMPUTED = 4
)

type SVM_Parameter struct {
	svm_type int
	kernel_type int
	degree int	// for poly
	gamma float64	// for poly/rbf/sigmoid
	coef0 float64	// for poly/sigmoid

	// these are for training only
	cache_size float64 // in MB
	eps float64	// stopping criteria
	C float64	// for C_SVC, EPSILON_SVR and NU_SVR
	nr_weight int		// for C_SVC
	weight_label []int	// for C_SVC
	weight []float64		// for C_SVC
	nu float64	// for NU_SVC, ONE_CLASS, and NU_SVR
	p float64	// for EPSILON_SVR
	shrinking int	// use the shrinking heuristics
	probability int // do probability estimates

}
func (this *SVM_Parameter) Clone() (rst *SVM_Parameter) {
	rst = new(SVM_Parameter)
	*rst = *this
	return
}
