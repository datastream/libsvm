package libsvm

type SVM_Model struct {
	Param      *SVM_Parameter // parameter
	Nr_class   int            // number of classes, = 2 in regression/one class svm
	L          int            // total #SV
	SV         [][]SVM_Node   // SVs (SV[l])
	Sv_coef    [][]float64    // coefficients for SVs in decision functions (sv_coef[k-1][l])
	Rho        []float64      // constants in decision functions (rho[k*(k-1)/2])
	ProbA      []float64      // pariwise probability information
	ProbB      []float64
	Sv_indices []int // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

	// for classification only
	Label []int // label of each class (label[k])
	NSV   []int // number of SVs for each class (nSV[k])  nSV[0] + nSV[1] + ... + nSV[k-1] = l
}
