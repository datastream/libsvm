package libsvm

type SVM_Model struct {
	param    *SVM_Parameter // parameter
	nr_class int            // number of classes, = 2 in regression/one class svm
	l        int            // total #SV
	SV       [][]SVM_Node   // SVs (SV[l])
	sv_coef  [][]float64    // coefficients for SVs in decision functions (sv_coef[k-1][l])
	rho      []float64      // constants in decision functions (rho[k*(k-1)/2])
	probA    []float64      // pariwise probability information
	probB    []float64
	sv_indices []int       // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

	label []int // label of each class (label[k])
	nSV   []int // number of SVs for each class (nSV[k])  nSV[0] + nSV[1] + ... + nSV[k-1] = l
}
