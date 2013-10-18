package libsvm

// SVMModel define model of svm
type SVMModel struct {
	Param     *SVMParameter // parameter
	NrClass   int           // number of classes, = 2 in regression/one class svm
	L         int           // total #SV
	SV        [][]SVMNode   // SVs (SV[l])
	SvCoef    [][]float64   // coefficients for SVs in decision functions (svCoef[k-1][l])
	Rho       []float64     // constants in decision functions (rho[k*(k-1)/2])
	ProbA     []float64     // pariwise probability information
	ProbB     []float64
	SvIndices []int // svIndices[0,...,nSV-1] are values in [1,...,numTraningData] to indicate SVs in the training set

	// for classification only
	Label []int // label of each class (label[k])
	NSV   []int // number of SVs for each class (nSV[k])  nSV[0] + nSV[1] + ... + nSV[k-1] = l
}
