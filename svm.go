package libsvm

import (
	"bufio"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type headT struct {
	prev, next *headT
	data       []float32
	length     int
}

// Cache for svm
type Cache struct {
	l       int
	size    int64
	head    []*headT
	lruHead *headT
}

// NewCache create cache
func NewCache(l int, size int64) *Cache {
	m := new(Cache)
	m.l = l
	m.size = size
	m.head = make([]*headT, m.l)
	for i := 0; i < m.l; i++ {
		m.head[i] = new(headT)
	}
	m.size /= 4
	m.size -= int64(m.l * (16 / 4))                           // sizeof(headT) == 16
	m.size = int64(math.Max(float64(m.size), float64(2*m.l))) // cache must be large enough for two columns
	m.lruHead = new(headT)
	m.lruHead.prev = m.lruHead
	m.lruHead.next = m.lruHead.prev
	return m
}

func (m *Cache) lruDelete(h *headT) {
	// delete from current location
	h.prev.next = h.next
	h.next.prev = h.prev
}
func (m *Cache) lruInsert(h *headT) {
	// insert to last position
	h.next = m.lruHead
	h.prev = m.lruHead.prev
	h.prev.next = h
	h.next.prev = h
}

// request data [0,length)
// return some position p where [p,length) need to be filled
// (p >= length if nothing needs to be filled)
// java: simulate pointer using single-element array

func (m *Cache) getData(index int, data [][]float32, length int) int {
	h := m.head[index]
	if h.length > 0 {
		m.lruDelete(h)
	}
	more := length - h.length
	if more > 0 {
		// free old space
		for m.size < int64(more) {
			old := m.lruHead.next
			m.lruDelete(old)
			m.size += int64(old.length)
			old.data = nil
			old.length = 0
		}

		// allocate new space
		newData := make([]float32, length)
		if h.data != nil {
			// System.arraycopy(h.data,0,newData,0,h.length)
			copy(newData, h.data[:length])
		}
		h.data = newData
		m.size -= int64(more)
		// do {int _=h.length; h.length=length; length=_;} while(false);
		h.length, length = length, h.length
	}

	m.lruInsert(h)
	data[0] = h.data
	return length
}

func (m *Cache) swapIndex(i, j int) {
	if i == j {
		return
	}

	if m.head[i].length > 0 {
		m.lruDelete(m.head[i])
	}
	if m.head[j].length > 0 {
		m.lruDelete(m.head[j])
	}
	//do {float32[] _=head[i].data; head[i].data=head[j].data; head[j].data=_;} while(false);
	m.head[i].data, m.head[j].data = m.head[j].data, m.head[i].data
	//do {int _=head[i].length; head[i].length=head[j].length; head[j].length=_;} while(false);
	m.head[i].length, m.head[j].length = m.head[j].length, m.head[i].length

	if m.head[i].length > 0 {
		m.lruInsert(m.head[i])
	}
	if m.head[j].length > 0 {
		m.lruInsert(m.head[j])
	}

	if i > j {
		//do {int _=i; i=j; j=_;} while(false);
		i, j = j, i
	}
	for h := m.lruHead.next; h != m.lruHead; h = h.next {
		if h.length > i {
			if h.length > j {
				//do {float32 _=h.data[i]; h.data[i]=h.data[j]; h.data[j]=_;} while(false);
				h.data[i], h.data[j] = h.data[j], h.data[i]
			} else {
				// give up
				m.lruDelete(h)
				m.size += int64(h.length)
				h.data = nil
				h.length = 0
			}
		}
	}
}

// QMatrix define qmatrix interface
type QMatrix interface {
	swapIndex(i, j int)
	getQ(column, length int) []float32
	getQD() []float64
}

// Kernel struct
type Kernel struct {
	x          [][]SVMNode
	xSquare    []float64
	kernelType int
	degree     int
	gamma      float64
	coef0      float64
}

func (m *Kernel) swapIndex(i, j int) {
	m.x[i], m.x[j] = m.x[j], m.x[i]
	if m.xSquare != nil {
		m.xSquare[i], m.xSquare[j] = m.xSquare[j], m.xSquare[i]
	}
}

func powi(base float64, times int) float64 {
	tmp := base
	ret := 1.0

	for t := times; t > 0; t /= 2 {
		if t%2 == 1 {
			ret *= tmp
		}
		tmp = tmp * tmp
	}
	return ret
}

func (m *Kernel) kernelFunction(i, j int) float64 {
	switch m.kernelType {
	case LINEAR:
		return dot(m.x[i], m.x[j])
	case POLY:
		return powi(m.gamma*dot(m.x[i], m.x[j])+m.coef0, m.degree)
	case RBF:
		return math.Exp(-m.gamma * (m.xSquare[i] + m.xSquare[j] - 2*dot(m.x[i], m.x[j])))
	case SIGMOID:
		return math.Tanh(m.gamma*dot(m.x[i], m.x[j]) + m.coef0)
	case PRECOMPUTED:
		return m.x[i][int(m.x[j][0].Value)].Value
	}
	return 0
}

// NewKernel create kernel from give svmnode
func NewKernel(l int, x [][]SVMNode, param *SVMParameter) *Kernel {
	m := new(Kernel)
	m.kernelType = param.KernelType
	m.degree = param.Degree
	m.gamma = param.Gamma
	m.coef0 = param.Coef0
	// x.clone
	copy(m.x, x)

	if m.kernelType == RBF {
		m.xSquare = make([]float64, l)
		for i := 0; i < l; i++ {
			m.xSquare[i] = dot(m.x[i], m.x[i])
		}
	} else {
		m.xSquare = nil
	}
	return m
}

func dot(x, y []SVMNode) float64 {
	var sum float64
	sum = 0
	xlength := len(x)
	ylength := len(y)
	i := 0
	j := 0
	for i < xlength && j < ylength {
		if x[i].Index == y[j].Index {
			sum += x[i].Value * y[j].Value
			i++
			j++
		} else {
			if x[i].Index > y[j].Index {
				j++
			} else {
				i++
			}
		}
	}
	return sum
}

// kFunction
func kFunction(x, y []SVMNode, param *SVMParameter) float64 {
	switch param.KernelType {
	case LINEAR:
		return dot(x, y)
	case POLY:
		return powi(param.Gamma*dot(x, y)+param.Coef0, param.Degree)
	case RBF:
		{
			var sum float64
			sum = 0
			xlength := len(x)
			ylength := len(y)
			i := 0
			j := 0
			for i < xlength && j < ylength {
				if x[i].Index == y[j].Index {
					d := x[i].Value - y[j].Value
					i++
					j++
					sum += d * d
				} else if x[i].Index > y[j].Index {
					sum += y[j].Value * y[j].Value
					j++
				} else {
					sum += x[i].Value * x[i].Value
					i++
				}
			}

			for i < xlength {
				sum += x[i].Value * x[i].Value
				i++
			}

			for j < ylength {
				sum += y[j].Value * y[j].Value
				j++
			}
			return math.Exp(-param.Gamma * sum)
		}
	case SIGMOID:
		return math.Tanh(param.Gamma*dot(x, y) + param.Coef0)
	case PRECOMPUTED:
		return x[int(y[0].Value)].Value
	}
	return 0
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		yI = +1 or -1
//		0 <= alphaI <= Cp for yI = 1
//		0 <= alphaI <= Cn for yI = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//

const (
	//byte
	LowerBound = int8(0)
	UpperBound = int8(1)
	FREE       = int8(2)
	INF        = math.MaxFloat64
)

// Solver struct
type Solver struct {
	activeSize  int
	y           []int8
	G           []float64 // gradient of objective function
	alphaStatus []int8    // LowerBound, UpperBound, FREE
	alpha       []float64
	Q           QMatrix
	QD          []float64
	eps         float64
	Cp, Cn      float64
	p           []float64
	activeSet   []int
	GBar        []float64 // gradient, if we treat free variables as 0
	l           int
	unshrink    bool // XXX
}

func (m *Solver) getC(i int) float64 {
	var rst float64
	if m.y[i] > 0 {
		rst = m.Cp
	} else {
		rst = m.Cn
	}
	return rst
}

func (m *Solver) updateAlphaStatus(i int) {
	if m.alpha[i] >= m.getC(i) {
		m.alphaStatus[i] = UpperBound
	} else if m.alpha[i] <= 0 {
		m.alphaStatus[i] = LowerBound
	} else {
		m.alphaStatus[i] = FREE
	}
}

func (m *Solver) isUpperBound(i int) bool {
	return m.alphaStatus[i] == UpperBound
}
func (m *Solver) isLowerBound(i int) bool {
	return m.alphaStatus[i] == LowerBound
}
func (m *Solver) isFree(i int) bool {
	return m.alphaStatus[i] == FREE
}

// SolutionInfo Solution_info
type SolutionInfo struct {
	obj         float64
	rho         float64
	upperBoundP float64
	upperBoundN float64
	r           float64 // for SolverNU
}

func (m *Solver) swapIndex(i, j int) {
	// todo
	m.Q.swapIndex(i, j)
	//do {byte _=y[i]; y[i]=y[j]; y[j]=_;} while(false);
	m.y[i], m.y[j] = m.y[j], m.y[i]
	//do {float64 _=G[i]; G[i]=G[j]; G[j]=_;} while(false);
	m.G[i], m.G[j] = m.G[j], m.G[i]
	//do {byte _=alphaStatus[i]; alphaStatus[i]=alphaStatus[j]; alphaStatus[j]=_;} while(false);
	m.alphaStatus[i], m.alphaStatus[j] = m.alphaStatus[j], m.alphaStatus[i]
	//do {float64 _=alpha[i]; alpha[i]=alpha[j]; alpha[j]=_;} while(false);
	m.alpha[i], m.alpha[j] = m.alpha[j], m.alpha[i]
	//do {float64 _=p[i]; p[i]=p[j]; p[j]=_;} while(false);
	m.p[i], m.p[j] = m.p[j], m.p[i]
	//do {int _=activeSet[i]; activeSet[i]=activeSet[j]; activeSet[j]=_;} while(false);
	m.activeSet[i], m.activeSet[j] = m.activeSet[j], m.activeSet[i]
	//do {float64 _=GBar[i]; GBar[i]=GBar[j]; GBar[j]=_;} while(false);
	m.GBar[i], m.GBar[j] = m.GBar[j], m.GBar[i]
}

func (m *Solver) reconstructGradient() {
	// reconstruct inactive elements of G from GBar and free variables

	if m.activeSize == m.l {
		return
	}
	var i, j int
	nrFree := 0

	for j = m.activeSize; j < m.l; j++ {
		m.G[j] = m.GBar[j] + m.p[j]
	}
	for j = 0; j < m.activeSize; j++ {
		if m.isFree(j) {
			nrFree++
		}
	}
	if (2 * nrFree) < m.activeSize {
		log.Print("\nWARNING: using -h 0 may be faster\n")
	}

	if (nrFree * m.l) > (2 * m.activeSize * (m.l - m.activeSize)) {
		for i = m.activeSize; i < m.l; i++ {
			QI := m.Q.getQ(i, m.activeSize)
			for j = 0; j < m.activeSize; j++ {
				if m.isFree(j) {
					m.G[i] += m.alpha[j] * float64(QI[j])
				}
			}
		}
	} else {
		for i = 0; i < m.activeSize; i++ {
			if m.isFree(i) {
				QI := m.Q.getQ(i, m.l)
				alphaI := m.alpha[i]
				for j = m.activeSize; j < m.l; j++ {
					m.G[j] += alphaI * float64(QI[j])
				}
			}
		}
	}
}

// Solve function
func (m *Solver) Solve(l int, Q QMatrix, p []float64, y []int8, alpha []float64, Cp, Cn, eps float64, si *SolutionInfo, shrinking int) {
	m.l = l
	m.Q = Q
	m.QD = Q.getQD()
	m.p = make([]float64, len(p))
	copy(m.p, p)
	m.y = make([]int8, len(y))
	copy(m.y, y)
	m.alpha = make([]float64, len(alpha))
	copy(m.alpha, alpha)
	m.Cp = Cp
	m.Cn = Cn
	m.eps = eps
	m.unshrink = false

	// initialize alphaStatus
	{
		m.alphaStatus = make([]int8, l)
		for i := 0; i < l; i++ {
			m.updateAlphaStatus(i)
		}
	}

	// initialize active set (for shrinking)
	{
		m.activeSet = make([]int, l)
		for i := 0; i < l; i++ {
			m.activeSet[i] = i
		}
		m.activeSize = l
	}

	// initialize gradient
	{
		m.G = make([]float64, l)
		m.GBar = make([]float64, l)
		var i int
		for i = 0; i < l; i++ {
			m.G[i] = m.p[i]
			m.GBar[i] = 0
		}
		for i = 0; i < l; i++ {
			if !m.isLowerBound(i) {
				QI := m.Q.getQ(i, l)
				alphaI := m.alpha[i]
				var j int
				for j = 0; j < l; j++ {
					m.G[j] += alphaI * float64(QI[j])
				}
				if m.isUpperBound(i) {
					for j = 0; j < l; j++ {
						m.GBar[j] += m.getC(i) * float64(QI[j])
					}
				}
			}
		}
	}

	// optimization step

	iter := 0
	var max float64
	if m.l > math.MaxInt32/100 {
		max = float64(math.MaxInt32)
	} else {
		max = float64(100 * m.l)
	}
	maxIter := math.Max(10000000, max)
	counter := math.Min(float64(m.l), 1000) + 1
	workingSet := make([]int, 2)

	for float64(iter) < maxIter {
		// show progress and do shrinking
		if counter--; counter == 0 {
			counter = math.Min(float64(m.l), 1000)
			if shrinking != 0 {
				m.doShrinking()
			}
			log.Print(".")
		}

		if m.selectWorkingSet(workingSet) != 0 {
			// reconstruct the whole gradient
			m.reconstructGradient()
			// reset active set size and check
			m.activeSize = l
			log.Print("*")
			if m.selectWorkingSet(workingSet) != 0 {
				break
			} else {
				counter = 1 // do shrinking next iteration
			}
		}

		i := workingSet[0]
		j := workingSet[1]

		iter++

		// update alpha[i] and alpha[j], handle bounds carefully

		QI := m.Q.getQ(i, m.activeSize)
		QJ := m.Q.getQ(j, m.activeSize)

		CI := m.getC(i)
		CJ := m.getC(j)

		oldAlphaI := m.alpha[i]
		oldAlphaJ := m.alpha[j]

		if m.y[i] != m.y[j] {
			quadCoef := m.QD[i] + m.QD[j] + 2*float64(QI[j])
			if quadCoef <= 0 {
				quadCoef = 1e-12
			}
			delta := (-m.G[i] - m.G[j]) / quadCoef
			diff := m.alpha[i] - m.alpha[j]
			m.alpha[i] += delta
			m.alpha[j] += delta

			if diff > 0 {
				if m.alpha[j] < 0 {
					m.alpha[j] = 0
					m.alpha[i] = diff
				}
			} else {
				if m.alpha[i] < 0 {
					m.alpha[i] = 0
					m.alpha[j] = -diff
				}
			}
			if diff > (CI - CJ) {
				if m.alpha[i] > CI {
					m.alpha[i] = CI
					m.alpha[j] = CI - diff
				}
			} else {
				if m.alpha[j] > CJ {
					m.alpha[j] = CJ
					m.alpha[i] = CJ + diff
				}
			}
		} else {
			quadCoef := m.QD[i] + m.QD[j] - 2*float64(QI[j])
			if quadCoef <= 0 {
				quadCoef = 1e-12
			}
			delta := (m.G[i] - m.G[j]) / quadCoef
			sum := m.alpha[i] + m.alpha[j]
			m.alpha[i] -= delta
			m.alpha[j] += delta

			if sum > CI {
				if m.alpha[i] > CI {
					m.alpha[i] = CI
					m.alpha[j] = sum - CI
				}
			} else {
				if m.alpha[j] < 0 {
					m.alpha[j] = 0
					m.alpha[i] = sum
				}
			}
			if sum > CJ {
				if m.alpha[j] > CJ {
					m.alpha[j] = CJ
					m.alpha[i] = sum - CJ
				}
			} else {
				if m.alpha[i] < 0 {
					m.alpha[i] = 0
					m.alpha[j] = sum
				}
			}
		}

		// update G

		deltaAlphaI := m.alpha[i] - oldAlphaI
		deltaAlphaJ := m.alpha[j] - oldAlphaJ

		for k := 0; k < m.activeSize; k++ {
			m.G[k] += float64(QI[k])*deltaAlphaI + float64(QJ[k])*deltaAlphaJ
		}

		// update alphaStatus and GBar

		{
			ui := m.isUpperBound(i)
			uj := m.isUpperBound(j)
			m.updateAlphaStatus(i)
			m.updateAlphaStatus(j)
			var k int
			if ui != m.isUpperBound(i) {
				QI = Q.getQ(i, l)
				if ui {
					for k = 0; k < l; k++ {
						m.GBar[k] -= CI * float64(QI[k])
					}
				} else {
					for k = 0; k < l; k++ {
						m.GBar[k] += CI * float64(QI[k])
					}
				}
			}

			if uj != m.isUpperBound(j) {
				QJ = Q.getQ(j, l)
				if uj {
					for k = 0; k < l; k++ {
						m.GBar[k] -= float64(CJ) * float64(QJ[k])
					}
				} else {
					for k = 0; k < l; k++ {
						m.GBar[k] += float64(CJ) * float64(QJ[k])
					}
				}
			}
		}
	}
	if float64(iter) >= maxIter {
		if m.activeSize < l {
			// reconstruct the whole gradient to calculate objective value
			m.reconstructGradient()
			m.activeSize = l
			log.Print("*")
		}
		log.Print("\nWARNING: reaching max number of iterations")
	}

	// calculate rho

	si.rho = m.calculateRho()

	// calculate objective value
	{
		v := float64(0)
		var i int
		for i = 0; i < l; i++ {
			v += m.alpha[i] * (m.G[i] + m.p[i])
		}

		si.obj = v / 2
	}

	// put back the solution
	{
		for i := 0; i < l; i++ {
			alpha[m.activeSet[i]] = m.alpha[i]
		}
	}

	si.upperBoundP = Cp
	si.upperBoundN = Cn
	log.Printf("\noptimization finished, #iter = %d\n", iter)
}

func (m *Solver) selectWorkingSet(workingSet []int) int {
	// return i,j such that
	// i: maximizes -yI * grad(f)I, i in IUp(\alpha)
	// j: mimimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -yJ*grad(f)J < -yI*grad(f)I, j in ILow(\alpha)

	Gmax := -INF
	Gmax2 := -INF
	GmaxIdx := -1
	GminIdx := -1
	objDiffMin := INF

	for t := 0; t < m.activeSize; t++ {
		if m.y[t] == +1 {
			if !m.isUpperBound(t) {
				if -m.G[t] >= Gmax {
					Gmax = -m.G[t]
					GmaxIdx = t
				}
			}
		} else {
			if !m.isLowerBound(t) {
				if m.G[t] >= Gmax {
					Gmax = m.G[t]
					GmaxIdx = t
				}
			}
		}
	}
	i := GmaxIdx
	var QI []float32
	QI = nil

	if i != -1 { // null QI not accessed: Gmax=-INF if i=-1
		QI = m.Q.getQ(i, m.activeSize)
	}

	for j := 0; j < m.activeSize; j++ {
		if m.y[j] == +1 {
			if !m.isLowerBound(j) {
				gradDiff := Gmax + m.G[j]
				if m.G[j] >= Gmax2 {
					Gmax2 = m.G[j]
				}
				if gradDiff > 0 {
					var objDiff float64
					// y []int8
					tmp := float64(m.y[i])
					tmp2 := float64(2 * QI[j])
					quadCoef := m.QD[i] + m.QD[j] - tmp*tmp2
					if quadCoef > 0 {
						objDiff = -(gradDiff * gradDiff) / quadCoef
					} else {
						objDiff = -(gradDiff * gradDiff) / 1e-12
					}

					if objDiff <= objDiffMin {
						GminIdx = j
						objDiffMin = objDiff
					}
				}
			}
		} else {
			if !m.isUpperBound(j) {
				gradDiff := Gmax - m.G[j]
				if -m.G[j] >= Gmax2 {
					Gmax2 = -m.G[j]
				}
				if gradDiff > 0 {
					var objDiff float64
					tmp := float64(m.y[i])
					tmp2 := float64(2 * QI[j])
					quadCoef := m.QD[i] + m.QD[j] + tmp*tmp2
					if quadCoef > 0 {
						objDiff = -(gradDiff * gradDiff) / quadCoef
					} else {
						objDiff = -(gradDiff * gradDiff) / 1e-12
					}

					if objDiff <= objDiffMin {
						GminIdx = j
						objDiffMin = objDiff
					}
				}
			}
		}
	}

	if Gmax+Gmax2 < m.eps {
		return 1
	}

	workingSet[0] = GmaxIdx
	workingSet[1] = GminIdx
	return 0
}

func (m *Solver) beShrunk(i int, Gmax1, Gmax2 float64) bool {
	var rst bool
	if m.isUpperBound(i) {
		if m.y[i] == +1 {
			rst = -m.G[i] > Gmax1
		} else {
			rst = -m.G[i] > Gmax2
		}
	} else if m.isLowerBound(i) {
		if m.y[i] == +1 {
			rst = m.G[i] > Gmax2
		} else {
			rst = m.G[i] > Gmax1
		}
	} else {
		rst = false
	}
	return rst
}

func (m *Solver) doShrinking() {
	var i int
	Gmax1 := -INF // max { -yI * grad(f)I | i in IUp(\alpha) }
	Gmax2 := -INF // max { yI * grad(f)I | i in ILow(\alpha) }
	// find maximal violating pair first
	for i = 0; i < m.activeSize; i++ {
		if m.y[i] == +1 {
			if !m.isUpperBound(i) {
				if -m.G[i] >= Gmax1 {
					Gmax1 = -m.G[i]
				}
			}
			if !m.isLowerBound(i) {
				if m.G[i] >= Gmax2 {
					Gmax2 = m.G[i]
				}
			}
		} else {
			if !m.isUpperBound(i) {
				if -m.G[i] >= Gmax2 {
					Gmax2 = -m.G[i]
				}
			}
			if !m.isLowerBound(i) {
				if m.G[i] >= Gmax1 {
					Gmax1 = m.G[i]
				}
			}
		}
	}

	if m.unshrink == false && Gmax1+Gmax2 <= m.eps*10 {
		m.unshrink = true
		m.reconstructGradient()
		m.activeSize = m.l
	}

	for i = 0; i < m.activeSize; i++ {
		if m.beShrunk(i, Gmax1, Gmax2) {
			m.activeSize--
			for m.activeSize > i {
				if !m.beShrunk(m.activeSize, Gmax1, Gmax2) {
					m.swapIndex(i, m.activeSize)
					break
				}
				m.activeSize--
			}
		}
	}
}

func (m *Solver) calculateRho() float64 {
	var r float64
	nrFree := 0
	ub := INF
	lb := -INF
	sumFree := float64(0)
	for i := 0; i < m.activeSize; i++ {
		yG := float64(m.y[i]) * m.G[i]

		if m.isLowerBound(i) {
			if m.y[i] > 0 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		} else if m.isUpperBound(i) {
			if m.y[i] < 0 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		} else {
			nrFree++
			sumFree += yG
		}
	}

	if nrFree > 0 {
		r = sumFree / float64(nrFree)
	} else {
		r = (ub + lb) / 2
	}
	return r
}

// SolverNU for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
type SolverNU struct {
	Solver
	si SolutionInfo
}

// Solve function for SolverNU
func (m *SolverNU) Solve(l int, Q QMatrix, p []float64, y []int8, alpha []float64, Cp, Cn, eps float64, si *SolutionInfo, shrinking int) {
	m.si = *si
	m.Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking)
}

func (m *SolverNU) selectWorkingSet(workingSet []int) int {
	// return i,j such that yI = yJ and
	// i: maximizes -yI * grad(f)I, i in IUp(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -yJ*grad(f)J < -yI*grad(f)I, j in ILow(\alpha)

	Gmaxp := -INF
	Gmaxp2 := -INF
	GmaxpIdx := -1

	Gmaxn := -INF
	Gmaxn2 := -INF
	GmaxnIdx := -1

	GminIdx := -1
	objDiffMin := INF

	for t := 0; t < m.activeSize; t++ {
		if m.y[t] == +1 {
			if !m.isUpperBound(t) {
				if -m.G[t] >= Gmaxp {
					Gmaxp = -m.G[t]
					GmaxpIdx = t
				}
			}
		} else {
			if !m.isLowerBound(t) {
				if m.G[t] >= Gmaxn {
					Gmaxn = m.G[t]
					GmaxnIdx = t
				}
			}
		}
	}

	ip := GmaxpIdx
	in := GmaxnIdx
	var QIp []float32
	QIp = nil
	var QIn []float32
	QIn = nil
	if ip != -1 { // null QIp not accessed: Gmaxp=-INF if ip=-1
		QIp = m.Q.getQ(ip, m.activeSize)
	}
	if in != -1 {
		QIn = m.Q.getQ(in, m.activeSize)
	}

	for j := 0; j < m.activeSize; j++ {
		if m.y[j] == +1 {
			if !m.isLowerBound(j) {
				gradDiff := Gmaxp + m.G[j]
				if m.G[j] >= Gmaxp2 {
					Gmaxp2 = m.G[j]
				}
				if gradDiff > 0 {
					var objDiff float64
					quadCoef := m.QD[ip] + m.QD[j] - 2*float64(QIp[j])
					if quadCoef > 0 {
						objDiff = -(gradDiff * gradDiff) / quadCoef
					} else {
						objDiff = -(gradDiff * gradDiff) / 1e-12
					}

					if objDiff <= objDiffMin {
						GminIdx = j
						objDiffMin = objDiff
					}
				}
			}
		} else {
			if !m.isUpperBound(j) {
				gradDiff := Gmaxn - m.G[j]
				if -m.G[j] >= Gmaxn2 {
					Gmaxn2 = -m.G[j]
				}
				if gradDiff > 0 {
					var objDiff float64
					quadCoef := m.QD[in] + m.QD[j] - 2*float64(QIn[j])
					if quadCoef > 0 {
						objDiff = -(gradDiff * gradDiff) / quadCoef
					} else {
						objDiff = -(gradDiff * gradDiff) / 1e-12
					}

					if objDiff <= objDiffMin {
						GminIdx = j
						objDiffMin = objDiff
					}
				}
			}
		}
	}

	if math.Max(Gmaxp+Gmaxp2, Gmaxn+Gmaxn2) < m.eps {
		return 1
	}

	if m.y[GminIdx] == +1 {
		workingSet[0] = GmaxpIdx
	} else {
		workingSet[0] = GmaxnIdx
	}
	workingSet[1] = GminIdx
	return 0
}

func (m *SolverNU) beShrunk(i int, Gmax1, Gmax2, Gmax3, Gmax4 float64) bool {
	var rst bool
	if m.isUpperBound(i) {
		if m.y[i] == +1 {
			rst = (-m.G[i] > Gmax1)
		} else {
			rst = (-m.G[i] > Gmax4)
		}
	} else if m.isLowerBound(i) {
		if m.y[i] == +1 {
			rst = (m.G[i] > Gmax2)
		} else {
			rst = (m.G[i] > Gmax3)
		}
	} else {
		rst = false
	}
	return rst
}

func (m *SolverNU) doShrinking() {
	Gmax1 := -INF // max { -yI * grad(f)I | yI = +1, i in IUp(\alpha) }
	Gmax2 := -INF // max { yI * grad(f)I | yI = +1, i in ILow(\alpha) }
	Gmax3 := -INF // max { -yI * grad(f)I | yI = -1, i in IUp(\alpha) }
	Gmax4 := -INF // max { yI * grad(f)I | yI = -1, i in ILow(\alpha) }

	// find maximal violating pair first
	var i int
	for i = 0; i < m.activeSize; i++ {
		if !m.isUpperBound(i) {
			if m.y[i] == +1 {
				if -m.G[i] > Gmax1 {
					Gmax1 = -m.G[i]
				}
			} else if -m.G[i] > Gmax4 {
				Gmax4 = -m.G[i]
			}
		}
		if !m.isLowerBound(i) {
			if m.y[i] == +1 {
				if m.G[i] > Gmax2 {
					Gmax2 = m.G[i]
				}
			} else if m.G[i] > Gmax3 {
				Gmax3 = m.G[i]
			}
		}
	}

	if m.unshrink == false && math.Max(Gmax1+Gmax2, Gmax3+Gmax4) <= m.eps*10 {
		m.unshrink = true
		m.reconstructGradient()
		m.activeSize = m.l
	}

	for i = 0; i < m.activeSize; i++ {
		if m.beShrunk(i, Gmax1, Gmax2, Gmax3, Gmax4) {
			m.activeSize--
			for m.activeSize > i {
				if !m.beShrunk(m.activeSize, Gmax1, Gmax2, Gmax3, Gmax4) {
					m.swapIndex(i, m.activeSize)
					break
				}
				m.activeSize--
			}
		}
	}
}

func (m *SolverNU) calculateRho() float64 {
	nrFree1 := 0
	nrFree2 := 0
	ub1 := INF
	ub2 := INF
	lb1 := -INF
	lb2 := -INF
	sumFree1 := float64(0)
	sumFree2 := float64(0)

	for i := 0; i < m.activeSize; i++ {
		if m.y[i] == +1 {
			if m.isLowerBound(i) {
				ub1 = math.Min(ub1, m.G[i])
			} else if m.isUpperBound(i) {
				lb1 = math.Max(lb1, m.G[i])
			} else {
				nrFree1++
				sumFree1 += m.G[i]
			}
		} else {
			if m.isLowerBound(i) {
				ub2 = math.Min(ub2, m.G[i])
			} else if m.isUpperBound(i) {
				lb2 = math.Max(lb2, m.G[i])
			} else {
				nrFree2++
				sumFree2 += m.G[i]
			}
		}
	}

	var r1, r2 float64
	if nrFree1 > 0 {
		r1 = sumFree1 / float64(nrFree1)
	} else {
		r1 = (ub1 + lb1) / 2
	}

	if nrFree2 > 0 {
		r2 = sumFree2 / float64(nrFree2)
	} else {
		r2 = (ub2 + lb2) / 2
	}

	m.si.r = (r1 + r2) / 2
	return (r1 - r2) / 2
}

// SVCQ struct
type SVCQ struct {
	y     []int8
	cache *Cache
	QD    []float64
	Kernel
}

// NewSVCQ create SVCQ struct
func NewSVCQ(prob *SVMProblem, param *SVMParameter, y []int8) *SVCQ {
	m := &SVCQ{
		Kernel: *NewKernel(prob.L, prob.X, param),
		y:      make([]int8, len(y)),
		cache:  NewCache(prob.L, int64(param.CacheSize*(1<<20))),
		QD:     make([]float64, prob.L),
	}
	copy(m.y, y)
	for i := 0; i < prob.L; i++ {
		m.QD[i] = m.kernelFunction(i, i)
	}
	return m
}
func (m *SVCQ) getQ(i, length int) []float32 {
	data := make([][]float32, 1)
	var start, j int
	if start = m.cache.getData(i, data, length); start < length {
		for j = start; j < length; j++ {
			data[0][j] = float32(m.y[i]*m.y[j]) * float32(m.kernelFunction(i, j))
		}
	}
	return data[0]
}

func (m *SVCQ) getQD() []float64 {
	return m.QD
}

func (m *SVCQ) swapIndex(i, j int) {
	m.cache.swapIndex(i, j)
	m.Kernel.swapIndex(i, j)
	//do {byte _=y[i]; y[i]=y[j]; y[j]=_;} while(false);
	m.y[i], m.y[j] = m.y[j], m.y[i]
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	m.QD[i], m.QD[j] = m.QD[j], m.QD[i]
}

// OneClassQ struct
type OneClassQ struct {
	Kernel
	cache *Cache
	QD    []float64
}

// NewOneClassQ create OneClassQ
func NewOneClassQ(prob *SVMProblem, param *SVMParameter) *OneClassQ {
	m := &OneClassQ{
		Kernel: *NewKernel(prob.L, prob.X, param),
		cache:  NewCache(prob.L, int64(param.CacheSize*(1<<20))),
		QD:     make([]float64, prob.L),
	}
	for i := 0; i < prob.L; i++ {
		m.QD[i] = m.kernelFunction(i, i)
	}
	return m
}
func (m *OneClassQ) getQ(i, length int) []float32 {
	data := make([][]float32, 1)
	var start, j int
	if start = m.cache.getData(i, data, length); start < length {
		for j = start; j < length; j++ {
			data[0][j] = float32(m.kernelFunction(i, j))
		}
	}
	return data[0]
}

func (m *OneClassQ) getQD() []float64 {
	return m.QD
}

func (m *OneClassQ) swapIndex(i, j int) {
	m.cache.swapIndex(i, j)
	m.Kernel.swapIndex(i, j)
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	m.QD[i], m.QD[j] = m.QD[j], m.QD[i]
}

// SVRQ struct
type SVRQ struct {
	Kernel
	l          int
	cache      *Cache
	sign       []int8
	index      []int
	nextBuffer int
	buffer     [][]float32
	QD         []float64
}

// NewSVRQ create SVRQ
func NewSVRQ(prob *SVMProblem, param *SVMParameter) *SVRQ {
	m := &SVRQ{
		Kernel: *NewKernel(prob.L, prob.X, param),
		l:      prob.L,
		cache:  NewCache(prob.L, int64(param.CacheSize*(1<<20))),
		QD:     make([]float64, 2*prob.L),
		sign:   make([]int8, 2*prob.L),
		index:  make([]int, 2*prob.L),
		buffer: make([][]float32, 2),
	}
	for k := 0; k < m.l; k++ {
		m.sign[k] = 1
		m.sign[k+m.l] = -1
		m.index[k] = k
		m.index[k+m.l] = k
		m.QD[k] = m.kernelFunction(k, k)
		m.QD[k+m.l] = m.QD[k]
	}
	for i := range m.buffer {
		m.buffer[i] = make([]float32, 2*m.l)
	}
	m.nextBuffer = 0
	return m
}
func (m *SVRQ) getQ(i, length int) []float32 {
	data := make([][]float32, 1)
	var j int
	realI := m.index[i]
	if m.cache.getData(realI, data, m.l) < m.l {
		for j = 0; j < m.l; j++ {
			data[0][j] = float32(m.kernelFunction(realI, j))
		}
	}

	// reorder and copy
	buf := m.buffer[m.nextBuffer]
	m.nextBuffer = 1 - m.nextBuffer
	si := m.sign[i]
	for j = 0; j < length; j++ {
		buf[j] = float32(si) * float32(m.sign[j]) * data[0][m.index[j]]
	}
	return buf
}

func (m *SVRQ) getQD() []float64 {
	return m.QD
}

func (m *SVRQ) swapIndex(i, j int) {
	//do {byte _=sign[i]; sign[i]=sign[j]; sign[j]=_;} while(false);
	m.sign[i], m.sign[j] = m.sign[j], m.sign[i]
	//do {int _=index[i]; index[i]=index[j]; index[j]=_;} while(false);
	m.index[i], m.index[j] = m.index[j], m.index[i]
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	m.QD[i], m.QD[j] = m.QD[j], m.QD[i]
}

// LibsvmVersion is libsvm version
const LibsvmVersion = 312

type decisionFunction struct {
	alpha []float64
	rho   float64
}

// SVM struct
type SVM struct {
	rand *rand.Rand

	//
	// decisionFunction
	//
	svmTypeTable []string

	kernelTypeTable []string
}

// NewSvm create SVM
func NewSvm() *SVM {
	m := new(SVM)
	m.rand = new(rand.Rand)
	m.svmTypeTable = []string{"cSvc", "nuSvc", "oneClass", "epsilonSvr", "nuSvr"}
	m.kernelTypeTable = []string{"linear", "polynomial", "rbf", "sigmoid", "precomputed"}
	return m
}

// SolveCSvc for SVM
func (m *SVM) SolveCSvc(prob *SVMProblem, param *SVMParameter, alpha []float64, si *SolutionInfo, Cp, Cn float64) {
	l := prob.L
	minusOnes := make([]float64, l)
	y := make([]int8, l)

	var i int

	for i = 0; i < l; i++ {
		alpha[i] = 0
		minusOnes[i] = -1
		if prob.Y[i] > 0 {
			y[i] = +1
		} else {
			y[i] = -1
		}
	}

	s := new(Solver)
	s.Solve(l, NewSVCQ(prob, param, y), minusOnes, y,
		alpha, Cp, Cn, param.Eps, si, param.Shrinking)

	sumAlpha := float64(0)
	for i = 0; i < l; i++ {
		sumAlpha += alpha[i]
	}

	if Cp == Cn {
		log.Println("nu = ", sumAlpha/(Cp*float64(prob.L)))
	}

	for i = 0; i < l; i++ {
		alpha[i] *= float64(y[i])
	}
}

// SolveNuSvc for SVM
func (m *SVM) SolveNuSvc(prob *SVMProblem, param *SVMParameter, alpha []float64, si *SolutionInfo) {
	var i int
	l := prob.L
	nu := param.Nu
	y := make([]int8, l)

	for i = 0; i < l; i++ {
		if prob.Y[i] > 0 {
			y[i] = +1
		} else {
			y[i] = -1
		}
	}

	sumPos := nu * float64(l) / 2
	sumNeg := nu * float64(l) / 2

	for i = 0; i < l; i++ {
		if y[i] == +1 {
			alpha[i] = math.Min(1.0, sumPos)
			sumPos -= alpha[i]
		} else {
			alpha[i] = math.Min(1.0, sumNeg)
			sumNeg -= alpha[i]
		}
	}

	zeros := make([]float64, l)

	for i = 0; i < l; i++ {
		zeros[i] = 0
	}

	s := new(SolverNU)
	s.Solve(l, NewSVCQ(prob, param, y), zeros, y,
		alpha, 1.0, 1.0, param.Eps, si, param.Shrinking)
	r := si.r

	log.Println("C = ", 1/r)

	for i = 0; i < l; i++ {
		alpha[i] *= float64(y[i]) / r
	}

	si.rho /= r
	si.obj /= (r * r)
	si.upperBoundP = 1 / r
	si.upperBoundN = 1 / r
}

// SolveOneClass for svm
func (m *SVM) SolveOneClass(prob *SVMProblem, param *SVMParameter, alpha []float64, si *SolutionInfo) {
	l := prob.L
	zeros := make([]float64, l)
	ones := make([]int8, l)
	var i int

	n := int(param.Nu) * prob.L // # of alpha's at upper bound

	for i = 0; i < n; i++ {
		alpha[i] = 1
	}
	if n < prob.L {
		alpha[n] = param.Nu*float64(prob.L) - float64(n)
	}
	for i = n + 1; i < l; i++ {
		alpha[i] = 0
	}

	for i = 0; i < l; i++ {
		zeros[i] = 0
		ones[i] = 1
	}

	s := new(Solver)
	s.Solve(l, NewOneClassQ(prob, param), zeros, ones,
		alpha, 1.0, 1.0, param.Eps, si, param.Shrinking)
}

// SolveEpsilonSvr for svm
func (m *SVM) SolveEpsilonSvr(prob *SVMProblem, param *SVMParameter, alpha []float64, si *SolutionInfo) {
	l := prob.L
	alpha2 := make([]float64, 2*l)
	linearTerm := make([]float64, 2*l)
	y := make([]int8, 2*l)
	var i int

	for i = 0; i < l; i++ {
		alpha2[i] = 0
		linearTerm[i] = param.P - prob.Y[i]
		y[i] = 1

		alpha2[i+l] = 0
		linearTerm[i+l] = param.P + prob.Y[i]
		y[i+l] = -1
	}

	s := new(Solver)
	s.Solve(2*l, NewSVRQ(prob, param), linearTerm, y,
		alpha2, param.C, param.C, param.Eps, si, param.Shrinking)
	sumAlpha := float64(0)
	for i = 0; i < l; i++ {
		alpha[i] = alpha2[i] - alpha2[i+l]
		sumAlpha += math.Abs(alpha[i])
	}
	log.Println("nu = ", sumAlpha/(param.C*float64(l)))
}

// SolveNuSvr for svm
func (m *SVM) SolveNuSvr(prob *SVMProblem, param *SVMParameter, alpha []float64, si *SolutionInfo) {
	l := prob.L
	C := param.C
	alpha2 := make([]float64, 2*l)
	linearTerm := make([]float64, 2*l)
	y := make([]int8, 2*l)
	var i int

	sum := C * param.Nu * float64(l) / 2
	for i = 0; i < l; i++ {
		alpha2[i+l] = math.Min(sum, C)
		alpha2[i] = alpha2[i+1]
		sum -= alpha2[i]

		linearTerm[i] = -prob.Y[i]
		y[i] = 1

		linearTerm[i+l] = prob.Y[i]
		y[i+l] = -1
	}

	s := new(SolverNU)
	s.Solve(2*l, NewSVRQ(prob, param), linearTerm, y,
		alpha2, C, C, param.Eps, si, param.Shrinking)

	log.Println("epsilon = ", -si.r)

	for i = 0; i < l; i++ {
		alpha[i] = alpha2[i] - alpha2[i+l]
	}
}

// SVMTrainOne for svm
func (m *SVM) SVMTrainOne(prob *SVMProblem, param *SVMParameter, Cp, Cn float64) *decisionFunction {
	alpha := make([]float64, prob.L)
	si := new(SolutionInfo)
	switch param.SvmType {
	case CSVC:
		m.SolveCSvc(prob, param, alpha, si, Cp, Cn)
	case NUSVC:
		m.SolveNuSvc(prob, param, alpha, si)
	case ONECLASS:
		m.SolveOneClass(prob, param, alpha, si)
	case EPSILONSVR:
		m.SolveEpsilonSvr(prob, param, alpha, si)
	case NUSVR:
		m.SolveNuSvr(prob, param, alpha, si)
	}

	log.Println("obj = ", si.obj, ", rho = ", si.rho)

	// output SVs

	nSV := 0
	nBSV := 0
	for i := 0; i < prob.L; i++ {
		if math.Abs(alpha[i]) > 0 {
			nSV++
			if prob.Y[i] > 0 {
				if math.Abs(alpha[i]) >= si.upperBoundP {
					nBSV++
				}
			} else {
				if math.Abs(alpha[i]) >= si.upperBoundN {
					nBSV++
				}
			}
		}
	}

	log.Println("nSV = ", nSV, ", nBSV = ", nBSV)

	f := new(decisionFunction)
	f.alpha = alpha
	f.rho = si.rho
	return f
}

func (m *SVM) sigmoidTrain(l int, decValues, labels, probAB []float64) {
	var A, B float64
	prior1 := float64(0)
	prior0 := float64(0)
	var i int

	for i = 0; i < l; i++ {
		if labels[i] > 0 {
			prior1 += 1
		} else {
			prior0 += 1
		}
	}

	maxIter := 100            // Maximal number of iterations
	minStep := float64(1e-10) // Minimal step taken in line search
	sigma := float64(1e-12)   // For numerically strict PD of Hessian
	eps := float64(1e-5)
	hiTarget := float64((prior1 + 1.0) / (prior1 + 2.0))
	loTarget := float64(1 / (prior0 + 2.0))
	t := make([]float64, l)
	var fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize float64
	var newA, newB, newf, d1, d2 float64
	var iter int

	// Initial Point and Initial Fun Value
	A = 0.0
	B = math.Log((prior0 + 1.0) / (prior1 + 1.0))
	fval := float64(0.0)

	for i = 0; i < l; i++ {
		if labels[i] > 0 {
			t[i] = hiTarget
		} else {
			t[i] = loTarget
		}
		fApB = decValues[i]*A + B
		if fApB >= 0 {
			fval += t[i]*fApB + math.Log(1+math.Exp(-fApB))
		} else {
			fval += (t[i]-1)*fApB + math.Log(1+math.Exp(fApB))
		}
	}
	for iter = 0; iter < maxIter; iter++ {
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11 = sigma // numerically ensures strict PD
		h22 = sigma
		h21 = 0.0
		g1 = 0.0
		g2 = 0.0
		for i = 0; i < l; i++ {
			fApB = decValues[i]*A + B
			if fApB >= 0 {
				p = math.Exp(-fApB) / (1.0 + math.Exp(-fApB))
				q = 1.0 / (1.0 + math.Exp(-fApB))
			} else {
				p = 1.0 / (1.0 + math.Exp(fApB))
				q = math.Exp(fApB) / (1.0 + math.Exp(fApB))
			}
			d2 = p * q
			h11 += decValues[i] * decValues[i] * d2
			h22 += d2
			h21 += decValues[i] * d2
			d1 = t[i] - p
			g1 += decValues[i] * d1
			g2 += d1
		}

		// Stopping Criteria
		if math.Abs(g1) < eps && math.Abs(g2) < eps {
			break
		}

		// Finding Newton direction: -inv(H') * g
		det = h11*h22 - h21*h21
		dA = -(h22*g1 - h21*g2) / det
		dB = -(-h21*g1 + h11*g2) / det
		gd = g1*dA + g2*dB

		stepsize = 1 // Line Search
		for stepsize >= minStep {
			newA = A + stepsize*dA
			newB = B + stepsize*dB

			// New function value
			newf = 0.0
			for i = 0; i < l; i++ {
				fApB = decValues[i]*newA + newB
				if fApB >= 0 {
					newf += t[i]*fApB + math.Log(1+math.Exp(-fApB))
				} else {
					newf += (t[i]-1)*fApB + math.Log(1+math.Exp(fApB))
				}
			}
			// Check sufficient decrease
			if newf < fval+0.0001*stepsize*gd {
				A = newA
				B = newB
				fval = newf
				break
			} else {
				stepsize = stepsize / 2.0
			}
		}

		if stepsize < minStep {
			log.Println("Line search fails in two-class probability estimates")
			break
		}
	}

	if iter >= maxIter {
		log.Println("Reaching maximal iterations in two-class probability estimates")
	}
	probAB[0] = A
	probAB[1] = B
}

func (m *SVM) sigmoidPredict(decisionValue, A, B float64) float64 {
	var rst float64
	fApB := decisionValue*A + B
	if fApB >= 0 {
		rst = math.Exp(-fApB) / (1.0 + math.Exp(-fApB))
	} else {
		rst = 1.0 / (1 + math.Exp(fApB))
	}
	return rst
}

func (m *SVM) multiclassProbability(k int, r [][]float64, p []float64) {
	var t, j int
	iter := 0
	maxIter := math.Max(100, float64(k))
	Q := make([][]float64, k)
	for i := range Q {
		Q[i] = make([]float64, k)
	}
	Qp := make([]float64, k)
	var pQp float64
	eps := float64(0.005) / float64(k)

	for t = 0; t < k; t++ {
		p[t] = float64(1.0) / float64(k) // Valid if k = 1
		Q[t][t] = 0
		for j = 0; j < t; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = Q[j][t]
		}
		for j = t + 1; j < k; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = -r[j][t] * r[t][j]
		}
	}
	for iter = 0; iter < int(maxIter); iter++ {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0
		for t = 0; t < k; t++ {
			Qp[t] = 0
			for j = 0; j < k; j++ {
				Qp[t] += Q[t][j] * p[j]
			}
			pQp += p[t] * Qp[t]
		}
		maxError := float64(0)
		for t = 0; t < k; t++ {
			error := math.Abs(Qp[t] - pQp)
			if error > maxError {
				maxError = error
			}
		}
		if maxError < eps {
			break
		}

		for t = 0; t < k; t++ {
			diff := (-Qp[t] + pQp) / Q[t][t]
			p[t] += diff
			pQp = (pQp + diff*(diff*Q[t][t]+2*Qp[t])) / (1 + diff) / (1 + diff)
			for j = 0; j < k; j++ {
				Qp[j] = (Qp[j] + diff*Q[t][j]) / (1 + diff)
				p[j] /= (1 + diff)
			}
		}
	}
	if iter >= int(maxIter) {
		log.Println("Exceeds maxIter in multiclassProb")
	}
}

// SVMBinarySvcProbability for svm
func (m *SVM) SVMBinarySvcProbability(prob *SVMProblem, param *SVMParameter, Cp, Cn float64, probAB []float64) {
	var i int
	nrFold := 5
	perm := make([]int, prob.L)
	decValues := make([]float64, prob.L)

	// random shuffle
	for i = 0; i < prob.L; i++ {
		perm[i] = i
	}
	for i = 0; i < prob.L; i++ {
		j := i + int(rand.Int31n(int32(prob.L-i)))
		//do {int _=perm[i]; perm[i]=perm[j]; perm[j]=_;} while(false);
		perm[i], perm[j] = perm[j], perm[i]
	}
	for i = 0; i < nrFold; i++ {
		begin := i * prob.L / nrFold
		end := (i + 1) * prob.L / nrFold
		var j, k int
		subprob := new(SVMProblem)

		subprob.L = prob.L - (end - begin)
		subprob.X = make([][]SVMNode, subprob.L)
		subprob.Y = make([]float64, subprob.L)

		k = 0
		for j = 0; j < begin; j++ {
			subprob.X[k] = prob.X[perm[j]]
			subprob.Y[k] = prob.Y[perm[j]]
			k++
		}
		for j = end; j < prob.L; j++ {
			subprob.X[k] = prob.X[perm[j]]
			subprob.Y[k] = prob.Y[perm[j]]
			k++
		}
		pCount := 0
		nCount := 0
		for j = 0; j < k; j++ {
			if subprob.Y[j] > 0 {
				pCount++
			} else {
				nCount++
			}
		}
		if pCount == 0 && nCount == 0 {
			for j = begin; j < end; j++ {
				decValues[perm[j]] = 0
			}
		} else if pCount > 0 && nCount == 0 {
			for j = begin; j < end; j++ {
				decValues[perm[j]] = 1
			}
		} else if pCount == 0 && nCount > 0 {
			for j = begin; j < end; j++ {
				decValues[perm[j]] = -1
			}
		} else {
			subparam := param.Clone()
			subparam.Probability = 0
			subparam.C = 1.0
			subparam.NrWeight = 2
			subparam.WeightLabel = make([]int, 2)
			subparam.Weight = make([]float64, 2)
			subparam.WeightLabel[0] = +1
			subparam.WeightLabel[1] = -1
			subparam.Weight[0] = Cp
			subparam.Weight[1] = Cn
			submodel := m.SVMTrain(subprob, subparam)
			for j = begin; j < end; j++ {
				decValue := make([]float64, 1)
				m.SVMPredictValues(submodel, prob.X[perm[j]], decValue)
				decValues[perm[j]] = decValue[0]
				// ensure +1 -1 order; reason not using CV subroutine
				decValues[perm[j]] *= float64(submodel.Label[0])
			}
		}
	}
	m.sigmoidTrain(prob.L, decValues, prob.Y, probAB)
}

// SVMSvrProbability for svm
func (m *SVM) SVMSvrProbability(prob *SVMProblem, param *SVMParameter) float64 {
	var i int
	nrFold := 5
	ymv := make([]float64, prob.L)
	mae := float64(0)

	newparam := param.Clone()
	newparam.Probability = 0
	m.SVMCrossValidation(prob, newparam, nrFold, ymv)
	for i = 0; i < prob.L; i++ {
		ymv[i] = prob.Y[i] - ymv[i]
		mae += math.Abs(ymv[i])
	}
	mae /= float64(prob.L)
	std := math.Sqrt(2 * mae * mae)
	count := 0
	mae = 0
	for i = 0; i < prob.L; i++ {
		if math.Abs(ymv[i]) > 5*std {
			count = count + 1
		} else {
			mae += math.Abs(ymv[i])
		}
	}
	mae /= float64(prob.L - count)
	log.Println("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=", mae)
	return mae
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling m subroutine

// SVMGroupClasses for svm
func (m *SVM) SVMGroupClasses(prob *SVMProblem, nrClassRet []int, labelRet, startRet, countRet [][]int, perm []int) {
	l := prob.L
	maxNrClass := 16
	nrClass := 0
	label := make([]int, maxNrClass)
	count := make([]int, maxNrClass)
	dataLabel := make([]int, l)
	var i int

	for i = 0; i < l; i++ {
		mLabel := int(prob.Y[i])
		var j int
		for j = 0; j < nrClass; j++ {
			if mLabel == label[j] {
				count[j]++
				break
			}
		}
		dataLabel[i] = j
		if j == nrClass {
			if nrClass == maxNrClass {
				maxNrClass *= 2
				newData := make([]int, maxNrClass)
				//System.arraycopy(label,0,newData,0,label.length);
				copy(newData, label)
				label = newData
				newData = make([]int, maxNrClass)
				//System.arraycopy(count,0,newData,0,count.length);
				copy(newData, count)
				count = newData
			}
			label[nrClass] = mLabel
			count[nrClass] = 1
			nrClass++
		}
	}

	start := make([]int, nrClass)
	start[0] = 0
	for i = 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}
	for i = 0; i < l; i++ {
		perm[start[dataLabel[i]]] = i
		start[dataLabel[i]]++
	}
	start[0] = 0
	for i = 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}

	nrClassRet[0] = nrClass
	labelRet[0] = label
	startRet[0] = start
	countRet[0] = count
}

// SVMTrain for svm
func (m *SVM) SVMTrain(prob *SVMProblem, param *SVMParameter) *SVMModel {

	model := new(SVMModel)
	model.Param = param

	if param.SvmType == ONECLASS || param.SvmType == EPSILONSVR || param.SvmType == NUSVR {
		// regression or one-class-svm
		model.NrClass = 2
		model.Label = nil
		model.NSV = nil
		model.ProbA = nil
		model.ProbB = nil
		model.SvCoef = make([][]float64, 1)

		if param.Probability == 1 && (param.SvmType == EPSILONSVR || param.SvmType == NUSVR) {
			model.ProbA = make([]float64, 1)
			model.ProbA[0] = m.SVMSvrProbability(prob, param)
		}

		f := m.SVMTrainOne(prob, param, 0, 0)
		model.Rho = make([]float64, 1)
		model.Rho[0] = f.rho

		nSV := 0
		var i int
		for i = 0; i < prob.L; i++ {
			if math.Abs(f.alpha[i]) > 0 {
				nSV++
			}
		}
		model.L = nSV
		model.SV = make([][]SVMNode, nSV)
		model.SvCoef[0] = make([]float64, nSV)
		model.SvIndices = make([]int, nSV)
		j := 0
		for i = 0; i < prob.L; i++ {
			if math.Abs(f.alpha[i]) > 0 {
				model.SV[j] = prob.X[i]
				model.SvCoef[0][j] = f.alpha[i]
				model.SvIndices[j] = i + 1
				j++
			}
		}
	} else {
		// classification
		l := prob.L
		tmpNrClass := make([]int, 1)
		tmpLabel := make([][]int, 1)
		tmpStart := make([][]int, 1)
		tmpCount := make([][]int, 1)
		perm := make([]int, l)

		// group training data of the same class
		m.SVMGroupClasses(prob, tmpNrClass, tmpLabel, tmpStart, tmpCount, perm)
		nrClass := tmpNrClass[0]
		label := tmpLabel[0]
		start := tmpStart[0]
		count := tmpCount[0]

		if nrClass == 1 {
			log.Println("WARNING: training data in only one class. See README for details.")
		}

		x := make([][]SVMNode, l)
		var i int
		for i = 0; i < l; i++ {
			x[i] = prob.X[perm[i]]
		}

		// calculate weighted C

		weightedC := make([]float64, nrClass)
		for i = 0; i < nrClass; i++ {
			weightedC[i] = param.C
		}
		for i = 0; i < param.NrWeight; i++ {
			var j int
			for j = 0; j < nrClass; j++ {
				if param.WeightLabel[i] == label[j] {
					break
				}
			}
			if j == nrClass {
				log.Printf("WARNING: class label %d specified in weight is not found\n", param.WeightLabel[i])
			} else {
				weightedC[j] *= param.Weight[i]
			}
		}

		// train k*(k-1)/2 models

		nonzero := make([]bool, l)
		for i = 0; i < l; i++ {
			nonzero[i] = false
		}
		f := make([]decisionFunction, nrClass*(nrClass-1)/2)

		var probA, probB []float64
		probA = nil
		probB = nil
		if param.Probability == 1 {
			probA = make([]float64, nrClass*(nrClass-1)/2)
			probB = make([]float64, nrClass*(nrClass-1)/2)
		}

		p := 0
		for i = 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				subProb := new(SVMProblem)
				si := start[i]
				sj := start[j]
				ci := count[i]
				cj := count[j]
				subProb.L = ci + cj
				subProb.X = make([][]SVMNode, subProb.L)
				subProb.Y = make([]float64, subProb.L)
				var k int
				for k = 0; k < ci; k++ {
					subProb.X[k] = x[si+k]
					subProb.Y[k] = +1
				}
				for k = 0; k < cj; k++ {
					subProb.X[ci+k] = x[sj+k]
					subProb.Y[ci+k] = -1
				}

				if param.Probability == 1 {
					probAB := make([]float64, 2)
					m.SVMBinarySvcProbability(subProb, param, weightedC[i], weightedC[j], probAB)
					probA[p] = probAB[0]
					probB[p] = probAB[1]
				}

				f[p] = *m.SVMTrainOne(subProb, param, weightedC[i], weightedC[j])
				for k = 0; k < ci; k++ {
					if !nonzero[si+k] && math.Abs(f[p].alpha[k]) > 0 {
						nonzero[si+k] = true
					}
				}
				for k = 0; k < cj; k++ {
					if !nonzero[sj+k] && math.Abs(f[p].alpha[ci+k]) > 0 {
						nonzero[sj+k] = true
					}
				}
				p++
			}
		}

		// build output

		model.NrClass = nrClass

		model.Label = make([]int, nrClass)
		for i = 0; i < nrClass; i++ {
			model.Label[i] = label[i]
		}

		model.Rho = make([]float64, nrClass*(nrClass-1)/2)
		for i = 0; i < nrClass*(nrClass-1)/2; i++ {
			model.Rho[i] = f[i].rho
		}

		if param.Probability == 1 {
			model.ProbA = make([]float64, nrClass*(nrClass-1)/2)
			model.ProbB = make([]float64, nrClass*(nrClass-1)/2)
			for i = 0; i < nrClass*(nrClass-1)/2; i++ {
				model.ProbA[i] = probA[i]
				model.ProbB[i] = probB[i]
			}
		} else {
			model.ProbA = nil
			model.ProbB = nil
		}

		nnz := 0
		nzCount := make([]int, nrClass)
		model.NSV = make([]int, nrClass)
		for i = 0; i < nrClass; i++ {
			nSV := 0
			for j := 0; j < count[i]; j++ {
				if nonzero[start[i]+j] {
					nSV++
					nnz++
				}
			}
			model.NSV[i] = nSV
			nzCount[i] = nSV
		}

		log.Println("Total nSV = ", nnz)

		model.L = nnz
		model.SV = make([][]SVMNode, nnz)
		model.SvIndices = make([]int, nnz)
		p = 0
		for i = 0; i < l; i++ {
			if nonzero[i] {
				model.SV[p] = x[i]
				model.SvIndices[p] = perm[i] + 1
				p++
			}
		}

		nzStart := make([]int, nrClass)
		nzStart[0] = 0
		for i = 1; i < nrClass; i++ {
			nzStart[i] = nzStart[i-1] + nzCount[i-1]
		}

		model.SvCoef = make([][]float64, nrClass-1)
		for i = 0; i < nrClass-1; i++ {
			model.SvCoef[i] = make([]float64, nnz)
		}

		p = 0
		for i = 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				// classifier (i,j): coefficients with
				// i are in svCoef[j-1][nzStart[i]...],
				// j are in svCoef[i][nzStart[j]...]

				si := start[i]
				sj := start[j]
				ci := count[i]
				cj := count[j]

				q := nzStart[i]
				var k int
				for k = 0; k < ci; k++ {
					if nonzero[si+k] {
						model.SvCoef[j-1][q] = f[p].alpha[k]
						q++
					}
				}
				q = nzStart[j]
				for k = 0; k < cj; k++ {
					if nonzero[sj+k] {
						model.SvCoef[i][q] = f[p].alpha[ci+k]
						q++
					}
				}
				p++
			}
		}
	}
	return model
}

// SVMCrossValidation for svm
func (m *SVM) SVMCrossValidation(prob *SVMProblem, param *SVMParameter, nrFold int, target []float64) {
	var i int
	foldStart := make([]int, nrFold+1)
	l := prob.L
	perm := make([]int, l)

	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if (param.SvmType == CSVC ||
		param.SvmType == NUSVC) && nrFold < l {
		tmpNrClass := make([]int, 1)
		tmpLabel := make([][]int, 1)
		tmpStart := make([][]int, 1)
		tmpCount := make([][]int, 1)

		m.SVMGroupClasses(prob, tmpNrClass, tmpLabel, tmpStart, tmpCount, perm)

		nrClass := tmpNrClass[0]
		start := tmpStart[0]
		count := tmpCount[0]

		// random shuffle and then data grouped by fold using the array perm
		foldCount := make([]int, nrFold)
		var c int
		index := make([]int, l)
		for i = 0; i < l; i++ {
			index[i] = perm[i]
		}
		for c = 0; c < nrClass; c++ {
			for i = 0; i < count[c]; i++ {
				j := i + int(rand.Int31n(int32(count[c]-i)))
				//do {int _=index[start[c]+j]; index[start[c]+j]=index[start[c]+i]; index[start[c]+i]=_;} while(false);
				index[start[c]+i], index[start[c]+j] = index[start[c]+j], index[start[c]+i]
			}
		}
		for i = 0; i < nrFold; i++ {
			foldCount[i] = 0
			for c = 0; c < nrClass; c++ {
				foldCount[i] += (i+1)*count[c]/nrFold - i*count[c]/nrFold
			}
		}
		foldStart[0] = 0
		for i = 1; i <= nrFold; i++ {
			foldStart[i] = foldStart[i-1] + foldCount[i-1]
		}
		for c = 0; c < nrClass; c++ {
			for i = 0; i < nrFold; i++ {
				begin := start[c] + i*count[c]/nrFold
				end := start[c] + (i+1)*count[c]/nrFold
				for j := begin; j < end; j++ {
					perm[foldStart[i]] = index[j]
					foldStart[i]++
				}
			}
		}
		foldStart[0] = 0
		for i = 1; i <= nrFold; i++ {
			foldStart[i] = foldStart[i-1] + foldCount[i-1]
		}
	} else {
		for i = 0; i < l; i++ {
			perm[i] = i
		}
		for i = 0; i < l; i++ {
			j := i + int(rand.Int31n(int32(l-i)))
			//do {int _=perm[i]; perm[i]=perm[j]; perm[j]=_;} while(false);
			perm[i], perm[j] = perm[j], perm[i]
		}
		for i = 0; i <= nrFold; i++ {
			foldStart[i] = i * l / nrFold
		}
	}

	for i = 0; i < nrFold; i++ {
		begin := foldStart[i]
		end := foldStart[i+1]
		var j, k int
		subprob := new(SVMProblem)

		subprob.L = l - (end - begin)
		subprob.X = make([][]SVMNode, subprob.L)
		subprob.Y = make([]float64, subprob.L)

		k = 0
		for j = 0; j < begin; j++ {
			subprob.X[k] = prob.X[perm[j]]
			subprob.Y[k] = prob.Y[perm[j]]
			k++
		}
		for j = end; j < l; j++ {
			subprob.X[k] = prob.X[perm[j]]
			subprob.Y[k] = prob.Y[perm[j]]
			k++
		}
		submodel := m.SVMTrain(subprob, param)
		if param.Probability == 1 &&
			(param.SvmType == CSVC ||
				param.SvmType == NUSVC) {
			probEstimates := make([]float64, m.SVMGetNrClass(submodel))
			for j = begin; j < end; j++ {
				target[perm[j]] = m.SVMPredictProbability(submodel, prob.X[perm[j]], probEstimates)
			}
		} else {
			for j = begin; j < end; j++ {
				target[perm[j]] = m.SVMPredict(submodel, prob.X[perm[j]])
			}
		}
	}
}

// SVMGetSvmType return svm type
func (m *SVM) SVMGetSvmType(model *SVMModel) int {
	return model.Param.SvmType
}

// SVMGetNrClass return nrclass
func (m *SVM) SVMGetNrClass(model *SVMModel) int {
	return model.NrClass
}

// SVMGetLabels return svm's labels
func (m *SVM) SVMGetLabels(model *SVMModel, label []int) {
	if model.Label != nil {
		for i := 0; i < model.NrClass; i++ {
			label[i] = model.Label[i]
		}
	}
}

// SVMGetSvIndices return svm's indices
func (m *SVM) SVMGetSvIndices(model SVMModel, indices []int) {
	if model.SvIndices != nil {
		for i := 0; i < model.L; i++ {
			indices[i] = model.SvIndices[i]
		}
	}
}

// SVMGetNrSv return svm's nrsv
func (m *SVM) SVMGetNrSv(model SVMModel) int {
	return model.L
}

// SVMGetSvrProbability return probability of svm
func (m *SVM) SVMGetSvrProbability(model *SVMModel) float64 {
	var rst float64
	if (model.Param.SvmType == EPSILONSVR || model.Param.SvmType == NUSVR) && model.ProbA != nil {
		rst = model.ProbA[0]
	} else {
		log.Print("Model doesn't contain information for SVR probability inference\n")
		rst = 0
	}
	return rst
}

// SVMPredictValues return perdict values
func (m *SVM) SVMPredictValues(model *SVMModel, x []SVMNode, decValues []float64) float64 {
	var i int
	var rst float64
	if model.Param.SvmType == ONECLASS ||
		model.Param.SvmType == EPSILONSVR ||
		model.Param.SvmType == NUSVR {
		svCoef := model.SvCoef[0]
		sum := float64(0)
		for i = 0; i < model.L; i++ {
			sum += svCoef[i] * kFunction(x, model.SV[i], model.Param)
		}
		sum -= model.Rho[0]
		decValues[0] = sum

		if model.Param.SvmType == ONECLASS {
			if sum > 0 {
				rst = 1
			} else {
				rst = -1
			}
		} else {
			rst = sum
		}
	} else {
		nrClass := model.NrClass
		l := model.L

		kvalue := make([]float64, l)
		for i = 0; i < l; i++ {
			kvalue[i] = kFunction(x, model.SV[i], model.Param)
		}

		start := make([]int, nrClass)
		start[0] = 0
		for i = 1; i < nrClass; i++ {
			start[i] = start[i-1] + model.NSV[i-1]
		}

		vote := make([]int, nrClass)
		for i = 0; i < nrClass; i++ {
			vote[i] = 0
		}

		p := 0
		for i = 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				sum := float64(0)
				si := start[i]
				sj := start[j]
				ci := model.NSV[i]
				cj := model.NSV[j]

				var k int
				coef1 := model.SvCoef[j-1]
				coef2 := model.SvCoef[i]
				for k = 0; k < ci; k++ {
					sum += coef1[si+k] * kvalue[si+k]
				}
				for k = 0; k < cj; k++ {
					sum += coef2[sj+k] * kvalue[sj+k]
				}
				sum -= model.Rho[p]
				decValues[p] = sum

				if decValues[p] > 0 {
					vote[i]++
				} else {
					vote[j]++
				}
				p++
			}
		}

		voteMaxIdx := 0
		for i = 1; i < nrClass; i++ {
			if vote[i] > vote[voteMaxIdx] {
				voteMaxIdx = i
			}
		}
		rst = float64(model.Label[voteMaxIdx])
	}
	return rst
}

// SVMPredict function
func (m *SVM) SVMPredict(model *SVMModel, x []SVMNode) float64 {
	nrClass := model.NrClass
	var decValues []float64
	if model.Param.SvmType == ONECLASS ||
		model.Param.SvmType == EPSILONSVR ||
		model.Param.SvmType == NUSVR {
		decValues = make([]float64, 1)
	} else {
		decValues = make([]float64, nrClass*(nrClass-1)/2)
	}
	predResult := m.SVMPredictValues(model, x, decValues)
	return predResult
}

// SVMPredictProbability function
func (m *SVM) SVMPredictProbability(model *SVMModel, x []SVMNode, probEstimates []float64) float64 {
	var rst float64
	if (model.Param.SvmType == CSVC ||
		model.Param.SvmType == NUSVC) &&
		model.ProbA != nil && model.ProbB != nil {
		var i int
		nrClass := model.NrClass
		decValues := make([]float64, nrClass*(nrClass-1)/2)
		m.SVMPredictValues(model, x, decValues)

		minProb := float64(1e-7)
		pairwiseProb := make([][]float64, nrClass)
		for i = range pairwiseProb {
			pairwiseProb[i] = make([]float64, nrClass)
		}

		k := 0
		for i = 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				pairwiseProb[i][j] = math.Min(math.Max(m.sigmoidPredict(decValues[k], model.ProbA[k], model.ProbB[k]), minProb), 1-minProb)
				pairwiseProb[j][i] = 1 - pairwiseProb[i][j]
				k++
			}
		}
		m.multiclassProbability(nrClass, pairwiseProb, probEstimates)

		probMaxIdx := 0
		for i = 1; i < nrClass; i++ {
			if probEstimates[i] > probEstimates[probMaxIdx] {
				probMaxIdx = i
			}
		}
		rst = float64(model.Label[probMaxIdx])
	} else {
		rst = m.SVMPredict(model, x)
	}
	return rst
}

// SVMSaveModel write svm to file
func (m *SVM) SVMSaveModel(modelFileName string, model *SVMModel) {

	fp, _ := os.OpenFile(modelFileName, os.O_CREATE|os.O_WRONLY|os.O_SYNC, 0644)
	defer fp.Close()
	fb := bufio.NewWriter(fp)
	param := model.Param

	fb.WriteString("svmType " + m.svmTypeTable[param.SvmType] + "\n")
	fb.WriteString("kernelType " + m.kernelTypeTable[param.KernelType] + "\n")

	if param.KernelType == POLY {
		fb.WriteString("degree " + strconv.Itoa(param.Degree) + "\n")
	}

	if param.KernelType == POLY ||
		param.KernelType == RBF ||
		param.KernelType == SIGMOID {
		fb.WriteString("gamma " + strconv.FormatFloat(param.Gamma, 'g', -1, 64) + "\n")
	}

	if param.KernelType == POLY ||
		param.KernelType == SIGMOID {
		fb.WriteString("coef0 " + strconv.FormatFloat(param.Coef0, 'g', -1, 64) + "\n")
	}

	nrClass := model.NrClass
	l := model.L
	fb.WriteString("nrClass " + strconv.Itoa(nrClass) + "\n")
	fb.WriteString("totalSv " + strconv.Itoa(l) + "\n")

	{
		fb.WriteString("rho")
		for i := 0; i < nrClass*(nrClass-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.Rho[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}

	if model.Label != nil {
		fb.WriteString("label")
		for i := 0; i < nrClass; i++ {
			fb.WriteString(" " + strconv.Itoa(model.Label[i]))
		}
		fb.WriteString("\n")
	}

	if model.ProbA != nil { // regression has probA only
		fb.WriteString("probA")
		for i := 0; i < nrClass*(nrClass-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.ProbA[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}
	if model.ProbB != nil {
		fb.WriteString("probB")
		for i := 0; i < nrClass*(nrClass-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.ProbB[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}

	if model.NSV != nil {
		fb.WriteString("nrSv")
		for i := 0; i < nrClass; i++ {
			fb.WriteString(" " + strconv.Itoa(model.NSV[i]))
		}
		fb.WriteString("\n")
	}

	fb.WriteString("SV\n")
	svCoef := model.SvCoef
	SV := model.SV

	for i := 0; i < l; i++ {
		for j := 0; j < nrClass-1; j++ {
			fb.WriteString(strconv.FormatFloat(svCoef[j][i], 'g', -1, 64) + " ")
		}

		p := SV[i]
		if param.KernelType == PRECOMPUTED {
			fb.WriteString("0:" + strconv.Itoa(int(p[0].Value)))
		} else {
			for j := 0; j < len(p); j++ {
				fb.WriteString(strconv.Itoa(p[j].Index) + ":" + strconv.FormatFloat(p[j].Value, 'g', -1, 64) + " ")
			}
			fb.WriteString("\n")
		}
	}
}

func (m *SVM) aotf(s string) float64 {
	rst, _ := strconv.ParseFloat(s, 64)
	return rst
}
func (m *SVM) atoi(s string) int {
	rst, _ := strconv.Atoi(s)
	return rst
}

/*
// svmLoadModel read svm file
func (m *SVM) svmLoadModel(modelFileName string) {
	file, err := os.Open(modelFileName)
	defer file.Close()
	fb := bufio.NewReader(file)
	model := new(SVMModel)
	param := new(SVMParameter)
	model.param = param
	model.rho = nil
	model.probA = nil
	model.probB = nil
	model.label = nil
	model.nSV = nil

	for {
		cmd, _ := fb.ReadLine()
		arg := cmd.substring(cmd.indexOf(' ')+1);

		if cmd.startsWith("svmType") {
			var i int
			for i=0;i< len(m.svmTypeTable);i++{
				if arg.indexOf(m.svmTypeTable[i])!=-1 {
					param.svmType=i
					break
				}
			}
			if i == len(m.svmTypeTable) {
				log.Print("unknown svm type.\n")
				return nil
			}
		} else if cmd.startsWith("kernelType") {
			var i int
			for i=0;i<len(m.kernelTypeTable);i++ {
				if arg.indexOf(m.kernelTypeTable[i])!=-1 {
					param.kernelType=i
					break
				}
			}
			if i == len(m.kernelTypeTable) {
				log.Print("unknown kernel function.\n")
				return nil
			}
		} else if cmd.startsWith("degree") {
			param.degree = atoi(arg)
		} else if cmd.startsWith("gamma") {
			param.gamma = atof(arg)
		} else if cmd.startsWith("coef0") {
			param.coef0 = atof(arg)
		} else if cmd.startsWith("nrClass") {
			model.nrClass = atoi(arg)
		} else if cmd.startsWith("totalSv") {
			model.l = atoi(arg)
		} else if cmd.startsWith("rho") {
			n := model.nrClass * (model.nrClass-1)/2
			model.rho = make([]float64,n)
			// st := new StringTokenizer(arg);
			for i:=0;i<n;i++ {
				model.rho[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("label") {
			n := model.nrClass
			model.label = make([]int,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i=0;i<n;i++ {
				model.label[i] = atoi(st.nextToken())
			}
		} else if cmd.startsWith("probA") {
			n := model.nrClass*(model.nrClass-1)/2
			model.probA = make([]float64,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i:=0;i<n;i++ {
				model.probA[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("probB") {
			n := model.nrClass*(model.nrClass-1)/2;
			model.probB = make([]float64,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i:=0;i<n;i++ {
					model.probB[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("nrSv") {
			n := model.nrClass
			model.nSV = make([]int,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i:=0;i<n;i++ {
				model.nSV[i] = atoi(st.nextToken())
			}
		} else if cmd.startsWith("SV") {
				break
		} else {
			System.err.print("unknown text in model file: ["+cmd+"]\n");
			return nil;
		}
	}

	// read svCoef and SV

	m := model.nrClass - 1;
	l := model.l;
	//model.svCoef = new float64[m][l];
	model.svCoef = make([][]float64, m);
	for i := range model.svCoef {
		model.svCoef[i] = make([]float64, l)
	}
	model.SV = make([][]SVMNode,l)

	for i:=0;i<l;i++ {
		line := fb.ReadLine();
		//StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

		for k:=0;k<m;k++ {
				model.svCoef[k][i] = atof(st.nextToken())
		}
		n := st.countTokens()/2;
		model.SV[i] = make([]SVMNode,n)
		for j:=0;j<n;j++ {
			model.SV[i][j] = new(SVMNode)
			model.SV[i][j].index = atoi(st.nextToken())
			model.SV[i][j].value = atof(st.nextToken())
		}
	}
	return model
}
*/

// SVMCheckParameter check param
func (m *SVM) SVMCheckParameter(prob *SVMProblem, param *SVMParameter) string {
	// svmType

	svmType := param.SvmType
	if svmType != CSVC &&
		svmType != NUSVC &&
		svmType != ONECLASS &&
		svmType != EPSILONSVR &&
		svmType != NUSVR {
		return "unknown svm type"
	}

	// kernelType, degree

	kernelType := param.KernelType
	if kernelType != LINEAR &&
		kernelType != POLY &&
		kernelType != RBF &&
		kernelType != SIGMOID &&
		kernelType != PRECOMPUTED {
		return "unknown kernel type"
	}

	if param.Gamma < 0 {
		return "gamma < 0"
	}

	if param.Degree < 0 {
		return "degree of polynomial kernel < 0"
	}

	// cacheSize,eps,C,nu,p,shrinking

	if param.CacheSize <= 0 {
		return "cacheSize <= 0"
	}

	if param.Eps <= 0 {
		return "eps <= 0"
	}

	if svmType == CSVC ||
		svmType == EPSILONSVR ||
		svmType == NUSVR {
		if param.C <= 0 {
			return "C <= 0"
		}
	}
	if svmType == NUSVC ||
		svmType == ONECLASS ||
		svmType == NUSVR {
		if param.Nu <= 0 || param.Nu > 1 {
			return "nu <= 0 or nu > 1"
		}
	}
	if svmType == EPSILONSVR {
		if param.P < 0 {
			return "p < 0"
		}
	}
	if param.Shrinking != 0 &&
		param.Shrinking != 1 {
		return "shrinking != 0 and shrinking != 1"
	}

	if param.Probability != 0 &&
		param.Probability != 1 {
		return "probability != 0 and probability != 1"
	}

	if param.Probability == 1 &&
		svmType == ONECLASS {
		return "one-class SVM probability output not supported yet"
	}

	// check whether nu-svc is feasible

	if svmType == NUSVC {
		l := prob.L
		maxNrClass := 16
		nrClass := 0
		label := make([]int, maxNrClass)
		count := make([]int, maxNrClass)

		var i int
		for i = 0; i < l; i++ {
			mLabel := int(prob.Y[i])
			var j int
			for j = 0; j < nrClass; j++ {
				if mLabel == label[j] {
					count[j]++
					break
				}
			}
			if j == nrClass {
				if nrClass == maxNrClass {
					maxNrClass *= 2
					newData := make([]int, maxNrClass)
					//System.arraycopy(label,0,newData,0,label.length);
					copy(newData, label)
					label = newData
					newData = make([]int, maxNrClass)
					//System.arraycopy(count,0,newData,0,count.length);
					copy(newData, count)
					count = newData
				}
				label[nrClass] = mLabel
				count[nrClass] = 1
				nrClass++
			}
		}

		for i = 0; i < nrClass; i++ {
			n1 := count[i]
			for j := i + 1; j < nrClass; j++ {
				n2 := count[j]
				if param.Nu*float64(n1+n2)/2 > math.Min(float64(n1), float64(n2)) {
					return "specified nu is infeasible"
				}
			}
		}
	}
	return ""
}

// SVMCheckProbabilityModel check probability model
func (m *SVM) SVMCheckProbabilityModel(model *SVMModel) int {
	var rst int
	if ((model.Param.SvmType == CSVC ||
		model.Param.SvmType == NUSVC) && model.ProbA != nil && model.ProbB != nil) ||
		((model.Param.SvmType == EPSILONSVR || model.Param.SvmType == NUSVR) && model.ProbA != nil) {
		rst = 1
	} else {
		rst = 0
	}
	return rst
}
