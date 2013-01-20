package libsvm

import (
	"bufio"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type head_t struct {
	prev, next *head_t
	data       []float32
	len        int
}

type Cache struct {
	l        int
	size     int64
	head     []*head_t
	lru_head *head_t
}

func NewCache(l_ int, size_ int64) *Cache {
	this := new(Cache)
	this.l = l_
	this.size = size_
	this.head = make([]*head_t, this.l)
	for i := 0; i < this.l; i++ {
		this.head[i] = new(head_t)
	}
	this.size /= 4
	this.size -= int64(this.l * (16 / 4))                              // sizeof(head_t) == 16
	this.size = int64(math.Max(float64(this.size), float64(2*this.l))) // cache must be large enough for two columns
	this.lru_head = new(head_t)
	this.lru_head.prev = this.lru_head
	this.lru_head.next = this.lru_head.prev
	return this
}

func (this *Cache) lru_delete(h *head_t) {
	// delete from current location
	h.prev.next = h.next
	h.next.prev = h.prev
}
func (this *Cache) lru_insert(h *head_t) {
	// insert to last position
	h.next = this.lru_head
	h.prev = this.lru_head.prev
	h.prev.next = h
	h.next.prev = h
}

// request data [0,len)
// return some position p where [p,len) need to be filled
// (p >= len if nothing needs to be filled)
// java: simulate pointer using single-element array

func (this *Cache) get_data(index int, data [][]float32, len int) int {
	h := this.head[index]
	if h.len > 0 {
		this.lru_delete(h)
	}
	more := len - h.len
	if more > 0 {
		// free old space
		for this.size < int64(more) {
			old := this.lru_head.next
			this.lru_delete(old)
			this.size += int64(old.len)
			old.data = nil
			old.len = 0
		}

		// allocate new space
		new_data := make([]float32, len)
		if h.data != nil {
			// System.arraycopy(h.data,0,new_data,0,h.len)
			copy(new_data, h.data[:len])
		}
		h.data = new_data
		this.size -= int64(more)
		// do {int _=h.len; h.len=len; len=_;} while(false);
		h.len, len = len, h.len
	}

	this.lru_insert(h)
	data[0] = h.data
	return len
}

func (this *Cache) swap_index(i, j int) {
	if i == j {
		return
	}

	if this.head[i].len > 0 {
		this.lru_delete(this.head[i])
	}
	if this.head[j].len > 0 {
		this.lru_delete(this.head[j])
	}
	//do {float32[] _=head[i].data; head[i].data=head[j].data; head[j].data=_;} while(false);
	this.head[i].data, this.head[j].data = this.head[j].data, this.head[i].data
	//do {int _=head[i].len; head[i].len=head[j].len; head[j].len=_;} while(false);
	this.head[i].len, this.head[j].len = this.head[j].len, this.head[i].len

	if this.head[i].len > 0 {
		this.lru_insert(this.head[i])
	}
	if this.head[j].len > 0 {
		this.lru_insert(this.head[j])
	}

	if i > j {
		//do {int _=i; i=j; j=_;} while(false);
		i, j = j, i
	}
	for h := this.lru_head.next; h != this.lru_head; h = h.next {
		if h.len > i {
			if h.len > j {
				//do {float32 _=h.data[i]; h.data[i]=h.data[j]; h.data[j]=_;} while(false);
				h.data[i], h.data[j] = h.data[j], h.data[i]
			} else {
				// give up
				this.lru_delete(h)
				this.size += int64(h.len)
				h.data = nil
				h.len = 0
			}
		}
	}
}

type QMatrix interface {
	swap_index(i, j int)
	get_Q(column, len int) []float32
	get_QD() []float64
}

type Kernel struct {
	x           [][]SVM_Node
	x_square    []float64
	kernel_type int
	degree      int
	gamma       float64
	coef0       float64
}

func (this *Kernel) swap_index(i, j int) {
	this.x[i], this.x[j] = this.x[j], this.x[i]
	if this.x_square != nil {
		this.x_square[i], this.x_square[j] = this.x_square[j], this.x_square[i]
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

func (this *Kernel) kernel_function(i, j int) float64 {
	switch this.kernel_type {
	case LINEAR:
		return dot(this.x[i], this.x[j])
	case POLY:
		return powi(this.gamma*dot(this.x[i], this.x[j])+this.coef0, this.degree)
	case RBF:
		return math.Exp(-this.gamma * (this.x_square[i] + this.x_square[j] - 2*dot(this.x[i], this.x[j])))
	case SIGMOID:
		return math.Tanh(this.gamma*dot(this.x[i], this.x[j]) + this.coef0)
	case PRECOMPUTED:
		return this.x[i][int(this.x[j][0].Value)].Value
	}
	return 0
}

func NewKernel(l int, x_ [][]SVM_Node, param *SVM_Parameter) *Kernel {
	this := new(Kernel)
	this.kernel_type = param.Kernel_type
	this.degree = param.Degree
	this.gamma = param.Gamma
	this.coef0 = param.Coef0
	// x_.clone
	this.x = x_

	if this.kernel_type == RBF {
		this.x_square = make([]float64, l)
		for i := 0; i < l; i++ {
			this.x_square[i] = dot(this.x[i], this.x[i])
		}
	} else {
		this.x_square = nil
	}
	return this
}

func dot(x, y []SVM_Node) float64 {
	var sum float64
	sum = 0
	xlen := len(x)
	ylen := len(y)
	i := 0
	j := 0
	for i < xlen && j < ylen {
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

func k_function(x, y []SVM_Node, param *SVM_Parameter) float64 {
	switch param.Kernel_type {
	case LINEAR:
		return dot(x, y)
	case POLY:
		return powi(param.Gamma*dot(x, y)+param.Coef0, param.Degree)
	case RBF:
		{
			var sum float64
			sum = 0
			xlen := len(x)
			ylen := len(y)
			i := 0
			j := 0
			for i < xlen && j < ylen {
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

			for i < xlen {
				sum += x[i].Value * x[i].Value
				i++
			}

			for j < ylen {
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
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
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
	LOWER_BOUND = int8(0)
	UPPER_BOUND = int8(1)
	FREE        = int8(2)
	INF         = math.MaxFloat64
)

type Solver struct {
	active_size  int
	y            []int8
	G            []float64 // gradient of objective function
	alpha_status []int8    // LOWER_BOUND, UPPER_BOUND, FREE
	alpha        []float64
	Q            QMatrix
	QD           []float64
	eps          float64
	Cp, Cn       float64
	p            []float64
	active_set   []int
	G_bar        []float64 // gradient, if we treat free variables as 0
	l            int
	unshrink     bool // XXX
}

func (this *Solver) get_C(i int) float64 {
	var rst float64
	if this.y[i] > 0 {
		rst = this.Cp
	} else {
		rst = this.Cn
	}
	return rst
}

func (this *Solver) update_alpha_status(i int) {
	if this.alpha[i] >= this.get_C(i) {
		this.alpha_status[i] = UPPER_BOUND
	} else if this.alpha[i] <= 0 {
		this.alpha_status[i] = LOWER_BOUND
	} else {
		this.alpha_status[i] = FREE
	}
}

func (this *Solver) is_upper_bound(i int) bool {
	return this.alpha_status[i] == UPPER_BOUND
}
func (this *Solver) is_lower_bound(i int) bool {
	return this.alpha_status[i] == LOWER_BOUND
}
func (this *Solver) is_free(i int) bool {
	return this.alpha_status[i] == FREE
}

type SolutionInfo struct {
	obj           float64
	rho           float64
	upper_bound_p float64
	upper_bound_n float64
	r             float64 // for Solver_NU
}

func (this *Solver) swap_index(i, j int) {
	// todo
	this.Q.swap_index(i, j)
	//do {byte _=y[i]; y[i]=y[j]; y[j]=_;} while(false);
	this.y[i], this.y[j] = this.y[j], this.y[i]
	//do {float64 _=G[i]; G[i]=G[j]; G[j]=_;} while(false);
	this.G[i], this.G[j] = this.G[j], this.G[i]
	//do {byte _=alpha_status[i]; alpha_status[i]=alpha_status[j]; alpha_status[j]=_;} while(false);
	this.alpha_status[i], this.alpha_status[j] = this.alpha_status[j], this.alpha_status[i]
	//do {float64 _=alpha[i]; alpha[i]=alpha[j]; alpha[j]=_;} while(false);
	this.alpha[i], this.alpha[j] = this.alpha[j], this.alpha[i]
	//do {float64 _=p[i]; p[i]=p[j]; p[j]=_;} while(false);
	this.p[i], this.p[j] = this.p[j], this.p[i]
	//do {int _=active_set[i]; active_set[i]=active_set[j]; active_set[j]=_;} while(false);
	this.active_set[i], this.active_set[j] = this.active_set[j], this.active_set[i]
	//do {float64 _=G_bar[i]; G_bar[i]=G_bar[j]; G_bar[j]=_;} while(false);
	this.G_bar[i], this.G_bar[j] = this.G_bar[j], this.G_bar[i]
}

func (this *Solver) reconstruct_gradient() {
	// reconstruct inactive elements of G from G_bar and free variables

	if this.active_size == this.l {
		return
	}
	var i, j int
	nr_free := 0

	for j = this.active_size; j < this.l; j++ {
		this.G[j] = this.G_bar[j] + this.p[j]
	}
	for j = 0; j < this.active_size; j++ {
		if this.is_free(j) {
			nr_free++
		}
	}
	if (2 * nr_free) < this.active_size {
		log.Fatal("\nWARNING: using -h 0 may be faster\n")
	}

	if (nr_free * this.l) > (2 * this.active_size * (this.l - this.active_size)) {
		for i = this.active_size; i < this.l; i++ {
			Q_i := this.Q.get_Q(i, this.active_size)
			for j = 0; j < this.active_size; j++ {
				if this.is_free(j) {
					this.G[i] += this.alpha[j] * float64(Q_i[j])
				}
			}
		}
	} else {
		for i = 0; i < this.active_size; i++ {
			if this.is_free(i) {
				Q_i := this.Q.get_Q(i, this.l)
				alpha_i := this.alpha[i]
				for j = this.active_size; j < this.l; j++ {
					this.G[j] += alpha_i * float64(Q_i[j])
				}
			}
		}
	}
}

func (this *Solver) Solve(l int, Q QMatrix, p_ []float64, y_ []int8, alpha_ []float64, Cp, Cn, eps float64, si *SolutionInfo, shrinking int) {
	this.l = l
	this.Q = Q
	this.QD = Q.get_QD()
	this.p = make([]float64, len(p_))
	copy(this.p, p_)
	this.y = make([]int8, len(y_))
	copy(this.y, y_)
	this.alpha = make([]float64, len(alpha_))
	copy(this.alpha, alpha_)
	this.Cp = Cp
	this.Cn = Cn
	this.eps = eps
	this.unshrink = false

	// initialize alpha_status
	{
		this.alpha_status = make([]int8, l)
		for i := 0; i < l; i++ {
			this.update_alpha_status(i)
		}
	}

	// initialize active set (for shrinking)
	{
		this.active_set = make([]int, l)
		for i := 0; i < l; i++ {
			this.active_set[i] = i
		}
		this.active_size = l
	}

	// initialize gradient
	{
		this.G = make([]float64, l)
		this.G_bar = make([]float64, l)
		var i int
		for i = 0; i < l; i++ {
			this.G[i] = this.p[i]
			this.G_bar[i] = 0
		}
		for i = 0; i < l; i++ {
			if !this.is_lower_bound(i) {
				Q_i := this.Q.get_Q(i, l)
				alpha_i := this.alpha[i]
				var j int
				for j = 0; j < l; j++ {
					this.G[j] += alpha_i * float64(Q_i[j])
				}
				if this.is_upper_bound(i) {
					for j = 0; j < l; j++ {
						this.G_bar[j] += this.get_C(i) * float64(Q_i[j])
					}
				}
			}
		}
	}

	// optimization step

	iter := 0
	var max float64
	if this.l > math.MaxInt32/100 {
		max = float64(math.MaxInt32)
	} else {
		max = float64(100 * this.l)
	}
	max_iter := math.Max(10000000, max)
	counter := math.Min(float64(this.l), 1000) + 1
	working_set := make([]int, 2)

	for float64(iter) < max_iter {
		// show progress and do shrinking
		if counter--; counter == 0 {
			counter = math.Min(float64(this.l), 1000)
			if shrinking != 0 {
				this.do_shrinking()
			}
			log.Fatal(".")
		}

		if this.select_working_set(working_set) != 0 {
			// reconstruct the whole gradient
			this.reconstruct_gradient()
			// reset active set size and check
			this.active_size = l
			log.Fatal("*")
			if this.select_working_set(working_set) != 0 {
				break
			} else {
				counter = 1 // do shrinking next iteration
			}
		}

		i := working_set[0]
		j := working_set[1]

		iter++

		// update alpha[i] and alpha[j], handle bounds carefully

		Q_i := this.Q.get_Q(i, this.active_size)
		Q_j := this.Q.get_Q(j, this.active_size)

		C_i := this.get_C(i)
		C_j := this.get_C(j)

		old_alpha_i := this.alpha[i]
		old_alpha_j := this.alpha[j]

		if this.y[i] != this.y[j] {
			quad_coef := this.QD[i] + this.QD[j] + 2*float64(Q_i[j])
			if quad_coef <= 0 {
				quad_coef = 1e-12
			}
			delta := (-this.G[i] - this.G[j]) / quad_coef
			diff := this.alpha[i] - this.alpha[j]
			this.alpha[i] += delta
			this.alpha[j] += delta

			if diff > 0 {
				if this.alpha[j] < 0 {
					this.alpha[j] = 0
					this.alpha[i] = diff
				}
			} else {
				if this.alpha[i] < 0 {
					this.alpha[i] = 0
					this.alpha[j] = -diff
				}
			}
			if diff > (C_i - C_j) {
				if this.alpha[i] > C_i {
					this.alpha[i] = C_i
					this.alpha[j] = C_i - diff
				}
			} else {
				if this.alpha[j] > C_j {
					this.alpha[j] = C_j
					this.alpha[i] = C_j + diff
				}
			}
		} else {
			quad_coef := this.QD[i] + this.QD[j] - 2*float64(Q_i[j])
			if quad_coef <= 0 {
				quad_coef = 1e-12
			}
			delta := (this.G[i] - this.G[j]) / quad_coef
			sum := this.alpha[i] + this.alpha[j]
			this.alpha[i] -= delta
			this.alpha[j] += delta

			if sum > C_i {
				if this.alpha[i] > C_i {
					this.alpha[i] = C_i
					this.alpha[j] = sum - C_i
				}
			} else {
				if this.alpha[j] < 0 {
					this.alpha[j] = 0
					this.alpha[i] = sum
				}
			}
			if sum > C_j {
				if this.alpha[j] > C_j {
					this.alpha[j] = C_j
					this.alpha[i] = sum - C_j
				}
			} else {
				if this.alpha[i] < 0 {
					this.alpha[i] = 0
					this.alpha[j] = sum
				}
			}
		}

		// update G

		delta_alpha_i := this.alpha[i] - old_alpha_i
		delta_alpha_j := this.alpha[j] - old_alpha_j

		for k := 0; k < this.active_size; k++ {
			this.G[k] += float64(Q_i[k])*delta_alpha_i + float64(Q_j[k])*delta_alpha_j
		}

		// update alpha_status and G_bar

		{
			ui := this.is_upper_bound(i)
			uj := this.is_upper_bound(j)
			this.update_alpha_status(i)
			this.update_alpha_status(j)
			var k int
			if ui != this.is_upper_bound(i) {
				Q_i = Q.get_Q(i, l)
				if ui {
					for k = 0; k < l; k++ {
						this.G_bar[k] -= C_i * float64(Q_i[k])
					}
				} else {
					for k = 0; k < l; k++ {
						this.G_bar[k] += C_i * float64(Q_i[k])
					}
				}
			}

			if uj != this.is_upper_bound(j) {
				Q_j = Q.get_Q(j, l)
				if uj {
					for k = 0; k < l; k++ {
						this.G_bar[k] -= float64(C_j) * float64(Q_j[k])
					}
				} else {
					for k = 0; k < l; k++ {
						this.G_bar[k] += float64(C_j) * float64(Q_j[k])
					}
				}
			}
		}
	}
	if float64(iter) >= max_iter {
		if this.active_size < l {
			// reconstruct the whole gradient to calculate objective value
			this.reconstruct_gradient()
			this.active_size = l
			log.Fatal("*")
		}
		log.Fatal("\nWARNING: reaching max number of iterations")
	}

	// calculate rho

	si.rho = this.calculate_rho()

	// calculate objective value
	{
		v := float64(0)
		var i int
		for i = 0; i < l; i++ {
			v += this.alpha[i] * (this.G[i] + this.p[i])
		}

		si.obj = v / 2
	}

	// put back the solution
	{
		for i := 0; i < l; i++ {
			alpha_[this.active_set[i]] = this.alpha[i]
		}
	}

	si.upper_bound_p = Cp
	si.upper_bound_n = Cn
	log.Fatal("\noptimization finished, #iter = " + strconv.Itoa(iter) + "\n")
}

func (this *Solver) select_working_set(working_set []int) int {
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: mimimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	Gmax := -INF
	Gmax2 := -INF
	Gmax_idx := -1
	Gmin_idx := -1
	obj_diff_min := INF

	for t := 0; t < this.active_size; t++ {
		if this.y[t] == +1 {
			if !this.is_upper_bound(t) {
				if -this.G[t] >= Gmax {
					Gmax = -this.G[t]
					Gmax_idx = t
				}
			}
		} else {
			if !this.is_lower_bound(t) {
				if this.G[t] >= Gmax {
					Gmax = this.G[t]
					Gmax_idx = t
				}
			}
		}
	}
	i := Gmax_idx
	var Q_i []float32
	Q_i = nil

	if i != -1 { // null Q_i not accessed: Gmax=-INF if i=-1
		Q_i = this.Q.get_Q(i, this.active_size)
	}

	for j := 0; j < this.active_size; j++ {
		if this.y[j] == +1 {
			if !this.is_lower_bound(j) {
				grad_diff := Gmax + this.G[j]
				if this.G[j] >= Gmax2 {
					Gmax2 = this.G[j]
				}
				if grad_diff > 0 {
					var obj_diff float64
					// y []int8
					tmp := float64(this.y[i])
					tmp2 := float64(2 * Q_i[j])
					quad_coef := this.QD[i] + this.QD[j] - tmp*tmp2
					if quad_coef > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coef
					} else {
						obj_diff = -(grad_diff * grad_diff) / 1e-12
					}

					if obj_diff <= obj_diff_min {
						Gmin_idx = j
						obj_diff_min = obj_diff
					}
				}
			}
		} else {
			if !this.is_upper_bound(j) {
				grad_diff := Gmax - this.G[j]
				if -this.G[j] >= Gmax2 {
					Gmax2 = -this.G[j]
				}
				if grad_diff > 0 {
					var obj_diff float64
					tmp := float64(this.y[i])
					tmp2 := float64(2 * Q_i[j])
					quad_coef := this.QD[i] + this.QD[j] + tmp*tmp2
					if quad_coef > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coef
					} else {
						obj_diff = -(grad_diff * grad_diff) / 1e-12
					}

					if obj_diff <= obj_diff_min {
						Gmin_idx = j
						obj_diff_min = obj_diff
					}
				}
			}
		}
	}

	if Gmax+Gmax2 < this.eps {
		return 1
	}

	working_set[0] = Gmax_idx
	working_set[1] = Gmin_idx
	return 0
}

func (this *Solver) be_shrunk(i int, Gmax1, Gmax2 float64) bool {
	var rst bool
	if this.is_upper_bound(i) {
		if this.y[i] == +1 {
			rst = -this.G[i] > Gmax1
		} else {
			rst = -this.G[i] > Gmax2
		}
	} else if this.is_lower_bound(i) {
		if this.y[i] == +1 {
			rst = this.G[i] > Gmax2
		} else {
			rst = this.G[i] > Gmax1
		}
	} else {
		rst = false
	}
	return rst
}

func (this *Solver) do_shrinking() {
	var i int
	Gmax1 := -INF // max { -y_i * grad(f)_i | i in I_up(\alpha) }
	Gmax2 := -INF // max { y_i * grad(f)_i | i in I_low(\alpha) }
	// find maximal violating pair first
	for i = 0; i < this.active_size; i++ {
		if this.y[i] == +1 {
			if !this.is_upper_bound(i) {
				if -this.G[i] >= Gmax1 {
					Gmax1 = -this.G[i]
				}
			}
			if !this.is_lower_bound(i) {
				if this.G[i] >= Gmax2 {
					Gmax2 = this.G[i]
				}
			}
		} else {
			if !this.is_upper_bound(i) {
				if -this.G[i] >= Gmax2 {
					Gmax2 = -this.G[i]
				}
			}
			if !this.is_lower_bound(i) {
				if this.G[i] >= Gmax1 {
					Gmax1 = this.G[i]
				}
			}
		}
	}

	if this.unshrink == false && Gmax1+Gmax2 <= this.eps*10 {
		this.unshrink = true
		this.reconstruct_gradient()
		this.active_size = this.l
	}

	for i = 0; i < this.active_size; i++ {
		if this.be_shrunk(i, Gmax1, Gmax2) {
			this.active_size--
			for this.active_size > i {
				if !this.be_shrunk(this.active_size, Gmax1, Gmax2) {
					this.swap_index(i, this.active_size)
					break
				}
				this.active_size--
			}
		}
	}
}

func (this *Solver) calculate_rho() float64 {
	var r float64
	nr_free := 0
	ub := INF
	lb := -INF
	sum_free := float64(0)
	for i := 0; i < this.active_size; i++ {
		yG := float64(this.y[i]) * this.G[i]

		if this.is_lower_bound(i) {
			if this.y[i] > 0 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		} else if this.is_upper_bound(i) {
			if this.y[i] < 0 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		} else {
			nr_free++
			sum_free += yG
		}
	}

	if nr_free > 0 {
		r = sum_free / float64(nr_free)
	} else {
		r = (ub + lb) / 2
	}
	return r
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//

type Solver_NU struct {
	Solver
	si SolutionInfo
}

func (this *Solver_NU) Solve(l int, Q QMatrix, p []float64, y []int8, alpha []float64, Cp, Cn, eps float64, si *SolutionInfo, shrinking int) {
	this.si = *si
	this.Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking)
}

func (this *Solver_NU) select_working_set(working_set []int) int {
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	Gmaxp := -INF
	Gmaxp2 := -INF
	Gmaxp_idx := -1

	Gmaxn := -INF
	Gmaxn2 := -INF
	Gmaxn_idx := -1

	Gmin_idx := -1
	obj_diff_min := INF

	for t := 0; t < this.active_size; t++ {
		if this.y[t] == +1 {
			if !this.is_upper_bound(t) {
				if -this.G[t] >= Gmaxp {
					Gmaxp = -this.G[t]
					Gmaxp_idx = t
				}
			}
		} else {
			if !this.is_lower_bound(t) {
				if this.G[t] >= Gmaxn {
					Gmaxn = this.G[t]
					Gmaxn_idx = t
				}
			}
		}
	}

	ip := Gmaxp_idx
	in := Gmaxn_idx
	var Q_ip []float32
	Q_ip = nil
	var Q_in []float32
	Q_in = nil
	if ip != -1 { // null Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = this.Q.get_Q(ip, this.active_size)
	}
	if in != -1 {
		Q_in = this.Q.get_Q(in, this.active_size)
	}

	for j := 0; j < this.active_size; j++ {
		if this.y[j] == +1 {
			if !this.is_lower_bound(j) {
				grad_diff := Gmaxp + this.G[j]
				if this.G[j] >= Gmaxp2 {
					Gmaxp2 = this.G[j]
				}
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := this.QD[ip] + this.QD[j] - 2*float64(Q_ip[j])
					if quad_coef > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coef
					} else {
						obj_diff = -(grad_diff * grad_diff) / 1e-12
					}

					if obj_diff <= obj_diff_min {
						Gmin_idx = j
						obj_diff_min = obj_diff
					}
				}
			}
		} else {
			if !this.is_upper_bound(j) {
				grad_diff := Gmaxn - this.G[j]
				if -this.G[j] >= Gmaxn2 {
					Gmaxn2 = -this.G[j]
				}
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := this.QD[in] + this.QD[j] - 2*float64(Q_in[j])
					if quad_coef > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coef
					} else {
						obj_diff = -(grad_diff * grad_diff) / 1e-12
					}

					if obj_diff <= obj_diff_min {
						Gmin_idx = j
						obj_diff_min = obj_diff
					}
				}
			}
		}
	}

	if math.Max(Gmaxp+Gmaxp2, Gmaxn+Gmaxn2) < this.eps {
		return 1
	}

	if this.y[Gmin_idx] == +1 {
		working_set[0] = Gmaxp_idx
	} else {
		working_set[0] = Gmaxn_idx
	}
	working_set[1] = Gmin_idx
	return 0
}

func (this *Solver_NU) be_shrunk(i int, Gmax1, Gmax2, Gmax3, Gmax4 float64) bool {
	var rst bool
	if this.is_upper_bound(i) {
		if this.y[i] == +1 {
			rst = (-this.G[i] > Gmax1)
		} else {
			rst = (-this.G[i] > Gmax4)
		}
	} else if this.is_lower_bound(i) {
		if this.y[i] == +1 {
			rst = (this.G[i] > Gmax2)
		} else {
			rst = (this.G[i] > Gmax3)
		}
	} else {
		rst = false
	}
	return rst
}

func (this *Solver_NU) do_shrinking() {
	Gmax1 := -INF // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	Gmax2 := -INF // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	Gmax3 := -INF // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	Gmax4 := -INF // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	var i int
	for i = 0; i < this.active_size; i++ {
		if !this.is_upper_bound(i) {
			if this.y[i] == +1 {
				if -this.G[i] > Gmax1 {
					Gmax1 = -this.G[i]
				}
			} else if -this.G[i] > Gmax4 {
				Gmax4 = -this.G[i]
			}
		}
		if !this.is_lower_bound(i) {
			if this.y[i] == +1 {
				if this.G[i] > Gmax2 {
					Gmax2 = this.G[i]
				}
			} else if this.G[i] > Gmax3 {
				Gmax3 = this.G[i]
			}
		}
	}

	if this.unshrink == false && math.Max(Gmax1+Gmax2, Gmax3+Gmax4) <= this.eps*10 {
		this.unshrink = true
		this.reconstruct_gradient()
		this.active_size = this.l
	}

	for i = 0; i < this.active_size; i++ {
		if this.be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4) {
			this.active_size--
			for this.active_size > i {
				if !this.be_shrunk(this.active_size, Gmax1, Gmax2, Gmax3, Gmax4) {
					this.swap_index(i, this.active_size)
					break
				}
				this.active_size--
			}
		}
	}
}

func (this *Solver_NU) calculate_rho() float64 {
	nr_free1 := 0
	nr_free2 := 0
	ub1 := INF
	ub2 := INF
	lb1 := -INF
	lb2 := -INF
	sum_free1 := float64(0)
	sum_free2 := float64(0)

	for i := 0; i < this.active_size; i++ {
		if this.y[i] == +1 {
			if this.is_lower_bound(i) {
				ub1 = math.Min(ub1, this.G[i])
			} else if this.is_upper_bound(i) {
				lb1 = math.Max(lb1, this.G[i])
			} else {
				nr_free1++
				sum_free1 += this.G[i]
			}
		} else {
			if this.is_lower_bound(i) {
				ub2 = math.Min(ub2, this.G[i])
			} else if this.is_upper_bound(i) {
				lb2 = math.Max(lb2, this.G[i])
			} else {
				nr_free2++
				sum_free2 += this.G[i]
			}
		}
	}

	var r1, r2 float64
	if nr_free1 > 0 {
		r1 = sum_free1 / float64(nr_free1)
	} else {
		r1 = (ub1 + lb1) / 2
	}

	if nr_free2 > 0 {
		r2 = sum_free2 / float64(nr_free2)
	} else {
		r2 = (ub2 + lb2) / 2
	}

	this.si.r = (r1 + r2) / 2
	return (r1 - r2) / 2
}

type SVC_Q struct {
	y     []int8
	cache *Cache
	QD    []float64
	Kernel
}

func NewSVC_Q(prob *SVM_Problem, param *SVM_Parameter, y_ []int8) *SVC_Q {
	this := &SVC_Q{
		Kernel: *NewKernel(prob.L, prob.X, param),
		y:      make([]int8, len(y_)),
		cache:  NewCache(prob.L, int64(param.Cache_size*(1<<20))),
		QD:     make([]float64, prob.L),
	}
	copy(this.y, y_)
	for i := 0; i < prob.L; i++ {
		this.QD[i] = this.kernel_function(i, i)
	}
	return this
}
func (this *SVC_Q) get_Q(i, len int) []float32 {
	data := make([][]float32, 1)
	var start, j int
	if start = this.cache.get_data(i, data, len); start < len {
		for j = start; j < len; j++ {
			data[0][j] = float32(this.y[i]*this.y[j]) * float32(this.kernel_function(i, j))
		}
	}
	return data[0]
}
func (this *SVC_Q) get_QD() []float64 {
	return this.QD
}
func (this *SVC_Q) swap_index(i, j int) {
	this.cache.swap_index(i, j)
	this.Kernel.swap_index(i, j)
	//do {byte _=y[i]; y[i]=y[j]; y[j]=_;} while(false);
	this.y[i], this.y[j] = this.y[j], this.y[i]
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	this.QD[i], this.QD[j] = this.QD[j], this.QD[i]
}

type ONE_CLASS_Q struct {
	Kernel
	cache *Cache
	QD    []float64
}

func NewONE_CLASS_Q(prob *SVM_Problem, param *SVM_Parameter) *ONE_CLASS_Q {
	this := &ONE_CLASS_Q{
		Kernel: *NewKernel(prob.L, prob.X, param),
		cache:  NewCache(prob.L, int64(param.Cache_size*(1<<20))),
		QD:     make([]float64, prob.L),
	}
	for i := 0; i < prob.L; i++ {
		this.QD[i] = this.kernel_function(i, i)
	}
	return this
}
func (this *ONE_CLASS_Q) get_Q(i, len int) []float32 {
	data := make([][]float32, 1)
	var start, j int
	if start = this.cache.get_data(i, data, len); start < len {
		for j = start; j < len; j++ {
			data[0][j] = float32(this.kernel_function(i, j))
		}
	}
	return data[0]
}

func (this *ONE_CLASS_Q) get_QD() []float64 {
	return this.QD
}

func (this *ONE_CLASS_Q) swap_index(i, j int) {
	this.cache.swap_index(i, j)
	this.Kernel.swap_index(i, j)
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	this.QD[i], this.QD[j] = this.QD[j], this.QD[i]
}

type SVR_Q struct {
	Kernel
	l           int
	cache       *Cache
	sign        []int8
	index       []int
	next_buffer int
	buffer      [][]float32
	QD          []float64
}

func NewSVR_Q(prob *SVM_Problem, param *SVM_Parameter) *SVR_Q {
	this := &SVR_Q{
		Kernel: *NewKernel(prob.L, prob.X, param),
		l:      prob.L,
		cache:  NewCache(prob.L, int64(param.Cache_size*(1<<20))),
		QD:     make([]float64, 2*prob.L),
		sign:   make([]int8, 2*prob.L),
		index:  make([]int, 2*prob.L),
		buffer: make([][]float32, 2),
	}
	for k := 0; k < this.l; k++ {
		this.sign[k] = 1
		this.sign[k+this.l] = -1
		this.index[k] = k
		this.index[k+this.l] = k
		this.QD[k] = this.kernel_function(k, k)
		this.QD[k+this.l] = this.QD[k]
	}
	for i := range this.buffer {
		this.buffer[i] = make([]float32, 2*this.l)
	}
	this.next_buffer = 0
	return this
}
func (this *SVR_Q) get_Q(i, len int) []float32 {
	data := make([][]float32, 1)
	var j int
	real_i := this.index[i]
	if this.cache.get_data(real_i, data, this.l) < this.l {
		for j = 0; j < this.l; j++ {
			data[0][j] = float32(this.kernel_function(real_i, j))
		}
	}

	// reorder and copy
	buf := this.buffer[this.next_buffer]
	this.next_buffer = 1 - this.next_buffer
	si := this.sign[i]
	for j = 0; j < len; j++ {
		buf[j] = float32(si) * float32(this.sign[j]) * data[0][this.index[j]]
	}
	return buf
}

func (this *SVR_Q) get_QD() []float64 {
	return this.QD
}

func (this *SVR_Q) swap_index(i, j int) {
	//do {byte _=sign[i]; sign[i]=sign[j]; sign[j]=_;} while(false);
	this.sign[i], this.sign[j] = this.sign[j], this.sign[i]
	//do {int _=index[i]; index[i]=index[j]; index[j]=_;} while(false);
	this.index[i], this.index[j] = this.index[j], this.index[i]
	//do {float64 _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	this.QD[i], this.QD[j] = this.QD[j], this.QD[i]
}

const LIBSVM_VERSION = 312

type decision_function struct {
	alpha []float64
	rho   float64
}

type SVM struct {
	rand *rand.Rand

	//
	// decision_function
	//
	svm_type_table []string

	kernel_type_table []string
}

func NewSvm() *SVM {
	this := new(SVM)
	this.rand = new(rand.Rand)
	this.svm_type_table = []string{"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"}
	this.kernel_type_table = []string{"linear", "polynomial", "rbf", "sigmoid", "precomputed"}
	return this
}

func (this *SVM) Solve_c_svc(prob *SVM_Problem, param *SVM_Parameter, alpha []float64, si *SolutionInfo, Cp, Cn float64) {
	l := prob.L
	minus_ones := make([]float64, l)
	y := make([]int8, l)

	var i int

	for i = 0; i < l; i++ {
		alpha[i] = 0
		minus_ones[i] = -1
		if prob.Y[i] > 0 {
			y[i] = +1
		} else {
			y[i] = -1
		}
	}

	s := new(Solver)
	s.Solve(l, NewSVC_Q(prob, param, y), minus_ones, y,
		alpha, Cp, Cn, param.Eps, si, param.Shrinking)

	sum_alpha := float64(0)
	for i = 0; i < l; i++ {
		sum_alpha += alpha[i]
	}

	if Cp == Cn {
		log.Println("nu = ", sum_alpha/(Cp*float64(prob.L)))
	}

	for i = 0; i < l; i++ {
		alpha[i] *= float64(y[i])
	}
}

func (this *SVM) Solve_nu_svc(prob *SVM_Problem, param *SVM_Parameter, alpha []float64, si *SolutionInfo) {
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

	sum_pos := nu * float64(l) / 2
	sum_neg := nu * float64(l) / 2

	for i = 0; i < l; i++ {
		if y[i] == +1 {
			alpha[i] = math.Min(1.0, sum_pos)
			sum_pos -= alpha[i]
		} else {
			alpha[i] = math.Min(1.0, sum_neg)
			sum_neg -= alpha[i]
		}
	}

	zeros := make([]float64, l)

	for i = 0; i < l; i++ {
		zeros[i] = 0
	}

	s := new(Solver_NU)
	s.Solve(l, NewSVC_Q(prob, param, y), zeros, y,
		alpha, 1.0, 1.0, param.Eps, si, param.Shrinking)
	r := si.r

	log.Println("C = ", 1/r)

	for i = 0; i < l; i++ {
		alpha[i] *= float64(y[i]) / r
	}

	si.rho /= r
	si.obj /= (r * r)
	si.upper_bound_p = 1 / r
	si.upper_bound_n = 1 / r
}

func (this *SVM) Solve_one_class(prob *SVM_Problem, param *SVM_Parameter, alpha []float64, si *SolutionInfo) {
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
	s.Solve(l, NewONE_CLASS_Q(prob, param), zeros, ones,
		alpha, 1.0, 1.0, param.Eps, si, param.Shrinking)
}

func (this *SVM) Solve_epsilon_svr(prob *SVM_Problem, param *SVM_Parameter, alpha []float64, si *SolutionInfo) {
	l := prob.L
	alpha2 := make([]float64, 2*l)
	linear_term := make([]float64, 2*l)
	y := make([]int8, 2*l)
	var i int

	for i = 0; i < l; i++ {
		alpha2[i] = 0
		linear_term[i] = param.P - prob.Y[i]
		y[i] = 1

		alpha2[i+l] = 0
		linear_term[i+l] = param.P + prob.Y[i]
		y[i+l] = -1
	}

	s := new(Solver)
	s.Solve(2*l, NewSVR_Q(prob, param), linear_term, y,
		alpha2, param.C, param.C, param.Eps, si, param.Shrinking)
	sum_alpha := float64(0)
	for i = 0; i < l; i++ {
		alpha[i] = alpha2[i] - alpha2[i+l]
		sum_alpha += math.Abs(alpha[i])
	}
	log.Println("nu = ", sum_alpha/(param.C*float64(l)))
}

func (this *SVM) Solve_nu_svr(prob *SVM_Problem, param *SVM_Parameter, alpha []float64, si *SolutionInfo) {
	l := prob.L
	C := param.C
	alpha2 := make([]float64, 2*l)
	linear_term := make([]float64, 2*l)
	y := make([]int8, 2*l)
	var i int

	sum := C * param.Nu * float64(l) / 2
	for i = 0; i < l; i++ {
		alpha2[i+l] = math.Min(sum, C)
		alpha2[i] = alpha2[i+1]
		sum -= alpha2[i]

		linear_term[i] = -prob.Y[i]
		y[i] = 1

		linear_term[i+l] = prob.Y[i]
		y[i+l] = -1
	}

	s := new(Solver_NU)
	s.Solve(2*l, NewSVR_Q(prob, param), linear_term, y,
		alpha2, C, C, param.Eps, si, param.Shrinking)

	log.Println("epsilon = ", -si.r)

	for i = 0; i < l; i++ {
		alpha[i] = alpha2[i] - alpha2[i+l]
	}
}

func (this *SVM) SVM_train_one(prob *SVM_Problem, param *SVM_Parameter, Cp, Cn float64) *decision_function {
	alpha := make([]float64, prob.L)
	si := new(SolutionInfo)
	switch param.Svm_type {
	case C_SVC:
		this.Solve_c_svc(prob, param, alpha, si, Cp, Cn)
	case NU_SVC:
		this.Solve_nu_svc(prob, param, alpha, si)
	case ONE_CLASS:
		this.Solve_one_class(prob, param, alpha, si)
	case EPSILON_SVR:
		this.Solve_epsilon_svr(prob, param, alpha, si)
	case NU_SVR:
		this.Solve_nu_svr(prob, param, alpha, si)
	}

	log.Println("obj = ", si.obj, ", rho = ", si.rho)

	// output SVs

	nSV := 0
	nBSV := 0
	for i := 0; i < prob.L; i++ {
		if math.Abs(alpha[i]) > 0 {
			nSV++
			if prob.Y[i] > 0 {
				if math.Abs(alpha[i]) >= si.upper_bound_p {
					nBSV++
				}
			} else {
				if math.Abs(alpha[i]) >= si.upper_bound_n {
					nBSV++
				}
			}
		}
	}

	log.Println("nSV = ", nSV, ", nBSV = ", nBSV)

	f := new(decision_function)
	f.alpha = alpha
	f.rho = si.rho
	return f
}

func (this *SVM) sigmoid_train(l int, dec_values, labels, probAB []float64) {
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

	max_iter := 100            // Maximal number of iterations
	min_step := float64(1e-10) // Minimal step taken in line search
	sigma := float64(1e-12)    // For numerically strict PD of Hessian
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
		fApB = dec_values[i]*A + B
		if fApB >= 0 {
			fval += t[i]*fApB + math.Log(1+math.Exp(-fApB))
		} else {
			fval += (t[i]-1)*fApB + math.Log(1+math.Exp(fApB))
		}
	}
	for iter = 0; iter < max_iter; iter++ {
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11 = sigma // numerically ensures strict PD
		h22 = sigma
		h21 = 0.0
		g1 = 0.0
		g2 = 0.0
		for i = 0; i < l; i++ {
			fApB = dec_values[i]*A + B
			if fApB >= 0 {
				p = math.Exp(-fApB) / (1.0 + math.Exp(-fApB))
				q = 1.0 / (1.0 + math.Exp(-fApB))
			} else {
				p = 1.0 / (1.0 + math.Exp(fApB))
				q = math.Exp(fApB) / (1.0 + math.Exp(fApB))
			}
			d2 = p * q
			h11 += dec_values[i] * dec_values[i] * d2
			h22 += d2
			h21 += dec_values[i] * d2
			d1 = t[i] - p
			g1 += dec_values[i] * d1
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
		for stepsize >= min_step {
			newA = A + stepsize*dA
			newB = B + stepsize*dB

			// New function value
			newf = 0.0
			for i = 0; i < l; i++ {
				fApB = dec_values[i]*newA + newB
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

		if stepsize < min_step {
			log.Println("Line search fails in two-class probability estimates")
			break
		}
	}

	if iter >= max_iter {
		log.Println("Reaching maximal iterations in two-class probability estimates")
	}
	probAB[0] = A
	probAB[1] = B
}

func (this *SVM) sigmoid_predict(decision_value, A, B float64) float64 {
	var rst float64
	fApB := decision_value*A + B
	if fApB >= 0 {
		rst = math.Exp(-fApB) / (1.0 + math.Exp(-fApB))
	} else {
		rst = 1.0 / (1 + math.Exp(fApB))
	}
	return rst
}

func (this *SVM) multiclass_probability(k int, r [][]float64, p []float64) {
	var t, j int
	iter := 0
	max_iter := math.Max(100, float64(k))
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
	for iter = 0; iter < int(max_iter); iter++ {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0
		for t = 0; t < k; t++ {
			Qp[t] = 0
			for j = 0; j < k; j++ {
				Qp[t] += Q[t][j] * p[j]
			}
			pQp += p[t] * Qp[t]
		}
		max_error := float64(0)
		for t = 0; t < k; t++ {
			error := math.Abs(Qp[t] - pQp)
			if error > max_error {
				max_error = error
			}
		}
		if max_error < eps {
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
	if iter >= int(max_iter) {
		log.Println("Exceeds max_iter in multiclass_prob")
	}
}

func (this *SVM) SVM_binary_svc_probability(prob *SVM_Problem, param *SVM_Parameter, Cp, Cn float64, probAB []float64) {
	var i int
	nr_fold := 5
	perm := make([]int, prob.L)
	dec_values := make([]float64, prob.L)

	// random shuffle
	for i = 0; i < prob.L; i++ {
		perm[i] = i
	}
	for i = 0; i < prob.L; i++ {
		j := i + int(rand.Int31n(int32(prob.L-i)))
		//do {int _=perm[i]; perm[i]=perm[j]; perm[j]=_;} while(false);
		perm[i], perm[j] = perm[j], perm[i]
	}
	for i = 0; i < nr_fold; i++ {
		begin := i * prob.L / nr_fold
		end := (i + 1) * prob.L / nr_fold
		var j, k int
		subprob := new(SVM_Problem)

		subprob.L = prob.L - (end - begin)
		subprob.X = make([][]SVM_Node, subprob.L)
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
		p_count := 0
		n_count := 0
		for j = 0; j < k; j++ {
			if subprob.Y[j] > 0 {
				p_count++
			} else {
				n_count++
			}
		}
		if p_count == 0 && n_count == 0 {
			for j = begin; j < end; j++ {
				dec_values[perm[j]] = 0
			}
		} else if p_count > 0 && n_count == 0 {
			for j = begin; j < end; j++ {
				dec_values[perm[j]] = 1
			}
		} else if p_count == 0 && n_count > 0 {
			for j = begin; j < end; j++ {
				dec_values[perm[j]] = -1
			}
		} else {
			subparam := param.Clone()
			subparam.Probability = 0
			subparam.C = 1.0
			subparam.Nr_weight = 2
			subparam.Weight_label = make([]int, 2)
			subparam.Weight = make([]float64, 2)
			subparam.Weight_label[0] = +1
			subparam.Weight_label[1] = -1
			subparam.Weight[0] = Cp
			subparam.Weight[1] = Cn
			submodel := this.SVM_train(subprob, subparam)
			for j = begin; j < end; j++ {
				dec_value := make([]float64, 1)
				this.SVM_predict_values(submodel, prob.X[perm[j]], dec_value)
				dec_values[perm[j]] = dec_value[0]
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= float64(submodel.Label[0])
			}
		}
	}
	this.sigmoid_train(prob.L, dec_values, prob.Y, probAB)
}

func (this *SVM) SVM_svr_probability(prob *SVM_Problem, param *SVM_Parameter) float64 {
	var i int
	nr_fold := 5
	ymv := make([]float64, prob.L)
	mae := float64(0)

	newparam := param.Clone()
	newparam.Probability = 0
	this.SVM_cross_validation(prob, newparam, nr_fold, ymv)
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
// perm, length l, must be allocated before calling this subroutine

func (this *SVM) SVM_group_classes(prob *SVM_Problem, nr_class_ret []int, label_ret, start_ret, count_ret [][]int, perm []int) {
	l := prob.L
	max_nr_class := 16
	nr_class := 0
	label := make([]int, max_nr_class)
	count := make([]int, max_nr_class)
	data_label := make([]int, l)
	var i int

	for i = 0; i < l; i++ {
		this_label := int(prob.Y[i])
		var j int
		for j = 0; j < nr_class; j++ {
			if this_label == label[j] {
				count[j]++
				break
			}
		}
		data_label[i] = j
		if j == nr_class {
			if nr_class == max_nr_class {
				max_nr_class *= 2
				new_data := make([]int, max_nr_class)
				//System.arraycopy(label,0,new_data,0,label.length);
				copy(new_data, label)
				label = new_data
				new_data = make([]int, max_nr_class)
				//System.arraycopy(count,0,new_data,0,count.length);
				copy(new_data, count)
				count = new_data
			}
			label[nr_class] = this_label
			count[nr_class] = 1
			nr_class++
		}
	}

	start := make([]int, nr_class)
	start[0] = 0
	for i = 1; i < nr_class; i++ {
		start[i] = start[i-1] + count[i-1]
	}
	for i = 0; i < l; i++ {
		perm[start[data_label[i]]] = i
		start[data_label[i]]++
	}
	start[0] = 0
	for i = 1; i < nr_class; i++ {
		start[i] = start[i-1] + count[i-1]
	}

	nr_class_ret[0] = nr_class
	label_ret[0] = label
	start_ret[0] = start
	count_ret[0] = count
}

func (this *SVM) SVM_train(prob *SVM_Problem, param *SVM_Parameter) *SVM_Model {

	model := new(SVM_Model)
	model.Param = param

	if param.Svm_type == ONE_CLASS || param.Svm_type == EPSILON_SVR || param.Svm_type == NU_SVR {
		// regression or one-class-svm
		model.Nr_class = 2
		model.Label = nil
		model.NSV = nil
		model.ProbA = nil
		model.ProbB = nil
		model.Sv_coef = make([][]float64, 1)

		if param.Probability == 1 && (param.Svm_type == EPSILON_SVR || param.Svm_type == NU_SVR) {
			model.ProbA = make([]float64, 1)
			model.ProbA[0] = this.SVM_svr_probability(prob, param)
		}

		f := this.SVM_train_one(prob, param, 0, 0)
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
		model.SV = make([][]SVM_Node, nSV)
		model.Sv_coef[0] = make([]float64, nSV)
		model.Sv_indices = make([]int, nSV)
		j := 0
		for i = 0; i < prob.L; i++ {
			if math.Abs(f.alpha[i]) > 0 {
				model.SV[j] = prob.X[i]
				model.Sv_coef[0][j] = f.alpha[i]
				model.Sv_indices[j] = i + 1
				j++
			}
		}
	} else {
		// classification
		l := prob.L
		tmp_nr_class := make([]int, 1)
		tmp_label := make([][]int, 1)
		tmp_start := make([][]int, 1)
		tmp_count := make([][]int, 1)
		perm := make([]int, l)

		// group training data of the same class
		this.SVM_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm)
		nr_class := tmp_nr_class[0]
		label := tmp_label[0]
		start := tmp_start[0]
		count := tmp_count[0]

		if nr_class == 1 {
			log.Println("WARNING: training data in only one class. See README for details.")
		}

		x := make([][]SVM_Node, l)
		var i int
		for i = 0; i < l; i++ {
			x[i] = prob.X[perm[i]]
		}

		// calculate weighted C

		weighted_C := make([]float64, nr_class)
		for i = 0; i < nr_class; i++ {
			weighted_C[i] = param.C
		}
		for i = 0; i < param.Nr_weight; i++ {
			var j int
			for j = 0; j < nr_class; j++ {
				if param.Weight_label[i] == label[j] {
					break
				}
			}
			if j == nr_class {
				log.Fatal("WARNING: class label " + strconv.Itoa(param.Weight_label[i]) + " specified in weight is not found\n")
			} else {
				weighted_C[j] *= param.Weight[i]
			}
		}

		// train k*(k-1)/2 models

		nonzero := make([]bool, l)
		for i = 0; i < l; i++ {
			nonzero[i] = false
		}
		f := make([]decision_function, nr_class*(nr_class-1)/2)

		var probA, probB []float64
		probA = nil
		probB = nil
		if param.Probability == 1 {
			probA = make([]float64, nr_class*(nr_class-1)/2)
			probB = make([]float64, nr_class*(nr_class-1)/2)
		}

		p := 0
		for i = 0; i < nr_class; i++ {
			for j := i + 1; j < nr_class; j++ {
				sub_prob := new(SVM_Problem)
				si := start[i]
				sj := start[j]
				ci := count[i]
				cj := count[j]
				sub_prob.L = ci + cj
				sub_prob.X = make([][]SVM_Node, sub_prob.L)
				sub_prob.Y = make([]float64, sub_prob.L)
				var k int
				for k = 0; k < ci; k++ {
					sub_prob.X[k] = x[si+k]
					sub_prob.Y[k] = +1
				}
				for k = 0; k < cj; k++ {
					sub_prob.X[ci+k] = x[sj+k]
					sub_prob.Y[ci+k] = -1
				}

				if param.Probability == 1 {
					probAB := make([]float64, 2)
					this.SVM_binary_svc_probability(sub_prob, param, weighted_C[i], weighted_C[j], probAB)
					probA[p] = probAB[0]
					probB[p] = probAB[1]
				}

				f[p] = *this.SVM_train_one(sub_prob, param, weighted_C[i], weighted_C[j])
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

		model.Nr_class = nr_class

		model.Label = make([]int, nr_class)
		for i = 0; i < nr_class; i++ {
			model.Label[i] = label[i]
		}

		model.Rho = make([]float64, nr_class*(nr_class-1)/2)
		for i = 0; i < nr_class*(nr_class-1)/2; i++ {
			model.Rho[i] = f[i].rho
		}

		if param.Probability == 1 {
			model.ProbA = make([]float64, nr_class*(nr_class-1)/2)
			model.ProbB = make([]float64, nr_class*(nr_class-1)/2)
			for i = 0; i < nr_class*(nr_class-1)/2; i++ {
				model.ProbA[i] = probA[i]
				model.ProbB[i] = probB[i]
			}
		} else {
			model.ProbA = nil
			model.ProbB = nil
		}

		nnz := 0
		nz_count := make([]int, nr_class)
		model.NSV = make([]int, nr_class)
		for i = 0; i < nr_class; i++ {
			nSV := 0
			for j := 0; j < count[i]; j++ {
				if nonzero[start[i]+j] {
					nSV++
					nnz++
				}
			}
			model.NSV[i] = nSV
			nz_count[i] = nSV
		}

		log.Println("Total nSV = ", nnz)

		model.L = nnz
		model.SV = make([][]SVM_Node, nnz)
		model.Sv_indices = make([]int, nnz)
		p = 0
		for i = 0; i < l; i++ {
			if nonzero[i] {
				model.SV[p] = x[i]
				model.Sv_indices[p] = perm[i] + 1
				p++
			}
		}

		nz_start := make([]int, nr_class)
		nz_start[0] = 0
		for i = 1; i < nr_class; i++ {
			nz_start[i] = nz_start[i-1] + nz_count[i-1]
		}

		model.Sv_coef = make([][]float64, nr_class-1)
		for i = 0; i < nr_class-1; i++ {
			model.Sv_coef[i] = make([]float64, nnz)
		}

		p = 0
		for i = 0; i < nr_class; i++ {
			for j := i + 1; j < nr_class; j++ {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				si := start[i]
				sj := start[j]
				ci := count[i]
				cj := count[j]

				q := nz_start[i]
				var k int
				for k = 0; k < ci; k++ {
					if nonzero[si+k] {
						model.Sv_coef[j-1][q] = f[p].alpha[k]
						q++
					}
				}
				q = nz_start[j]
				for k = 0; k < cj; k++ {
					if nonzero[sj+k] {
						model.Sv_coef[i][q] = f[p].alpha[ci+k]
						q++
					}
				}
				p++
			}
		}
	}
	return model
}

func (this *SVM) SVM_cross_validation(prob *SVM_Problem, param *SVM_Parameter, nr_fold int, target []float64) {
	var i int
	fold_start := make([]int, nr_fold+1)
	l := prob.L
	perm := make([]int, l)

	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if (param.Svm_type == C_SVC ||
		param.Svm_type == NU_SVC) && nr_fold < l {
		tmp_nr_class := make([]int, 1)
		tmp_label := make([][]int, 1)
		tmp_start := make([][]int, 1)
		tmp_count := make([][]int, 1)

		this.SVM_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm)

		nr_class := tmp_nr_class[0]
		start := tmp_start[0]
		count := tmp_count[0]

		// random shuffle and then data grouped by fold using the array perm
		fold_count := make([]int, nr_fold)
		var c int
		index := make([]int, l)
		for i = 0; i < l; i++ {
			index[i] = perm[i]
		}
		for c = 0; c < nr_class; c++ {
			for i = 0; i < count[c]; i++ {
				j := i + int(rand.Int31n(int32(count[c]-i)))
				//do {int _=index[start[c]+j]; index[start[c]+j]=index[start[c]+i]; index[start[c]+i]=_;} while(false);
				index[start[c]+i], index[start[c]+j] = index[start[c]+j], index[start[c]+i]
			}
		}
		for i = 0; i < nr_fold; i++ {
			fold_count[i] = 0
			for c = 0; c < nr_class; c++ {
				fold_count[i] += (i+1)*count[c]/nr_fold - i*count[c]/nr_fold
			}
		}
		fold_start[0] = 0
		for i = 1; i <= nr_fold; i++ {
			fold_start[i] = fold_start[i-1] + fold_count[i-1]
		}
		for c = 0; c < nr_class; c++ {
			for i = 0; i < nr_fold; i++ {
				begin := start[c] + i*count[c]/nr_fold
				end := start[c] + (i+1)*count[c]/nr_fold
				for j := begin; j < end; j++ {
					perm[fold_start[i]] = index[j]
					fold_start[i]++
				}
			}
		}
		fold_start[0] = 0
		for i = 1; i <= nr_fold; i++ {
			fold_start[i] = fold_start[i-1] + fold_count[i-1]
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
		for i = 0; i <= nr_fold; i++ {
			fold_start[i] = i * l / nr_fold
		}
	}

	for i = 0; i < nr_fold; i++ {
		begin := fold_start[i]
		end := fold_start[i+1]
		var j, k int
		subprob := new(SVM_Problem)

		subprob.L = l - (end - begin)
		subprob.X = make([][]SVM_Node, subprob.L)
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
		submodel := this.SVM_train(subprob, param)
		if param.Probability == 1 &&
			(param.Svm_type == C_SVC ||
				param.Svm_type == NU_SVC) {
			prob_estimates := make([]float64, this.SVM_get_nr_class(submodel))
			for j = begin; j < end; j++ {
				target[perm[j]] = this.SVM_predict_probability(submodel, prob.X[perm[j]], prob_estimates)
			}
		} else {
			for j = begin; j < end; j++ {
				target[perm[j]] = this.SVM_predict(submodel, prob.X[perm[j]])
			}
		}
	}
}

func (this *SVM) SVM_get_svm_type(model *SVM_Model) int {
	return model.Param.Svm_type
}

func (this *SVM) SVM_get_nr_class(model *SVM_Model) int {
	return model.Nr_class
}

func (this *SVM) SVM_get_labels(model *SVM_Model, label []int) {
	if model.Label != nil {
		for i := 0; i < model.Nr_class; i++ {
			label[i] = model.Label[i]
		}
	}
}

func (this *SVM) SVM_get_sv_indices(model SVM_Model, indices []int) {
	if model.Sv_indices != nil {
		for i := 0; i < model.L; i++ {
			indices[i] = model.Sv_indices[i]
		}
	}
}

func (this *SVM) SVM_get_nr_sv(model SVM_Model) int {
	return model.L
}

func (this *SVM) SVM_get_svr_probability(model *SVM_Model) float64 {
	var rst float64
	if (model.Param.Svm_type == EPSILON_SVR || model.Param.Svm_type == NU_SVR) && model.ProbA != nil {
		rst = model.ProbA[0]
	} else {
		log.Fatal("Model doesn't contain information for SVR probability inference\n")
		rst = 0
	}
	return rst
}

func (this *SVM) SVM_predict_values(model *SVM_Model, x []SVM_Node, dec_values []float64) float64 {
	var i int
	var rst float64
	if model.Param.Svm_type == ONE_CLASS ||
		model.Param.Svm_type == EPSILON_SVR ||
		model.Param.Svm_type == NU_SVR {
		sv_coef := model.Sv_coef[0]
		sum := float64(0)
		for i = 0; i < model.L; i++ {
			sum += sv_coef[i] * k_function(x, model.SV[i], model.Param)
		}
		sum -= model.Rho[0]
		dec_values[0] = sum

		if model.Param.Svm_type == ONE_CLASS {
			if sum > 0 {
				rst = 1
			} else {
				rst = -1
			}
		} else {
			rst = sum
		}
	} else {
		nr_class := model.Nr_class
		l := model.L

		kvalue := make([]float64, l)
		for i = 0; i < l; i++ {
			kvalue[i] = k_function(x, model.SV[i], model.Param)
		}

		start := make([]int, nr_class)
		start[0] = 0
		for i = 1; i < nr_class; i++ {
			start[i] = start[i-1] + model.NSV[i-1]
		}

		vote := make([]int, nr_class)
		for i = 0; i < nr_class; i++ {
			vote[i] = 0
		}

		p := 0
		for i = 0; i < nr_class; i++ {
			for j := i + 1; j < nr_class; j++ {
				sum := float64(0)
				si := start[i]
				sj := start[j]
				ci := model.NSV[i]
				cj := model.NSV[j]

				var k int
				coef1 := model.Sv_coef[j-1]
				coef2 := model.Sv_coef[i]
				for k = 0; k < ci; k++ {
					sum += coef1[si+k] * kvalue[si+k]
				}
				for k = 0; k < cj; k++ {
					sum += coef2[sj+k] * kvalue[sj+k]
				}
				sum -= model.Rho[p]
				dec_values[p] = sum

				if dec_values[p] > 0 {
					vote[i]++
				} else {
					vote[j]++
				}
				p++
			}
		}

		vote_max_idx := 0
		for i = 1; i < nr_class; i++ {
			if vote[i] > vote[vote_max_idx] {
				vote_max_idx = i
			}
		}
		rst = float64(model.Label[vote_max_idx])
	}
	return rst
}

func (this *SVM) SVM_predict(model *SVM_Model, x []SVM_Node) float64 {
	nr_class := model.Nr_class
	var dec_values []float64
	if model.Param.Svm_type == ONE_CLASS ||
		model.Param.Svm_type == EPSILON_SVR ||
		model.Param.Svm_type == NU_SVR {
		dec_values = make([]float64, 1)
	} else {
		dec_values = make([]float64, nr_class*(nr_class-1)/2)
	}
	pred_result := this.SVM_predict_values(model, x, dec_values)
	return pred_result
}

func (this *SVM) SVM_predict_probability(model *SVM_Model, x []SVM_Node, prob_estimates []float64) float64 {
	var rst float64
	if (model.Param.Svm_type == C_SVC ||
		model.Param.Svm_type == NU_SVC) &&
		model.ProbA != nil && model.ProbB != nil {
		var i int
		nr_class := model.Nr_class
		dec_values := make([]float64, nr_class*(nr_class-1)/2)
		this.SVM_predict_values(model, x, dec_values)

		min_prob := float64(1e-7)
		pairwise_prob := make([][]float64, nr_class)
		for i = range pairwise_prob {
			pairwise_prob[i] = make([]float64, nr_class)
		}

		k := 0
		for i = 0; i < nr_class; i++ {
			for j := i + 1; j < nr_class; j++ {
				pairwise_prob[i][j] = math.Min(math.Max(this.sigmoid_predict(dec_values[k], model.ProbA[k], model.ProbB[k]), min_prob), 1-min_prob)
				pairwise_prob[j][i] = 1 - pairwise_prob[i][j]
				k++
			}
		}
		this.multiclass_probability(nr_class, pairwise_prob, prob_estimates)

		prob_max_idx := 0
		for i = 1; i < nr_class; i++ {
			if prob_estimates[i] > prob_estimates[prob_max_idx] {
				prob_max_idx = i
			}
		}
		rst = float64(model.Label[prob_max_idx])
	} else {
		rst = this.SVM_predict(model, x)
	}
	return rst
}

func (this *SVM) SVM_save_model(model_file_name string, model *SVM_Model) {

	fp, _ := os.OpenFile(model_file_name, os.O_CREATE|os.O_WRONLY|os.O_SYNC, 0644)
	defer fp.Close()
	fb := bufio.NewWriter(fp)
	param := model.Param

	fb.WriteString("svm_type " + this.svm_type_table[param.Svm_type] + "\n")
	fb.WriteString("kernel_type " + this.kernel_type_table[param.Kernel_type] + "\n")

	if param.Kernel_type == POLY {
		fb.WriteString("degree " + strconv.Itoa(param.Degree) + "\n")
	}

	if param.Kernel_type == POLY ||
		param.Kernel_type == RBF ||
		param.Kernel_type == SIGMOID {
		fb.WriteString("gamma " + strconv.FormatFloat(param.Gamma, 'g', -1, 64) + "\n")
	}

	if param.Kernel_type == POLY ||
		param.Kernel_type == SIGMOID {
		fb.WriteString("coef0 " + strconv.FormatFloat(param.Coef0, 'g', -1, 64) + "\n")
	}

	nr_class := model.Nr_class
	l := model.L
	fb.WriteString("nr_class " + strconv.Itoa(nr_class) + "\n")
	fb.WriteString("total_sv " + strconv.Itoa(l) + "\n")

	{
		fb.WriteString("rho")
		for i := 0; i < nr_class*(nr_class-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.Rho[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}

	if model.Label != nil {
		fb.WriteString("label")
		for i := 0; i < nr_class; i++ {
			fb.WriteString(" " + strconv.Itoa(model.Label[i]))
		}
		fb.WriteString("\n")
	}

	if model.ProbA != nil { // regression has probA only
		fb.WriteString("probA")
		for i := 0; i < nr_class*(nr_class-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.ProbA[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}
	if model.ProbB != nil {
		fb.WriteString("probB")
		for i := 0; i < nr_class*(nr_class-1)/2; i++ {
			fb.WriteString(" " + strconv.FormatFloat(model.ProbB[i], 'g', -1, 64))
		}
		fb.WriteString("\n")
	}

	if model.NSV != nil {
		fb.WriteString("nr_sv")
		for i := 0; i < nr_class; i++ {
			fb.WriteString(" " + strconv.Itoa(model.NSV[i]))
		}
		fb.WriteString("\n")
	}

	fb.WriteString("SV\n")
	sv_coef := model.Sv_coef
	SV := model.SV

	for i := 0; i < l; i++ {
		for j := 0; j < nr_class-1; j++ {
			fb.WriteString(strconv.FormatFloat(sv_coef[j][i], 'g', -1, 64) + " ")
		}

		p := SV[i]
		if param.Kernel_type == PRECOMPUTED {
			fb.WriteString("0:" + strconv.Itoa(int(p[0].Value)))
		} else {
			for j := 0; j < len(p); j++ {
				fb.WriteString(strconv.Itoa(p[j].Index) + ":" + strconv.FormatFloat(p[j].Value, 'g', -1, 64) + " ")
			}
			fb.WriteString("\n")
		}
	}
}

func (this *SVM) aotf(s string) float64 {
	rst, _ := strconv.ParseFloat(s, 64)
	return rst
}
func (this *SVM) atoi(s string) int {
	rst, _ := strconv.Atoi(s)
	return rst
}

/*
func (this *SVM) svm_load_model(model_file_name string) {
	file, err := os.Open(model_file_name)
	defer file.Close()
	fb := bufio.NewReader(file)
	model := new(SVM_Model)
	param := new(SVM_Parameter)
	model.param = param
	model.rho = nil
	model.probA = nil
	model.probB = nil
	model.label = nil
	model.nSV = nil

	for {
		cmd, _ := fb.ReadLine()
		arg := cmd.substring(cmd.indexOf(' ')+1);

		if cmd.startsWith("svm_type") {
			var i int
			for i=0;i< len(this.svm_type_table);i++{
				if arg.indexOf(this.svm_type_table[i])!=-1 {
					param.svm_type=i
					break
				}
			}
			if i == len(this.svm_type_table) {
				log.Fatal("unknown svm type.\n")
				return nil
			}
		} else if cmd.startsWith("kernel_type") {
			var i int
			for i=0;i<len(this.kernel_type_table);i++ {
				if arg.indexOf(this.kernel_type_table[i])!=-1 {
					param.kernel_type=i
					break
				}
			}
			if i == len(this.kernel_type_table) {
				log.Fatal("unknown kernel function.\n")
				return nil
			}
		} else if cmd.startsWith("degree") {
			param.degree = atoi(arg)
		} else if cmd.startsWith("gamma") {
			param.gamma = atof(arg)
		} else if cmd.startsWith("coef0") {
			param.coef0 = atof(arg)
		} else if cmd.startsWith("nr_class") {
			model.nr_class = atoi(arg)
		} else if cmd.startsWith("total_sv") {
			model.l = atoi(arg)
		} else if cmd.startsWith("rho") {
			n := model.nr_class * (model.nr_class-1)/2
			model.rho = make([]float64,n)
			// st := new StringTokenizer(arg);
			for i:=0;i<n;i++ {
				model.rho[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("label") {
			n := model.nr_class
			model.label = make([]int,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i=0;i<n;i++ {
				model.label[i] = atoi(st.nextToken())
			}
		} else if cmd.startsWith("probA") {
			n := model.nr_class*(model.nr_class-1)/2
			model.probA = make([]float64,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i:=0;i<n;i++ {
				model.probA[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("probB") {
			n := model.nr_class*(model.nr_class-1)/2;
			model.probB = make([]float64,n)
			//StringTokenizer st = new StringTokenizer(arg);
			for i:=0;i<n;i++ {
					model.probB[i] = atof(st.nextToken())
			}
		} else if cmd.startsWith("nr_sv") {
			n := model.nr_class
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

	// read sv_coef and SV

	m := model.nr_class - 1;
	l := model.l;
	//model.sv_coef = new float64[m][l];
	model.sv_coef = make([][]float64, m);
	for i := range model.sv_coef {
		model.sv_coef[i] = make([]float64, l)
	}
	model.SV = make([][]SVM_Node,l)

	for i:=0;i<l;i++ {
		line := fb.ReadLine();
		//StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

		for k:=0;k<m;k++ {
				model.sv_coef[k][i] = atof(st.nextToken())
		}
		n := st.countTokens()/2;
		model.SV[i] = make([]SVM_Node,n)
		for j:=0;j<n;j++ {
			model.SV[i][j] = new(SVM_Node)
			model.SV[i][j].index = atoi(st.nextToken())
			model.SV[i][j].value = atof(st.nextToken())
		}
	}
	return model
}
*/
func (this *SVM) SVM_check_parameter(prob *SVM_Problem, param *SVM_Parameter) string {
	// svm_type

	svm_type := param.Svm_type
	if svm_type != C_SVC &&
		svm_type != NU_SVC &&
		svm_type != ONE_CLASS &&
		svm_type != EPSILON_SVR &&
		svm_type != NU_SVR {
		return "unknown svm type"
	}

	// kernel_type, degree

	kernel_type := param.Kernel_type
	if kernel_type != LINEAR &&
		kernel_type != POLY &&
		kernel_type != RBF &&
		kernel_type != SIGMOID &&
		kernel_type != PRECOMPUTED {
		return "unknown kernel type"
	}

	if param.Gamma < 0 {
		return "gamma < 0"
	}

	if param.Degree < 0 {
		return "degree of polynomial kernel < 0"
	}

	// cache_size,eps,C,nu,p,shrinking

	if param.Cache_size <= 0 {
		return "cache_size <= 0"
	}

	if param.Eps <= 0 {
		return "eps <= 0"
	}

	if svm_type == C_SVC ||
		svm_type == EPSILON_SVR ||
		svm_type == NU_SVR {
		if param.C <= 0 {
			return "C <= 0"
		}
	}
	if svm_type == NU_SVC ||
		svm_type == ONE_CLASS ||
		svm_type == NU_SVR {
		if param.Nu <= 0 || param.Nu > 1 {
			return "nu <= 0 or nu > 1"
		}
	}
	if svm_type == EPSILON_SVR {
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
		svm_type == ONE_CLASS {
		return "one-class SVM probability output not supported yet"
	}

	// check whether nu-svc is feasible

	if svm_type == NU_SVC {
		l := prob.L
		max_nr_class := 16
		nr_class := 0
		label := make([]int, max_nr_class)
		count := make([]int, max_nr_class)

		var i int
		for i = 0; i < l; i++ {
			this_label := int(prob.Y[i])
			var j int
			for j = 0; j < nr_class; j++ {
				if this_label == label[j] {
					count[j]++
					break
				}
			}
			if j == nr_class {
				if nr_class == max_nr_class {
					max_nr_class *= 2
					new_data := make([]int, max_nr_class)
					//System.arraycopy(label,0,new_data,0,label.length);
					copy(new_data, label)
					label = new_data
					new_data = make([]int, max_nr_class)
					//System.arraycopy(count,0,new_data,0,count.length);
					copy(new_data, count)
					count = new_data
				}
				label[nr_class] = this_label
				count[nr_class] = 1
				nr_class++
			}
		}

		for i = 0; i < nr_class; i++ {
			n1 := count[i]
			for j := i + 1; j < nr_class; j++ {
				n2 := count[j]
				if param.Nu*float64(n1+n2)/2 > math.Min(float64(n1), float64(n2)) {
					return "specified nu is infeasible"
				}
			}
		}
	}
	return ""
}

func (this *SVM) SVM_check_probability_model(model *SVM_Model) int {
	var rst int
	if ((model.Param.Svm_type == C_SVC ||
		model.Param.Svm_type == NU_SVC) && model.ProbA != nil && model.ProbB != nil) ||
		((model.Param.Svm_type == EPSILON_SVR || model.Param.Svm_type == NU_SVR) && model.ProbA != nil) {
		rst = 1
	} else {
		rst = 0
	}
	return rst
}
