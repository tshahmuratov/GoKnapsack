package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

/////////////////////////////////////////////////////////////
// Global vars
var solverWG sync.WaitGroup
var integerSolution IntegerSolution
var simplexSolutionChannel chan *SimplexSolution
var nodesChannel chan *TreeNode

/////////////////////////////////////////////////////////////
// Consts
const NOMINAL_VALUE = 1000
const INDEX_NOT_FOUND = -1
const TOLERANCE = 0.000001
const INVALID_Z = -1
const MAX_WORKERS = 100

/////////////////////////////////////////////////////////////
// Types
/////////////////////////////////////////////////////////////
// InputParams struct for input params
type InputParams struct {
	N    int32
	M    int32
	S    int32
	Lots []Lot
}

/////////////////////////////////////////////////////////////
// Lot struct for lot
type Lot struct {
	Day   int32
	Name  string
	Price int32
	Count int32
}

// GetProfit get lot profit
func (l *Lot) GetProfit(N int32) int32 {
	return (NOMINAL_VALUE - l.Price + // price profit
		(N + 30 - l.Day)) * // coupon
		l.Count // multiplied by count
}

// GetCost get lot cost
func (l *Lot) GetCost() int32 {
	return l.Price * l.Count
}

// Print print lot
func (l *Lot) Print() {
	fmt.Printf("%v %v %.1f %v\n", l.Day, l.Name, float64(l.Price)/10, l.Count)
}

/////////////////////////////////////////////////////////////
// TreeNode type for branch and bounds
type TreeNode struct {
	Parent *TreeNode
	I      int32
	Val    bool
}

func (n *TreeNode) GetDepth() int32 {
	node := n
	count := int32(0)
	for node.Parent != nil {
		count++
		node = node.Parent
	}
	return count
}

/////////////////////////////////////////////////////////////
// IntegerSolution resulted by B&B integer solution
type IntegerSolution struct {
	BestZ     int32
	Incumbent []bool
}

/////////////////////////////////////////////////////////////
// SimplexSolution resulted by simplex method solution
type SimplexSolution struct {
	Node       *TreeNode
	Result     []float32
	Infeasible bool
	Z          float32
}

// IsInteger checks whether simplex solution is integer
func (s *SimplexSolution) IsInteger() bool {
	for _, val := range s.Result {
		if val != float32(math.Trunc(float64(val))) {
			return false
		}
	}
	return true
}

/////////////////////////////////////////////////////////////
// SimplexTable table with data for simplex method
type SimplexTable struct {
	Data [][]float32
}

/////////////////////////////////////////////////////////////
// ObjectiveFunction vector of coefficient of objective function
type ObjectiveFunction struct {
	C []int32
}

/////////////////////////////////////////////////////////////
// Constraint vector of coefficient of constraints. It should be matrix but constraint is single
type Constraint struct {
	A []int32
	B int32
}

/////////////////////////////////////////////////////////////
// Helpers
// Round unfortunately golang has no round
func Round(val float64) int32 {
	if val < 0 {
		return int32(val - 0.5)
	}
	return int32(val + 0.5)
}

func FloatIs1(val float32) bool {
	return math.Abs(
		float64(
			val-1.0,
		),
	) < TOLERANCE
}

func FloatIs0(val float32) bool {
	return math.Abs(
		float64(
			val-0.0,
		),
	) < TOLERANCE
}

/////////////////////////////////////////////////////////////
// work with input params
func getInputParams() *InputParams {
	var err error
	params := InputParams{}
	if len(os.Args) < 2 {
		panic("No filename provided\n")
	}
	filename := os.Args[1]
	inFile, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Coudn't open filename %s: %s\n", filename, err)
		os.Exit(1)
	}
	defer inFile.Close()
	scanner := bufio.NewScanner(inFile)
	scanner.Split(bufio.ScanLines)

	//read first line
	scanner.Scan()
	lineParams := strings.Split(scanner.Text(), " ")
	if len(lineParams) < 3 {
		fmt.Printf("Wrong first line %s\n", scanner.Text())
		os.Exit(1)
	}

	inN, _ := strconv.ParseInt(lineParams[0], 10, 32)
	inM, _ := strconv.ParseInt(lineParams[1], 10, 32)
	inS, _ := strconv.ParseFloat(lineParams[2], 32)

	params.N = int32(inN)
	params.M = int32(inM)
	params.S = int32(inS) // assume money - integer

	params.Lots = make([]Lot, 0, int64(
		math.Ceil(
			float64(
				params.N*params.M/2, // assume normal distribution with mean M * N / 2
			),
		),
	))

	for scanner.Scan() {
		lineParams = strings.Split(scanner.Text(), " ")
		if len(lineParams) < 4 {
			fmt.Printf("Wrong line %s\n", scanner.Text())
			os.Exit(1)
		}

		inDay, _ := strconv.ParseInt(lineParams[0], 10, 32)
		inPrice, _ := strconv.ParseFloat(lineParams[2], 32)
		inCount, _ := strconv.ParseInt(lineParams[3], 10, 32)

		// assume min price step 0.1
		params.Lots = append(params.Lots, Lot{
			Day:   int32(inDay),
			Name:  lineParams[1],
			Price: Round(inPrice * 10),
			Count: int32(inCount),
		})
	}
	return &params
}

func removeNonProfitLots(params *InputParams) {
	for i, val := range params.Lots {
		if val.GetProfit(params.N) <= 0 {
			params.Lots = append(params.Lots[:i], params.Lots[i+1:]...)
		}
	}
}

func printSolution(params *InputParams) {
	if integerSolution.BestZ == INVALID_Z {
		fmt.Println("No solution")
		os.Exit(1)
	}
	fmt.Println(integerSolution.BestZ)
	for i, val := range integerSolution.Incumbent {
		if val {
			params.Lots[i].Print()
		}
	}
}

/////////////////////////////////////////////////////////////
// prepare data for branch and bound
func getObjectiveFunction(params *InputParams) *ObjectiveFunction {
	objective := ObjectiveFunction{}
	objective.C = make([]int32, 0, len(params.Lots))
	for _, val := range params.Lots {
		objective.C = append(objective.C, val.GetProfit(params.N))
	}
	return &objective
}

func getConstraint(params *InputParams) *Constraint {
	constraint := Constraint{}
	constraint.A = make([]int32, 0, len(params.Lots))
	for _, val := range params.Lots {
		constraint.A = append(constraint.A, val.GetCost())
	}
	constraint.B = params.S
	return &constraint
}

/////////////////////////////////////////////////////////////
// simplex solving
func createSimplexTable(node *TreeNode, objective *ObjectiveFunction, constraint *Constraint) *SimplexTable {
	depth := node.GetDepth()
	varCount := int32(len(objective.C)) - depth
	rowNum := 1 + // main constraint
		varCount + // each var should be less than 1
		1 // indicator row
	colNum := varCount + // count of variables
		rowNum - 1 + // for converting inequalities to equalities
		1 //bi row
	table := SimplexTable{}
	table.Data = make([][]float32, rowNum)
	for i := range table.Data {
		table.Data[i] = make([]float32, colNum)
	}
	//get new B0
	subtractFromMoney := int32(0)
	for node.Parent != nil {
		if node.Val {
			subtractFromMoney += constraint.A[node.I]
		}
		node = node.Parent
	}
	newB := constraint.B - subtractFromMoney
	// fill bi colNum
	table.Data[0][colNum-1] = float32(newB)
	for i := int32(1); i < rowNum-1; i++ {
		table.Data[i][colNum-1] = 1
	}
	table.Data[rowNum-1][colNum-1] = 0
	//fill main constraints
	for i := int32(0); i < varCount; i++ {
		table.Data[0][i] = float32(constraint.A[i+depth])
	}
	// fill constraint xi <= 1
	for i := int32(0); i < varCount; i++ {
		for j := int32(0); j < varCount; j++ {
			if i == j {
				table.Data[i+1][j] = 1
			} else {
				table.Data[i+1][j] = 0
			}
		}
	}
	// fill  for converting inequalities to equalities
	for i := int32(0); i < rowNum-1; i++ {
		for j := int32(0); j < rowNum-1; j++ {
			if i == j {
				table.Data[i][j+varCount] = 1
			} else {
				table.Data[i][j+varCount] = 0
			}
		}
	}
	// fill indicator row
	for j := int32(0); j < colNum-1; j++ {
		if j < varCount {
			table.Data[1+varCount][j] = float32(-objective.C[j+depth])
		} else {
			table.Data[1+varCount][j] = 0
		}
	}
	return &table
}

func getLowestColIndex(table *SimplexTable) int {
	lastRow := len(table.Data) - 1
	lastCol := len(table.Data[lastRow])
	lowestIndex := INDEX_NOT_FOUND
	for i := 0; i < lastCol-1; i++ {
		elem := table.Data[lastRow][i]
		if elem < 0 {
			if lowestIndex == INDEX_NOT_FOUND || elem < table.Data[lastRow][lowestIndex] {
				lowestIndex = i
			}
		}
	}
	return lowestIndex
}

func getHighestRowIndex(table *SimplexTable, colIndex int) int {
	lowestValue := float32(-1.0)
	lowestIndex := INDEX_NOT_FOUND
	lastCol := len(table.Data[0]) - 1
	for i := range table.Data {
		ratio := table.Data[i][lastCol] / table.Data[i][colIndex]
		if ratio > 0 {
			if lowestIndex == INDEX_NOT_FOUND || ratio < lowestValue {
				lowestValue = ratio
				lowestIndex = i
			}
		}
	}
	return lowestIndex
}

func solveSimplexTable(table *SimplexTable, solution *SimplexSolution) {
	rowCount := len(table.Data)
	colCount := len(table.Data[0])
	lastRow := len(table.Data) - 1
	lastCol := len(table.Data[0]) - 1

	B0 := table.Data[0][lastCol]
	if B0 < 0 {
		solution.Infeasible = true
		return
	}
	colIndex := getLowestColIndex(table)
	for colIndex != INDEX_NOT_FOUND {
		rowIndex := getHighestRowIndex(table, colIndex)
		if rowIndex == INDEX_NOT_FOUND {
			solution.Infeasible = true
			return
		}

		//TODO loop unwinding possible, SIMD?
		//normalize row
		divider := table.Data[rowIndex][colIndex]
		for j := range table.Data[rowIndex] {
			table.Data[rowIndex][j] /= divider
		}

		//update all rows using pivot
		for i := range table.Data {
			if i == rowIndex {
				continue
			}
			multiplier := table.Data[i][colIndex]
			if FloatIs0(multiplier) {
				continue
			}
			for j := range table.Data[i] {
				table.Data[i][j] -= table.Data[rowIndex][j] * multiplier
			}
		}

		colIndex = getLowestColIndex(table)
	}
	//fill solution
	solution.Result = make([]float32, 0, colCount-1)
	for j := 0; j < colCount-1; j++ {
		if table.Data[lastRow][j] > 0 {
			solution.Result = append(solution.Result, 0)
		} else {
			for i := 0; i < rowCount; i++ {
				if FloatIs1(table.Data[i][j]) {
					solution.Result = append(solution.Result, table.Data[i][lastCol])
				}
			}
		}
	}
	solution.Z = table.Data[lastRow][lastCol]
}

/////////////////////////////////////////////////////////////
// branch and bounds controller
func simplexSolverWorker(objective *ObjectiveFunction, constraint *Constraint) {
	defer solverWG.Done()
	node := <-nodesChannel
	solution := SimplexSolution{
		Node:       node,
		Infeasible: false,
	}
	table := createSimplexTable(node, objective, constraint)
	solveSimplexTable(table, &solution)
	simplexSolutionChannel <- &solution
}

func workerController(objective *ObjectiveFunction, constraint *Constraint) {
	defer solverWG.Done()
	workerCount := 1
	for workerCount > 0 {
		simplexSolution := <-simplexSolutionChannel
		workerCount--
		if !simplexSolution.Infeasible {
			if simplexSolution.IsInteger() {
				newZ := int32(simplexSolution.Z)
				if integerSolution.BestZ < newZ {
					integerSolution.BestZ = newZ
					integerSolution.Incumbent = integerSolution.Incumbent[:0]
					hasVars := 0
					node := simplexSolution.Node
					for node.Parent != nil {
						//insert into beginning
						integerSolution.Incumbent = append([]bool{simplexSolution.Node.Val}, integerSolution.Incumbent...)
						node = node.Parent
						hasVars++
					}
					varCount := len(objective.C)
					for i := 0; i < varCount-hasVars; i++ {
						integerSolution.Incumbent = append(integerSolution.Incumbent, FloatIs1(simplexSolution.Result[i]))
					}
				}

			} else {
				leftNode := TreeNode{
					Parent: simplexSolution.Node,
					I:      simplexSolution.Node.I + 1,
					Val:    false,
				}
				rightNode := TreeNode{
					Parent: simplexSolution.Node,
					I:      simplexSolution.Node.I + 1,
					Val:    true,
				}
				nodesChannel <- &leftNode
				nodesChannel <- &rightNode
				solverWG.Add(2)
				go simplexSolverWorker(objective, constraint)
				go simplexSolverWorker(objective, constraint)
				workerCount += 2
			}
		}
	}
}

func solveBranchAndBound(objective *ObjectiveFunction, constraint *Constraint) {
	rootNode := TreeNode{
		Parent: nil,
		I:      -1,
	}
	integerSolution = IntegerSolution{
		BestZ: INVALID_Z,
	}
	simplexSolutionChannel = make(chan *SimplexSolution, MAX_WORKERS)
	defer close(simplexSolutionChannel)
	nodesChannel = make(chan *TreeNode, MAX_WORKERS)
	defer close(nodesChannel)
	solverWG.Add(1)
	go workerController(objective, constraint)
	nodesChannel <- &rootNode
	solverWG.Add(1)
	go simplexSolverWorker(objective, constraint)
	solverWG.Wait()
}

/////////////////////////////////////////////////////////////
// main
func main() {
	params := getInputParams()
	removeNonProfitLots(params)
	objectiveFunction := getObjectiveFunction(params)
	constraint := getConstraint(params)
	solveBranchAndBound(objectiveFunction, constraint)
	printSolution(params)
}
