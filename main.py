from SolveLinearEquation import SolveLinearEquation

matrix = [[1.5, 5.75, 2.6], [1, 1, 1], [0, 1, -1]]
target_vector = [589.5, 200, -20]
solveLinearEquation = SolveLinearEquation(matrix, target_vector)
print(solveLinearEquation.solution)