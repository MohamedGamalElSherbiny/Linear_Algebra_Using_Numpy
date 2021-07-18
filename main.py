from SolveLinearEquation import SolveLinearEquation

matrix = [[1.5, 5.75, 2.6], [1, 1, 1], [0, 1, -1]]
solveLinearEquation = SolveLinearEquation(matrix)
target_vector = [589.5, 200, -20]
print(solveLinearEquation.solve_linear_equation(target_vector))