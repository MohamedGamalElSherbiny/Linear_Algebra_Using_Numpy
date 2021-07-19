from LinearAlgebra import LinearAlgebra

equations = [[1.5, 5.75, 2.6], [1, 1, 1], [0, 1, -1]]
answer = [589.5, 200, -20]
linearAlgebra = LinearAlgebra(equations, answer)
print(linearAlgebra.solution)