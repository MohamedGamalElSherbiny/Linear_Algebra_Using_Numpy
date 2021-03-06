import numpy as np

class LinearAlgebra:
    """
    Implements a set of Linear Algebra Functions on a given Matrix

    Parameters:
    -----------
    matrix :             list of lists
                         each embedded list represents a row
                         list of integers / float numbers
    target_vector :      list
                         list of integer / float numbers with the final answer of the equations
    print_properties :   boolean, optional
                         To print the rank and the determinant of the matrix

    Returns:
    -------
    solution:        numpy
                     Matrix in the echelon form
    """
    def __init__(self, matrix, target_vector, print_properties=False):
        self.matrix = np.array(matrix, dtype=np.float64)
        self.target_vector = np.array(target_vector).reshape(-1, 1)
        self.num_rows = len(self.matrix)
        self.print_properties = print_properties
        self.solution = self._solve_linear_equation()

    def __is_square(self):
        """
        Checks if the matrix sent is a square matrix or not.

        Parameters:
        -----------
        matrix : numpy array
                 Matrix of integers / float numbers

        Returns:
        -------
        Boolean
        """
        # Get the number of columns in the matrix
        for row in self.matrix:
            if self.num_rows == len(row):
                return True
            return False

    def __matrix_properties(self):
        """
        Checks if the matrix columns are independent.

        Parameters:
        -----------
        matrix :             numpy array
                             Matrix of integers / float numbers

        Returns:
        -------
        Boolean

        See also:
        --------
        Prints the matrix rank of the input matrix using np.linalg.matrix_rank
        Prints the determinant of the input matrix using np.linalg.det function
        """
        if self.__is_square():
            matrix_rank = np.linalg.matrix_rank(self.matrix)
            if self.print_properties:
                print(f"The rank of the matrix is: {matrix_rank}")
                print(f"The determinant of the matrix is: {self.det()}")
            if matrix_rank == self.num_rows:
                return True
            else:
                return False

    @staticmethod
    def __gaussian_reduction(matrix, pivot_row, target_row, pivot_element, checked=False):
        """
        Applies Gaussian Elimination on the given Matrix

        Parameters:
        -----------
        matrix :        numpy array
                        Matrix of integers / float numbers
        pivot_row :     int
                        The index of the pivot row
        target_row :    int
                        The index of the target row to reduce
        pivot_element:  int
                        The index of the pivot element
        checked:        boolean, optional
                        To overcome the infinite loop of re-checking a 0 pivot element

        Returns:
        -------
        Numpy Matrix

        See also:
        --------
        https://mathworld.wolfram.com/GaussianElimination.html
        https://en.wikipedia.org/wiki/Gaussian_elimination
        """
        matrix_copy = np.copy(matrix)
        if matrix[pivot_row, pivot_element] != 1:
            matrix_copy[pivot_row, :] /= matrix[pivot_row, pivot_element]
        elif matrix[pivot_row, pivot_element] == 0 and not checked:
            matrix[-1, :] = matrix[pivot_row, :]
            matrix[pivot_row, :] = matrix_copy[-1, :]
            LinearAlgebra.__gaussian_reduction(matrix, pivot_row, target_row, pivot_element, checked=True)
        elif checked:
            print("Matrix has no answer")
            return False
        else:
            return matrix
        target_element = matrix[target_row, pivot_element]
        matrix_copy[pivot_row, :] *= target_element
        matrix[target_row, :] = matrix_copy[target_row, :] - matrix_copy[pivot_row, :]
        return matrix

    def _solve_linear_equation(self):
        """
        Solves a system of linear equations using Gaussian Echelon Form

        Parameters:
        -----------
        matrix :         numpy array
                         Matrix of integers / float numbers
        target_vector :  numpy array
                         numpy vector of integer / float numbers with the final answer of the equations

        Returns:
        -------
        Numpy Matrix

        See also:
        --------
        https://mathworld.wolfram.com/GaussianElimination.html
        https://en.wikipedia.org/wiki/Gaussian_elimination
        """
        # To check if the matrix already has a solution or not
        matrix_solution = self.__matrix_properties()
        if matrix_solution:
            # To concatenate the final vector to the matrix
            concatenated_matrix = np.concatenate((self.matrix, self.target_vector), axis=1)
            for i in range(len(concatenated_matrix) - 1):
                concatenated_matrix = self.__gaussian_reduction(concatenated_matrix, pivot_row=i, target_row=i + 1,
                                                                pivot_element=i)
            for row in range(len(concatenated_matrix)):
                concatenated_matrix[row, :] /= concatenated_matrix[row, row]
            return concatenated_matrix
        else:
            return "There is no solution to this matrix.\nOr this matrix has infinite number of solutions."

    @staticmethod
    def __determinant(matrix):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    def __interchange_matrix(self):
        inverted_matrix = np.array([[self.matrix[1, 1], -1 * self.matrix[0, 1]],
                                    [-1 * self.matrix[1, 0], self.matrix[0, 0]]])
        return inverted_matrix

    def inverse_matrix(self):
        return (1 / self.__determinant(self.matrix)) * self.__interchange_matrix()

    # TODO: Coding a function that checks if a 3x3 matrix is invertible
    def __check_invertible(self):
        if self.det() == 0:
            return False
        else:
            return True

    # TODO: Coding a function that generates the transpose of a 3x3 matrix
    def transpose(self):
        transposed_matrix_list = [np.array(i) for i in zip(*self.matrix)]
        return np.array(transposed_matrix_list)

    # TODO: Coding a function that generates the matrix of minors of a 3x3 matrix
    @staticmethod
    def __minor(matrix, i, j):
        matrix = np.delete(matrix, i, axis=0)
        matrix = np.delete(matrix, j, axis=1)
        return matrix

    # TODO: Coding a function that generates the matrix of cofactors of a 3x3 matrix
    def det(self):
        if self.num_rows == 1:
            return self.matrix[0, 0]
        if self.num_rows == 2:
            return self.__determinant(self.matrix)
        total_sum = 0
        # looping on columns of the first row (minor values)
        for i in range(self.num_rows):
            # Calculating the minor's corresponding cofactor matrix
            m = self.__minor(self.matrix, 0, i)
            # summing the product of the minor and the "signed" determinant of the cofactor
            total_sum += ((-1) ** i) * self.matrix[0, i] * self.__determinant(m)
        return total_sum

    # TODO: Coding a function that generates the inverse of a 3x3 matrix if it exists

if __name__ == "__main__":
    equations = [[1.5, 5.75, 2.6], [1, 1, 1], [0, 1, -1]]
    answers = [589.5, 200, -20]
    linearAlgebra = LinearAlgebra(equations, answers, True)
    print(linearAlgebra.solution)
