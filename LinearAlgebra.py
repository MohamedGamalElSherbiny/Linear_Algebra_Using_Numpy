import numpy as np


class LinearAlgebra:
    """
    Solves a system of linear equations using Gaussian Echelon Form

    Parameters:
    -----------
    matrix :         list of lists
                     each embedded list represents a row
                     list of integers / float numbers
    target_vector :  list
                     list of integer / float numbers with the final answer of the equations

    Returns:
    -------
    solution:        numpy
                     Matrix in the echelon form
    """

    def __init__(self, matrix, target_vector):
        self.matrix = np.array(matrix, dtype=np.float64)
        self.target_vector = np.array(target_vector).reshape(-1, 1)
        self.solution = self._solve_linear_equation()
        self.det = 0

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
        num_rows = len(self.matrix)
        is_square_bool = True
        for row in self.matrix:
            if num_rows != len(row):
                is_square_bool = False
        return is_square_bool

    def __matrix_properties(self, print_properties=True):
        """
        Checks if the matrix columns are independent.

        Parameters:
        -----------
        matrix :             numpy array
                             Matrix of integers / float numbers
        print_properties :   boolean, optional
                             To print the rank and the determinant of the matrix

        Returns:
        -------
        Boolean

        See also:
        --------
        Prints the matrix rank of the input matrix using np.linalg.matrix_rank
        Prints the determinant of the input matrix using np.linalg.det function
        """
        if self.__is_square():
            num_rows = len(self.matrix)
            matrix_rank = np.linalg.matrix_rank(self.matrix)
            if print_properties:
                print(f"The rank of the matrix is: {matrix_rank}")
                print(f"The determinant of the matrix is: {np.linalg.det(self.matrix)}")
            if matrix_rank == num_rows:
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
            "There is no solution to this matrix.\nOr this matrix has infinite number of solutions."

    def det(self):
        self.det = self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

    def __interchange_matrix(self):
        inverted_matrix = np.array([[self.matrix[1][1], -1 * self.matrix[0][1]],
                                    [-1 * self.matrix[1][0], self.matrix[0][0]]])
        return inverted_matrix

    def inverse_matrix(self):
        return (1 / self.det) * self.__interchange_matrix()

    # TODO: Coding a function that checks if a 3x3 matrix is invertible
    # TODO: Coding a function that generates the transpose of a 3x3 matrix
    # TODO: Coding a function that generates the matrix of minors of a 3x3 matrix
    # TODO: Coding a function that generates the matrix of cofactors of a 3x3 matrix
    # TODO: Coding a function that generates the inverse of a 3x3 matrix if it exists

    # def minor(arr, i, j):
    #     c = arr[:]
    #     c = np.delete(c, (i), axis=0)  # deleting the row containing the matrix minor from the original matrix copy
    #     matrix = [np.delete(row, (j), axis=0) for row in
    #               (c)]  # Looping on each row and deleting the column element below the minor
    #     print(matrix)
    #     return matrix
    #
    # def det(arr):
    #     n = len(arr)
    #     if n == 1: return arr[0][0]
    #     if n == 2: return arr[0][0] * arr[1][1] - arr[0][1] * arr[1][0]
    #     sum = 0
    #     for i in range(0, n):  # looping on columns of the first row (minor values)
    #         m = minor(arr, 0, i)  # Calculating the minor's corresponding cofactor matrix
    #         # print("minor of 0 and",i, "is:", m)
    #         sum = sum + ((-1) ** i) * arr[0][i] * det(
    #             m)  # summing the product of the minor and the "signed" determinant of the cofactor
    #     return sum

if __name__ == "__main__":
    equations = [[1.5, 5.75, 2.6], [1, 1, 1], [0, 1, -1]]
    answers = [589.5, 200, -20]
    solveLinearEquation = LinearAlgebra(equations, answers)
    print(solveLinearEquation.solution)
