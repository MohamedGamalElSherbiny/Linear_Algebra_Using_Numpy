from copy import deepcopy
import numpy as np

class SolveLinearEquation:
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

    See also:
    --------
    https://mathworld.wolfram.com/GaussianElimination.html
    https://en.wikipedia.org/wiki/Gaussian_elimination
    """

    def __init__(self, matrix, target_vector):
        self.matrix = np.array(matrix, dtype=np.float64)
        self.target_vector = np.array(target_vector).reshape(-1, 1)
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
        matrix_copy = deepcopy(matrix)
        if matrix[pivot_row, pivot_element] != 1:
            matrix_copy[pivot_row, :] /= matrix[pivot_row, pivot_element]
        elif matrix[pivot_row, pivot_element] == 0 and not checked:
            matrix[-1, :] = matrix[pivot_row, :]
            matrix[pivot_row, :] = matrix_copy[-1, :]
            SolveLinearEquation.__gaussian_reduction(matrix, pivot_row, target_row, pivot_element, checked=True)
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
            matrix = np.concatenate((self.matrix, self.target_vector), axis=1)
            for i in range(len(matrix) - 1):
                matrix = self.__gaussian_reduction(matrix, pivot_row=i, target_row=i + 1, pivot_element=i)
            for row in range(len(matrix)):
                matrix[row, :] /= matrix[row, row]
            return matrix
        else:
            "There is no solution to this matrix.\nOr this matrix has infinite number of solutions."