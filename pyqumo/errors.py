from typing import Optional, Tuple, Union

from numpy import inf, ndarray


class MatrixError(Exception):
    pass


class MatrixShapeError(MatrixError):
    """
    An error is thrown if a matrix shape is not what expected.

    Attributes
    ----------
    expected : str or tuple of ints
        expected matrix shape, either given as a tuple, or a descriptive
        string, e.g. "square", "NxM, M > 1" or "(N, N)"
    shape : tuple of ints
        actual matrix shape
    label : str, optional
        typically, name of the matrix
    """
    def __init__(self, expected: Union[Tuple[int, int], str],
                 shape: Tuple[int, int], label: str = ""):
        self.expected = expected
        self.shape = shape
        self.label = label

        prefix = f"{label} " if label else ""
        self.message = f"{prefix}matrix shape error: expected {expected}, " \
                       f"but {shape} found"
        super().__init__(self.message)


class CellValueError(MatrixError):
    """
    An error indicating that a matrix element is out of bounds.

    Attributes
    ----------
    row : int
    col : int
    value : float
    error : float
        error of the value that caused the exception. No distinction made
        for relative and absolute errors, since all comparisons are made
        with zeros or ones.
    tol : float
        tolerance that was used when comparing values. No distinction made
        for relative and absolute tolerance, since all comparisons are made
        with zeros or ones.
    lower : float
    upper : float
    label : str, optional
    """
    def __init__(self, row: int, col: int, value: float, error: float,
                 tol: float, lower: Optional[float], upper: Optional[float],
                 label: str = ""):
        # Store attributes:
        self.row = row
        self.col = col
        self.value = value
        self.error = error
        self.tol = tol
        self.lower = lower if lower is not None else inf
        self.upper = upper if upper is not None else inf
        self.label = label

        # Build a message:
        label_string = f"{label} " if label else ""
        self.message = \
            f"{label_string}cell ({row}, {col}) value {value} is out of " \
            f"bounds {(self.lower, self.upper)}\n" \
            f"\terror: {error}, tolerance: {tol}"

        # Call parent constructor:
        super().__init__(self.message)


class RowSumError(MatrixError):
    """
    An error indicating that sum of row elements if invalid.

    Attributes
    ----------
    row : int
        a row at which error was discovered
    actual : float
        actual row sum
    desired : float
        expected row sum
    error : float
        error between actual and desired values. No distinction made for
        relative and absolute errors, since all comparisons are made with
        zeros or ones.
    tol : float
        tolerance that was used when comparing values. No distinction made
        for relative and absolute tolerance, since all comparisons are made
        with zeros or ones.
    """
    def __init__(self, row: int, actual: float, desired: float, error: float,
                 tol: float):
        # Store attributes:
        self.row = row
        self.actual = actual
        self.desired = desired
        self.error = error
        self.tol = tol

        # Build a message:
        self.message = \
            f"sum of row {row} elements is {actual}, expected {desired}\n" \
            f"\terror: {error}, tolerance: {tol}"

        # Call parent constructor:
        super().__init__(self.message)


class RowsSumsError(MatrixError):
    """
    An error indicate that all rows sums are zero, while at least one negative
    or positive row expected.

    Attributes
    ----------
    row_sums : ndarray
    """
    def __init__(self, row_sums: ndarray, message: str):
        self.row_sums = row_sums
        super().__init__(f"{message}: {row_sums}")
