from typing import Union, Tuple, Iterable, Sequence, Optional

from numpy import ndarray, asarray, float64, ones, diag, zeros, eye

from pyqumo.errors import MatrixShapeError, CellValueError, \
    RowSumError, RowsSumsError


# ############################################################################
# MATRIX-STRING CONVERTING ROUTINES
# ############################################################################

def row2string(row: Iterable[float], sep: str = ', ') -> str:
    """
    Converts a one-dimensional iterable of floats to string.

    Parameters
    ----------
    row: list or tuple or 1-D ndarray
    sep: str
        string separator between elements (default: ', ')

    Returns
    -------
    string representation of a row
    """
    return sep.join("{0}".format(item) for item in row)


def matrix2string(
        matrix: Iterable[Iterable[float]],
        col_sep: str = ', ',
        row_sep: str = '; ') -> str:
    """
    Converts a two-dimensional iterable of iterables of floats to string.

    Parameters
    ----------
    matrix: Iterable[Iterable[float]]
    col_sep: str, optional
        string separator between columns (default: ', ')
    row_sep: str, optional
        string separator between rows (default: '; ')

    Returns
    -------
    string representation of a matrix
    """
    return row_sep.join("{0}".format(
        row2string(row, col_sep)) for row in matrix)


def array2string(
        array: Union[Iterable[float], Iterable[Iterable[float]]],
        col_sep: str = ', ',
        row_sep: str = '; ') -> str:
    """
    Converts a 1- or 2-dimensional list, tuple or numpy.ndarray to string

    Parameters
    ----------
    array: Iterable[float] or Iterable[Iterable[float]]
        a 1d vector or 2d matrix
    col_sep: str, optional
        string separator between columns (default: ', ')
    row_sep: str, optional
        string separator between rows (default: '; ')

    Returns
    -------
    string representation of a matrix
    """
    try:
        return matrix2string(array, col_sep, row_sep)
    except TypeError:
        return row2string(array, col_sep)


def str_array(
        array: Union[ndarray, Sequence[float], Sequence[Sequence[float]]]
) -> str:
    """
    Returns array representation as list of lists using .3g cell formatter.
    In contrast to array2string and matrix2string, this routine keeps
    square brackets and behaves much like str(list(...)).

    Examples
    --------
    >>> str_array(asarray([[1, 2.5], [0.125, 2]]))
    >>> [[1, 2.5], [0.125, 2]]

    Parameters
    ----------
    array : ndarray
        A matrix to print

    Returns
    -------
    string : str
    """
    if not isinstance(array, ndarray):
        array = asarray(array)
    num_axis = len(array.shape)
    if num_axis == 1:
        return "[" + ", ".join([f"{x:.3g}" for x in array]) + "]"
    return "[" + ", ".join(
        [str_array(array[i]) for i in range(array.shape[0])]
    ) + "]"


def parse_array(
        s: str, col_sep: str = ',', row_sep: str = ';',
        dtype=float64) -> ndarray:
    """
    Parse a string into a NumPy array, 1D or 2D.

    If the string contains a single row (it contains no `row_sep` chars),
    then the result will be a 1D vector. Otherwise, the result will be a
    2D matrix.

    Parameters
    ----------
    s: str
    col_sep: str (optional, default ',')
    row_sep: str (optional, default ';')
    dtype: type (optional, default float64)

    Returns
    -------
    NumPy ndarray
    """
    rows = [x.strip() for x in s.split(row_sep)]
    if len(rows) == 1:
        columns = [x.strip() for x in s.split(col_sep)]
        if len(columns) == 1 and len(columns[0]) == 0:
            return asarray([], dtype=dtype)
        else:
            return asarray(list(dtype(item) for item in s.split(col_sep)),
                           dtype=dtype)
    else:
        return asarray(list(list(dtype(item) for item in row.split(col_sep))
                            for row in rows), dtype=dtype)


# ############################################################################
# MATRIX SHAPE STUDY
# ############################################################################
def is_square(matrix: ndarray) -> bool:
    """
    Checks that a given matrix has square shape.
    """
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def is_vector(vector: ndarray) -> bool:
    """
    Checks that a given matrix represents a vector.
    """
    return len(vector.shape) == 1 or (
            len(vector.shape) == 2 and
            (vector.shape[0] == 1 or vector.shape[1] == 1))


def order_of(matrix: ndarray) -> int:
    """
    Get the order of square matrix or a vector.

    Returns an order of a square matrix or a vector - a number of rows
    (equal to columns) for square matrix and number of elements for a vector.

    Parameters
    ----------
    matrix: ndarray
        a square matrix or a vector

    Returns
    -------
    If matrix is 2d, number of rows, if 1d - number of elements.

    Raises
    ------
    ValueError
        if matrix is 2d and is not a square matrix
    """
    if is_square(matrix):
        return matrix.shape[0]
    elif is_vector(matrix):
        return len(matrix.flatten())
    else:
        raise MatrixShapeError(
            expected="(N,) or (N, N)",
            shape=matrix.shape)


# ############################################################################
# SPECIAL MATRIX INSPECTION ROUTINES
# check matrix has a special property, e.g. it is stochastic or infinitesimal
# ############################################################################

def is_stochastic(mat: ndarray) -> bool:
    """
    Check all elements are in [0, 1] interval and row sums are all ones.

    Note, that this routine doesn't require the matrix to be square,
    so it can be used for checking vectors as well.

    Parameters
    ----------
    mat : ndarray

    Returns
    -------
    flag : bool
        True iff all elements are between 0 and 1, or close to them,
        and each row sum is 1.
    """
    if (mat.size == 0) or (mat < 0).any() or (mat > 1).any():
        return False
    if is_vector(mat):
        return mat.sum() == 1.0
    return (mat.sum(axis=1) == ones(mat.shape[0])).all()


def is_substochastic(mat: ndarray) -> bool:
    """
    Check all elements are in [0, 1] interval, row sums are less or equal one.

    Sums of elements of each row should be less or equal to one, and at
    least one row sum should be strictly less than one.

    Note, that this routine doesn't require the matrix to be square,
    so it can be used for checking vectors as well.

    Parameters
    ----------
    mat : ndarray

    Returns
    -------
    flag : bool
        True iff all elements are between 0 and 1, or close to them,
        and each row sum is 1.
    """
    if (mat.size == 0) or (mat < 0).any() or (mat > 1).any():
        return False
    if is_vector(mat):
        return mat.sum() < 1.0
    order = mat.shape[0]
    delta = ones((order, 1)) - mat.sum(axis=1)
    return (delta >= 0).all() and (delta > 0).any()


def is_pmf(mat: ndarray) -> bool:
    """
    Check that 1D array is a probability mass function - a stochastic row.

    Parameters
    ----------
    mat : ndarray
        1D array with sum of elements equal to 1.0

    Returns
    -------
    flag : bool
        True iff the argument is a 1D array, sum of its elements is 1.0 and
        all elements are non-negative.
    """
    return len(mat.shape) == 1 and is_stochastic(mat)


def is_infinitesimal(mat: ndarray) -> bool:
    """
    Check whether the matrix is infinitesimal.

    Matrix is infinitesimal, if it is a square matrix, and:

    1) each row sum is zero
    2) each non-diagonal element is greater or equal to zero
    3) each diagonal element is smaller or equal to zero (implied from 1, 2)

    Parameters
    ----------
    mat : ndarray
        a square matrix

    Returns
    -------
    flag : bool
        True iff the matrix is stochastic.
    """
    return (is_square(mat) and
            ((mat - diag(mat.diagonal().flatten())) >= 0).all() and
            (mat.sum(axis=1) == zeros(order_of(mat))).all())


def is_subinfinitesimal(mat: ndarray) -> bool:
    """
    Check whether the matrix is subinfinitesimal.

    Matrix is subinfinitesimal, if it is a square matrix, and:

    1) each row sum is smaller or equal to zero
    2) each non-diagonal element is greater or equal to zero
    3) at least one row sum is strictly smaller then zero
    3) each diagonal element is smaller or equal to zero (implied from 1, 2)

    Parameters
    ----------
    mat : ndarray

    Returns
    -------
    flag : bool
    """
    if not is_square(mat):
        return False
    row_sum = mat.sum(axis=1)
    return (((mat - diag(mat.diagonal().flatten())) >= 0).all() and
            (row_sum <= 0).all() and (row_sum < 0).any())


def check_markovian_arrival(matrices: Sequence[ndarray]) -> bool:
    """
    Check matrices for Markovian arrival process and return errors.

    Matrices can be used for Markovian arrival representation, if:

    1) two or more matrices are given
    2) all of them are square of the same order
    3) the first matrix is subinfinitesimal
    4) all matrices except the first one do not contain negative elements
    5) sum of all matrices is an infinitesimal matrix
    
    Parameters
    ----------
    matrices : sequence of ndarray

    Returns
    -------
    flag : bool
        True iff matrices can be used for Markovian arrival process.
    """
    if len(matrices) < 2 or not all(is_square(mat) for mat in matrices):
        return False
    order = order_of(matrices[0])
    return (all(order_of(mat) == order for mat in matrices[1:]) and
            is_subinfinitesimal(matrices[0]) and
            all((mat >= 0).all() for mat in matrices[1:]) and
            is_infinitesimal(sum(matrices)))


# ############################################################################
# SPECIAL MATRIX FIXES:
# these routines try to fix the matrix to make it sufficient for
# a given task, e.g. make stochastic or infinitesimal.
# ############################################################################

def fix_stochastic(mat: ndarray, tol: float = 1e-3) -> Tuple[ndarray, float]:
    """
    Try to fix elements of a matrix that is not fully stochastic.

    This function performs the following two steps:

    1) If any row `i` contains negative elements, it finds the column with the
    smallest element `j0`. If `M[i][j0] < -max_err`, then an error is raised.
    Otherwise, value `-M[i][j0]` is added to each i-th row cell.

    2) Let `Si` be the sum of i-th row elements. If `0 < |Si - 1| <= max_err`,
    then all row elements are divided on Si: `M[i][j] := M[i][j] / Si`

    If at any step the error is too large, function raises a corresponding
    exception.

    Parameters
    ----------
    mat: ndarray
        a 2d matrix, not necessary square
    tol: float64, optional
        maximum deviation of any matrix element from (0, 1) interval,
        as well as of the row sum from 1.

    Returns
    -------
    matrix: ndarray
        matrix of the same dimension, but stochastic.
    err: float
        maximum error

    Raises
    ------
    CellValueError
        raise this exception if some element is too smaller then zero,
        or too greater than one.
    RowSumError
        raise this exception if some row differs too much from one.
    """
    is_1d_vector = len(mat.shape) == 1

    # To make processing simpler, we convert 1D vector to (1,N) matrix,
    # and before returning result, convert them back:
    if len(mat.shape) == 1:
        mat = mat.reshape((1, mat.shape[0]))

    # From this point, we are sure that matrix has two dimensions.
    num_rows, num_cols = mat.shape
    new_mat: ndarray = mat.copy()

    # Find the smallest elements of each row and record errors.
    min_cols = new_mat.argmin(axis=1)

    # Store maximum cell error: row, col, value, error
    cell_err: Tuple[int, int, float, float] = (-1, -1, 0, 0)

    for row, col in enumerate(min_cols):
        # Extract the smallest element of the row. If it is less then 0,
        # track error and add its absolute value to all row elements:
        el = new_mat[(row, col)]
        if (error := -el) > 0:
            if error > cell_err[-1]:
                cell_err = (row, col, el, error)
            new_mat[row] -= el * ones(num_cols)

    # If the smallest element is smaller than -max_err, raise an exception:
    if cell_err[-1] > tol:
        row, col, el, error = cell_err
        raise CellValueError(
            row, col, el, error=error, tol=tol, lower=0.0, upper=1.0)

    # Now all new_mat elements should be non-negative:
    if (new_mat < -1e-5).any():
        print(new_mat)
        assert False
    # assert (new_mat >= -1e-5).all()

    # Iterate over all rows and find deviations from 1:
    row_sums = new_mat.sum(axis=1)
    row_errors: ndarray = ones(num_rows) - row_sums
    abs_row_errors: ndarray = abs(row_errors)
    row_with_max_err: int = abs_row_errors.argmax()
    row_error = abs_row_errors[row_with_max_err]

    # If the maximum deviation is larger then max_err, raise an error:
    if row_error > tol:
        row_sum = row_sums[row_with_max_err]
        raise RowSumError(
            row_with_max_err, row_sum, 1.0, error=row_error, tol=tol)

    # Otherwise, divide all elements on the row sum, if this sum is
    # not equal to one:
    new_mat = new_mat / row_sums.reshape((num_rows, 1))

    # If original matrix was 1D array, reshape it back:
    if is_1d_vector:
        new_mat = new_mat.reshape((new_mat.shape[1],))

    return new_mat, max(cell_err[-1], row_error)


def fix_infinitesimal(
        mat: ndarray,
        tol: float = 1e-9,
        sub: bool = False
) -> Tuple[ndarray, float]:
    """
    Try to fix an almost infinitesimal matrix.

    This function fixes two types of errors. Firstly, for each non-diagonal
    element between `-max_err` and 0, set it to 0.

    Second step depends on whether `sub = True` or not. Let Si be the sum of
    i-th row elements. If `sub = False`, if `|Si| < tol`, then Si is
    subtracted from the diagonal element M[i][i], so the sum is made 0.
    Otherwise, if `sub = True`, then Si is subtracted only if `0 < Si < tol`.

    If matrix shape is not square, then `MatrixShapeError` exception is thrown,
    since it is not possible to recover.

    Also, if `sub = True`, afterwards check that for at least one row sum
    of its elements is strictly negative. If not, raise NoNegRowSumError.

    Parameters
    ----------
    mat : ndarray
    tol : maximum absolute tolerance
        comparisons are made with 0, so no need to give relative tolerance
    sub : bool
        if set to True, the matrix is supposed to be subinfinitesimal, i.e.
        sum of its row elements is less or equal to zero for all rows, except
        at least one row, where it is strictly negative.

    Returns
    -------
    matrix: ndarray
        matrix of the same dimension, but infinitesimal.
    err: float64
        maximum error in elements

    Raises
    ------
    CellValueError
        raise this exception if some element is too smaller then zero,
        or too greater than one.
    RowSumError
        raise this exception if some row differs too much from one.
    MatrixShapeError
        raise this exception if matrix shape is not a square
    ZeroRowSumsError
        raise this exception if `sub = True` and all rows sum are zero.
    """
    # Check matrix is square:
    if not is_square(mat):
        raise MatrixShapeError('(N, N)', mat.shape)

    order = mat.shape[0]
    new_mat = mat.copy()
    cell_err: Tuple[int, int, float, float] = (-1, -1, 0.0, 0.0)
    row_err: Tuple[int, float, float] = (-1, 0.0, 0.0)

    for row in range(order):
        # 1) Fix non-diagonal elements
        for col in range(order):
            if row == col:
                continue
            el = new_mat[(row, col)]
            if el < 0:
                if (err := -el) > cell_err[-1]:
                    cell_err = row, col, el, err
                new_mat[(row, col)] = 0.0
        # 2) Fix diagonal element
        delta = new_mat[row].sum()
        error = abs(delta) if not sub else (delta if delta > 0 else 0)
        if error > 0:
            if error > row_err[-1]:
                row_err = row, delta, error
            new_mat[(row, row)] -= delta

    # Check errors and raise exceptions if errors are larger then tolerance:
    if cell_err[-1] > tol:
        raise CellValueError(
            cell_err[0], cell_err[1], cell_err[2], error=cell_err[-1], tol=tol,
            lower=0, upper=None)
    if row_err[-1] > tol:
        raise RowSumError(
            row_err[0], row_err[1], 0.0, error=row_err[-1], tol=tol)

    # If matrix should be strictly subinfinitesimal, check that at least one
    # row sum is less then zero:
    row_sums: ndarray = new_mat.sum(axis=1)
    if sub and not (row_sums < 0).any():
        raise RowsSumsError(row_sums, 'all rows sums are zero')

    return new_mat, max(cell_err[-1], row_err[-1])


def fix_markovian_arrival(
        matrices: Sequence[ndarray],
        tol: float = 1e-9
) -> Tuple[Tuple[ndarray, ...], float]:
    """
    Attempt to fix matrices D0, D1, ..., Dn so they can represent xMAP.

    In the simplest case, a sequence contain two matrices D0 and D1.
    If there are more matrices, they can be used for batch Markovian arrival
    or MMAP process.

    The routine fix two types of errors:

    1) negative non-diagonal D0 elements, or negative elements in Di, i > 0
    2) sum of rows over all matrices not equal to zero

    If each error is less, then `tol` value, then it is fixed. Negative
    cells are made equal to zero. If sum of row elements is not zero, then
    the error is subtracted from the corresponding D0 diagonal element.

    If some non-diagonal D0 element or any Di element is smaller then `-tol`,
    CellValueError is raised. If absolute value of sum of a row is greater
    then `tol`, then `RowSumError` is raised.

    All matrices must be square and have the same order. Otherwise,
    MatrixShapeError is raised.

    Parameters
    ----------
    matrices : sequence of ndarray
        D0, D1, ..., Dn matrices
    tol : float
        absolute tolerance

    Returns
    -------
    matrices : sequence of ndarray
        fixed D0, D1, ..., Dn matrices
    error : float
        maximum fixed error

    Raises
    ------
    ValueError
        raise this if less then two matrices provided
    CellValueError
        raise this exception if some element is too smaller then zero,
        or too greater than one.
    RowSumError
        raise this exception if some row differs too much from one.
    MatrixShapeError
        raise this exception if matrix shape is not a square
    """
    # Validate then at least two matrices provided:
    if len(matrices) < 2:
        raise ValueError("require at least two matrices")

    # Validate matrices have the same square shape:
    if not is_square(matrices[0]):
        raise MatrixShapeError("(N, N)", matrices[0].shape, "D0")
    order = order_of(matrices[0])
    for mi, mat in enumerate(matrices[1:]):
        if mat.shape != (order, order):
            raise MatrixShapeError((order, order), mat.shape, f"D{mi + 1}")

    # Store cell errors: matrix index, row, col, value, error
    cell_err: Tuple[int, int, int, float, float] = (-1, -1, -1, 0.0, 0.0)

    # Store row sum errors: row, value, error
    row_err: Tuple[int, float, float] = (-1, 0.0, 0.0)

    new_matrices = tuple(mat.copy() for mat in matrices)

    for row in range(order):
        # Check and fix elements:
        for mi, mat in enumerate(new_matrices):
            for col in range(order):
                if row == col and mi == 0:
                    continue
                el = mat[(row, col)]
                if el < 0:
                    if (err := -el) > cell_err[-1]:
                        cell_err = mi, row, col, el, err
                    mat[(row, col)] = 0.0
        # Check and fix row errors:
        row_sum = sum(mat[row].sum() for mat in new_matrices)
        if (error := abs(row_sum)) > 0:
            if error > row_err[-1]:
                row_err = row, row_sum, error
            new_matrices[0][(row, row)] -= row_sum

    # Check error values and raise exceptions if needed:
    if cell_err[-1] > tol:
        raise CellValueError(
            cell_err[1], cell_err[2], cell_err[3], error=cell_err[-1], tol=tol,
            lower=0, upper=None, label=f"D{cell_err[0]}")
    if row_err[-1] > tol:
        raise RowSumError(
            row_err[0], row_err[1], 0.0, error=row_err[-1], tol=tol)

    return tuple(new_matrices), max([cell_err[-1], row_err[-1]])


# ############################################################################
# OPERATIONS FOR BLOCK MATRICES
# ############################################################################
def _validate_cb_blocks(blocks: Iterable[Sequence]) -> None:
    """
    Validate blocks for blocks sequence for cbXXX() routines.

    It is assumed that blocks is an iterable of sequences, and last element
     of each sequence is an `ndarray` instance. Then we validate:

    1) `blocks` sequence is not empty
    2) `blocks[0][-1]` is a 2D `ndarray`
    3) all `blocks[i][-1].shape` are the same.

    If first check failed, `ValueError` is thrown. If cases 2 or 3 are
    violated, then `MatrixShapeError` is thrown.

    Parameters
    ----------
    blocks : iterable of sequences (tuples), last element of each is ndarray

    Raises
    ------
    ValueError
        thrown if `blocks` iterable is empty
    MatrixShapeError
        thrown if the first block is not a 2D array, or if any block shape
        differs from the first block shape.
    """
    if not blocks:
        raise ValueError("need at least one block")
    block_shape = blocks[0][-1].shape
    if len(block_shape) != 2:
        raise MatrixShapeError('(N, M)', block_shape, f"B[0]")
    for bi, block in enumerate(blocks[1:]):
        if (mat := block[-1]).shape != block_shape:
            raise MatrixShapeError(block_shape, mat.shape, f"B[{bi + 1}]")


def cbmat(blocks: Iterable[Tuple[int, int, ndarray]]) -> ndarray:
    """
    For a given sequence of blocks [(i, j, Bij)...] build a block matrix.

    Each block is given with its coordinates in the block matrix, and
    the block value. All blocks must have the same shape.

    Block shape is studied from the first block. Number of columns and rows
    in the block matrix are studied from indicies: maximum block row index
    is the number of rows, maximum block col index is the number of columns.

    Examples
    --------
    >>> cbmat([
    >>>     (0, 0, asarray([[1, 2]])),
    >>>     (1, 0, asarray([[3, 4]])),
    >>>     (1, 2, asarray([[5, 6]])),
    >>> ]).tolist()
    >>> # Output:
    >>> [[1, 2, 0, 0, 0, 0], [3, 4, 0, 0, 5, 6]]

    Parameters
    ----------
    blocks : iterable of tuples of int, int and ndarray
        blocks given as `(row, col, block)`

    Returns
    -------
    matrix : ndarray
        a block matrix built from the given blocks

    Raises
    ------
    MatrixShapeError
        raised when any block Bi have shape different from the first block B0,
        or if block is not a 2D matrix.
    ValueError
        raised if an empty sequence of blocks was given
    """
    _validate_cb_blocks(blocks)
    block_shape = blocks[0][-1].shape
    # Check parameters and learn shape:
    max_row = max(row for row, col, block in blocks)
    max_col = max(col for row, col, block in blocks)
    num_rows = (max_row + 1) * block_shape[0]
    num_cols = (max_col + 1) * block_shape[1]
    matrix = zeros((num_rows, num_cols))
    for row, col, block in blocks:
        row_0 = row * block_shape[0]
        row_1 = (row + 1) * block_shape[0]
        col_0 = col * block_shape[1]
        col_1 = (col + 1) * block_shape[1]
        matrix[row_0:row_1, col_0:col_1] = block
    return matrix


def cbdiag(size: int, blocks: Iterable[Tuple[int, ndarray]]) -> ndarray:
    """
    Build a block matrix with (sub-)diagonal blocks and the given size.

    Each block is specified with its offset from the diagonal and its data
    (sub matrix). All blocks are expected to have the same shape. If not,
    a `MatrixShapeError` exception is raised.

    Offset of each block can be positive, zero or negative. Zero offset means
    that the block will be used in the diagonal. Non-zero offsets are counted
    from left to right. For example, offset 1 means a subdiagonal to the
    right of the main diagonal, i.e. positions (0, 1), (1, 2), etc. Negative
    offsets specify subdiagonals to the left of the main diagonal.

    Matrix size is computed from the given block size and block shape. If
    size is equal to N, and block shape is (M, K), then the matrix will have
    the shape (NM, NK).

    Examples
    --------
    >>> # First example: 5-diagonal matrix:
    >>> cbdiag (6, [
    >>>     (0, asarray([[1]])), (1, asarray([[2]])), (2, asarray([[3]])),
    >>>     (-1, asarray([[4]])), (-2, asarray([[5]]))
    >>> ]).tolist()
    >>> [[1, 2, 3, 0, 0, 0],
    >>>  [4, 1, 2, 3, 0, 0],
    >>>  [5, 4, 1, 2, 3, 0],
    >>>  [0, 5, 4, 1, 2, 3],
    >>>  [0, 0, 5, 4, 1, 2],
    >>>  [0, 0, 0, 5, 4, 1]]

    >>> # Second example: 3-diagonal matrix with 1x2 blocks:
    >>> cbdiag(4, [
    >>>     (0, asarray([[1, 1]])),
    >>>     (1, asarray([[2, 2]])),
    >>>     (-1, asarray([[3, 3]])),
    >>> ]).tolist()
    >>> [[1, 1, 2, 2, 0, 0, 0, 0],
    >>>  [3, 3, 1, 1, 2, 2, 0, 0],
    >>>  [0, 0, 3, 3, 1, 1, 2, 2],
    >>>  [0, 0, 0, 0, 3, 3, 1, 1]]

    Parameters
    ----------
    size : int
        number of block-rows and block-cols
    blocks : sequence of tuples of int and ndarray
        blocks, each block is specified with integer offset and the matrix

    Returns
    -------
    mat : ndarray
       a block matrix built from the given blocks.

    Raises
    ------
    MatrixShapeError
        raised if blocks have different shapes
    """
    _validate_cb_blocks(blocks)
    block_shape = blocks[0][-1].shape
    mat = zeros((size * block_shape[0], size * block_shape[1]))
    for block_row in range(size):
        mat_row_0 = block_row * block_shape[0]
        mat_row_1 = mat_row_0 + block_shape[0]
        for offset, block in blocks:
            block_col = offset + block_row
            if 0 <= block_col < size:
                mat_col_0 = block_col * block_shape[1]
                mat_col_1 = mat_col_0 + block_shape[1]
                mat[mat_row_0:mat_row_1, mat_col_0:mat_col_1] = block
    return mat


# ############################################################################
# SPECIAL MATRICES
# ############################################################################
def identity(length: int, k: int, axis: Optional[int] = None) -> ndarray:
    """
    Get a vector of given length L with one 1 at position K.

    Parameters
    ----------
    length : int
        order of the row vector
    k : int
        position of 1
    axis : int or None, optional
        Direction. If given, a 2D row or column matrix will be returned:
        0 - column, 1 - row. If not provided (None, default), a 1D vector
        will be returned.

    Returns
    -------
    vector : np.ndarray
    """
    if axis is None:
        ret = zeros(length)
        ret[k] = 1
        return ret
    if axis == 0:
        return eye(length, 1, -k)
    if axis == 1:
        return eye(1, length, k)
    raise ValueError(f'axis should be None, 0 or 1, but {axis} found')




# def pmf2pdf(pmf):
#     """Converts a PMF matrix (each row is expected to be a PMF) into a matrix
#     of the same size representing PDF. Please note, that no is_pmf() check
#     is called. If needed, it should be called directly before this call.
#
#     Args:
#         pmf: a vector or a matrix where each row represents probability mass
#             function
#
#     Returns:
#         a vector or matrix of the same size as the provided pmf, where
#         each row corresponds to cumulative probability distribution
#     """
#     pmf = np.asarray(pmf)
#     if is_vector(pmf):
#         vector = pmf.toarray().flatten() if isinstance(
#             pmf, sparse.spmatrix) else pmf.flatten()
#         pdf = np.zeros(vector.size)
#         pdf[0] = vector[0]
#         for i in range(1, vector.size):
#             pdf[i] = pdf[i - 1] + vector[i]
#         return pdf
#     else:
#         rows_num, cols_num = pmf.shape
#         if isinstance(pmf, sparse.spmatrix):
#             matrix = pmf.tocsc().T
#             pdf = matrix[0]
#             for i in range(1, cols_num):
#                 pdf = sparse.vstack((pdf, pdf[i - 1] + matrix[i]))
#         else:
#             matrix = pmf.T
#             pdf = matrix[0].reshape((1, rows_num))
#             for i in range(1, cols_num):
#                 pdf = np.vstack((pdf, pdf[i - 1] + matrix[i]))
#         return pdf.T
#
#
# def pdf2pmf(pdf):
#     """Converts a PDF matrix (each row is expected to be a PDF) into a matrix
#     of the same size representing PMF. Please note, that no check ispdf()
#     is called. If needed, it should be called directly before this call.
#
#     Args:
#         pdf: a vector or a matrix where each row represents cumulative
#             probability distribution
#
#     Return:
#         a vector or matrix of the same size as the provided pdf, where
#         each row corresponds to probability mass function
#     """
#     pdf = np.asarray(pdf)
#     if is_vector(pdf):
#         pdf = pdf.toarray().flatten() if isinstance(
#             pdf, sparse.spmatrix) else pdf.flatten()
#         pmf = np.zeros(pdf.size)
#         pmf[0] = pdf[0]
#         for i in range(pmf.size - 1, 0, -1):
#             pmf[i] = pdf[i] - pdf[i - 1]
#         return pmf
#     else:
#         rows_num, cols_num = pdf.shape
#         if isinstance(pdf, sparse.spmatrix):
#             matrix = pdf.tocsc().T
#             pmf = matrix[0]
#             for i in range(1, cols_num):
#                 pmf = sparse.vstack((pmf, matrix[i] - matrix[i - 1]))
#         else:
#             matrix = pdf.T
#             pmf = matrix[0].reshape((1, rows_num))
#             for i in range(1, cols_num):
#                 pmf = np.vstack((pmf, matrix[i] - matrix[i - 1]))
#         return pmf.T
