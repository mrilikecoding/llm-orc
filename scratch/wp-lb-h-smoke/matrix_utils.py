def transpose(matrix):
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("All rows in matrix must have the same length")
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed

def multiply(matrix_a, matrix_b):
    if not matrix_a or not matrix_b:
        raise ValueError("Both matrices must be non-empty")
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError(f"Matrix A columns ({len(matrix_a[0])}) must match Matrix B rows ({len(matrix_b)})")
    rows = len(matrix_a)
    cols = len(matrix_b[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            dot_product = 0
            for k in range(len(matrix_b)):
                dot_product += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = dot_product
    return result