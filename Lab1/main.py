import pathlib
import re


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    with open(path, 'r') as file:
        lines = file.readlines()

    coeff_matrix = []
    result_values = []

    for line in lines:
        line = line.replace(" ", "")
        matches = re.findall(r'([+-]?\d*)([xyz])', line)
        constant = int(re.search(r'=(.*)', line).group(1))
        coefficients = {'x': 0, 'y': 0, 'z': 0}

        for match in matches:
            coeff_str, variable = match
            if coeff_str == '' or coeff_str == '+':
                coeff = 1
            elif coeff_str == '-':
                coeff = -1
            else:
                coeff = int(coeff_str)

            coefficients[variable] = coeff

        coeff_matrix.append([coefficients['x'], coefficients['y'], coefficients['z']])
        result_values.append(constant)

    return coeff_matrix, result_values


A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=}, {B=}")


def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))


print(f"{determinant(A)=}")


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


print(f"{trace(A)=}")


def norm(vector: list[float]) -> float:
    return sum(x ** 2 for x in vector) ** 0.5


print(f"{norm(B)=}")


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[j][i] for j in range(3)] for i in range(3)]


print(f"{transpose(A)=}")


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []
    for row in matrix:
        result.append(sum(row[i] * vector[i] for i in range(3)))
    return result


print(f"{multiply(A, B)=}")


def replace_column_cramer(matrix: list[list[float]], vector: list[float], col: int) -> list[list[float]]:
    modified_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        modified_matrix[i][col] = vector[i]
    return modified_matrix


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    if det_A == 0:
        raise ValueError("Singular matrix")

    Ax = replace_column_cramer(matrix, vector, 0)
    Ay = replace_column_cramer(matrix, vector, 1)
    Az = replace_column_cramer(matrix, vector, 2)

    x = determinant(Ax) / det_A
    y = determinant(Ay) / det_A
    z = determinant(Az) / det_A

    return [x, y, z]


print(f"{solve_cramer(A, B)=}")


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [[matrix[n][m] for m in range(3) if m != j] for n in range(3) if n != i]


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    return [[((-1) ** (i + j)) * determinant(minor(matrix, i, j)) for j in range(3)] for i in range(3)]


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    if det_A == 0:
        raise ValueError("Matrix is singular, no unique solution")

    adj_A = adjoint(matrix)
    inv_A = [[adj_A[i][j] / det_A for j in range(3)] for i in range(3)]

    return multiply(inv_A, vector)


print(f"{solve(A, B)=}")
