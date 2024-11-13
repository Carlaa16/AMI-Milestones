module LinearSystems

using LinearAlgebra  # Necesario para la función dot

function Gauss(A, b)
    """
    Solutions to a system of linear equations A * x = b
    Method: Gauss elimination (with scaling and pivoting)
    
    Inputs:
        A: system matrix,
        b: independent term
    
    return:
        x: solution vector
    note: matrix A and b term are modified
    
    Juan A. Hernandez, juanantonio.hernandez@upm.es (Oct 2022)
    """
    
    N = length(b)
    x = zeros(N)
    
    # begin forward elimination
    for k in 1:N
        pivoting_row_swapping!(A, b, k, N)  # Llamada a la función de pivoteo
        
        # the elimination (after swapping)
        # for all rows below pivot:
        for i in k+1:N
            c = A[i,k] / A[k,k]
            A[i,k:N] .= A[i,k:N] - c * A[k,k:N]  # Element-wise operation
            b[i] -= c * b[k]
        end
    end
    
    # back substitution
    for i in N:-1:1
        x[i] = (b[i] - dot(A[i,i+1:N], x[i+1:N])) / A[i,i]
    end
    
    return x
end

function pivoting_row_swapping!(A, b, k, N)
    """
    Performs row swapping to ensure numerical stability by choosing the largest pivot element.
    
    Inputs:
        A: system matrix,
        b: independent term,
        k: current pivot step,
        N: matrix size
    """
    
    s = zeros(N)
    
    # s[i] is the largest element of row i
    for i in k:N  # Loop over rows
        s[i] = maximum(abs.(A[i,k:N]))  # Find maximum element in the row
    end
    
    # find a row with the largest pivoting element
    pivot = abs(A[k,k]) / s[k]
    l = k
    for j in k+1:N
        if abs(A[j,k] / s[j]) > pivot
            pivot = abs(A[j,k] / s[j])
            l = j
        end
    end
    
    # Check if the system has a singular matrix
    if pivot == 0.0
        println("The matrix is singular")
        exit()  # Exits the program
    end
    
    # pivoting: swap rows k and l (if needed)
    if l != k
        A[k,:], A[l,:] = A[l,:], A[k,:]  # Swap rows in matrix A
        b[k], b[l] = b[l], b[k]          # Swap corresponding elements in b
    end
end

# Main testing block
if abspath(PROGRAM_FILE) == @__FILE__
    # Testing examples
    A = [1.0 1.0 1.0; 1.0 -1.0 -1.0; 1.0 2.0 3.0]
    b = [1.5, -1.0, 0.5]
    result = Gauss(A, b)
    println("Result 1: ", result)

    A2 = [1.0 2.0 3.0; 1.0 -2.0 -3.0; 2.0 3.0 5.0]
    b2 = [1.5, -0.5, 1/6]
    result2 = Gauss(A2, b2)
    println("Result 2: ", result2)
end

end  # module LinearSystems
