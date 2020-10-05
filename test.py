import numpy as np
import pandas as pd
from qubovert import boolean_var
from qubovert.sim import anneal_qubo
import matplotlib.pyplot as plt 

def plot_mat(matrix, values=True):
    n = matrix.shape[0]
    fig, ax = plt.subplots()
    

    if values:
        ax.matshow(matrix[tuple([n-i for i in range(1, n+1)]),:], cmap=plt.cm.Blues)
        min_val, max_val = 0, n
        for i in range(n):
            for j in range(n):
                ax.text(i, j, str(matrix[n-j-1,i]), va='center', ha='center')
        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(-0.5, n-0.5)
        #ax.set_xticks(np.arange(max_val))
        #ax.set_yticks(np.arange(max_val))
    else:
        ax.matshow(matrix, cmap=plt.cm.Blues)

        plt.show()

# Define sudoku size (must be a square number).
N = 9
num_bits = N**3
sq_N = np.int64(np.sqrt(N))
z = np.array([i+1 for i in range(N)], dtype=np.int64)

# Set initial values.
# vals_df = pd.DataFrame({'i' : idx_init // N, 'j' : idx_init % N, 'value' : val_init})
idx_init = np.random.choice(np.arange(N*N), N, replace=False)
a = np.array([[i for i in range(N)]+idx_init[j]*N for j in range(N)]).ravel()
x_init = {a[i] : int(i%N == i//N) for i in range(N*N)}

init_vec = np.zeros((num_bits, 1), dtype=np.int64)
idx = np.array([k for (k, v) in x_init.items() if v])
init_vec[idx] = 1

sudoku_init = (np.kron(np.identity(N**2, dtype=np.int64), z) @ init_vec).reshape((N, N))
plot_mat(sudoku_init)

# Define binary variables for QUBO.
x = {i : boolean_var(f'x{i}') for i in range(num_bits) if i not in x_init.keys()}
x.update(x_init)

# Define QUBO matrix.
penalty = 200
id_N = np.identity(N)
id_sqN = np.identity(sq_N)
A = np.concatenate((np.kron(np.kron(id_N, id_N), np.ones((1, N))), 
                    np.kron(np.kron(id_N, np.ones((1, N))), id_N),
                    np.kron(np.kron(np.ones((1, N)), id_N), id_N),
                    np.kron(np.kron(np.kron(id_sqN, 
                                            np.ones((1, sq_N))), 
                                    np.kron(id_sqN, 
                                            np.ones((1, sq_N)))), id_N)))
b = np.ones((4*N*N, 1))

Q = penalty * (A.T @ A - 2 * np.diag((b.T @ A)[0]))

# Define model.
model = penalty * (b.T @ b)[0,0]
for i in range(num_bits):
    for j in range(num_bits):
        model += Q[i,j]*x[i]*x[j]

# Run simulated annealing.
res = anneal_qubo(model, num_anneals=3)

# Check solution.
if res.best.value == 0:
    print("Solved!")
else:
    print("Invalid solution. QUBO value: ", res.best.value + penalty * (b.T @ b)[0,0])

# Convert solution into sudoku grid.
sol_vec = np.zeros((num_bits, 1), dtype=np.int64)
res.best.state.update({('x'+str(k)):v for (k,v) in x_init.items()})
idx = np.array([np.int64(k[1:]) for (k, v) in res.best.state.items() if v])
sol_vec[idx] = 1

sudoku_sol = (np.kron(np.identity(N**2, dtype=np.int64), z) @ sol_vec).reshape((N, N))

plot_mat(sudoku_sol)