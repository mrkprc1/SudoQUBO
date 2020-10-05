# A QUBO formulation for sudoku

For each square on the sudoku we can assign 9 bits encoding the number assigned to each square. The value in each square can therefore be expressed as the inner product:

$$
    \begin{bmatrix}
        1 & 2 & \dots & 9
    \end{bmatrix}
    {\bf b}_{ij} 
$$

where ${\bf b}_{ij} \in \{ 0, 1\}^{\otimes9}$ is the column vector of bits for the sudoku square on row $i$ and column $j$. To be a valid sudoku square, the vector ${\bf b}_{ij}$ should contain only a single 1, this can be expressed as the constraint ${\bf 1}^{\rm T} {\bf b}_{ij} = 1$, where ${\bf 1}$ is the vector of ones. Defining ${\bf z} := [1, \dots, 9]^{\rm T}$ we can write a complete sudoku (unravelled as a vector) as

$$
    {\bf d} = ( {\mathbb I}_{82} \otimes {\bf z}^{\rm T} ) 
    \begin{bmatrix}
        {\bf b}_{11} \\
        {\bf b}_{12} \\
        \vdots \\
        {\bf b}_{99}
    \end{bmatrix}
$$

To find a solution where each grid square has a single digit selected from ${\bf z}$, we can enforce the constraint:

$$
    ({\mathbb I}_{81} \otimes {\bf 1}_9^{\rm T}){\bf b} = {\bf 1}_{81}
$$

The row constraints can be expressed as:

$$
    ({\mathbb I}_9 \otimes {\bf 1}_9^{\rm T} \otimes {\mathbb I}_9) {\bf b} = {\bf 1}_{81}
$$

and the column constraints:

$$
    ({\bf 1}_9^{\rm T} \otimes {\mathbb I}_9 \otimes  {\mathbb I}_9) {\bf b} = {\bf 1}_{81}
$$

The block constraints can be enforced with:

$$
\begin{aligned}
    [(\sum_{i=1}^3 {\bf e}_i^{\rm T}) \otimes (\sum_{i=1}^3 {\bf e}_i^{\rm T}) \otimes {\mathbb I}] {\bf b} = {\bf 1}_9 \\
    [(\sum_{i=4}^6 {\bf e}_i^{\rm T}) \otimes (\sum_{i=1}^3 {\bf e}_i^{\rm T}) \otimes {\mathbb I}] {\bf b} = {\bf 1}_9 \\
    [(\sum_{i=7}^9 {\bf e}_i^{\rm T}) \otimes (\sum_{i=1}^3 {\bf e}_i^{\rm T}) \otimes {\mathbb I}] {\bf b} = {\bf 1}_9 \\
    [(\sum_{i=1}^3 {\bf e}_i^{\rm T}) \otimes (\sum_{i=4}^6 {\bf e}_i^{\rm T}) \otimes {\mathbb I}] {\bf b} = {\bf 1}_9 \\
    \vdots \\
    [(\sum_{i=7}^9 {\bf e}_i^{\rm T}) \otimes (\sum_{i=7}^9 {\bf e}_i^{\rm T}) \otimes {\mathbb I}] {\bf b} = {\bf 1}_9 
\end{aligned}
$$

which can be expressed as 

$$
    [({\mathbb I}_3 \otimes {\bf 1}_3^{\rm T}) \otimes ({\mathbb I}_3 \otimes {\bf 1}_3^{\rm T}) \otimes {\mathbb I}_9] {\bf b} = {\bf 1}_{81}
$$

All together we have the following set of constraints:

$$
    \begin{bmatrix}
        {\mathbb I}_{9} \otimes {\mathbb I}_{9} \otimes {\bf 1}_9^{\rm T} \\
        {\mathbb I}_9 \otimes {\bf 1}_9^{\rm T} \otimes {\mathbb I}_9 \\
        {\bf 1}_9^{\rm T} \otimes {\mathbb I}_9 \otimes  {\mathbb I}_9 \\
        ({\mathbb I}_3 \otimes {\bf 1}_3^{\rm T}) \otimes ({\mathbb I}_3 \otimes {\bf 1}_3^{\rm T}) \otimes {\mathbb I}_9
    \end{bmatrix} {\bf b} = {\bf 1}_{324}
$$

which can be converted to a QUBO with the standard techniques.