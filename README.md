# ANN--Basics


A feed-forward network with 2 hidden layers for the following problem statement:

"Two-dimensional patterns with x = (x(1),x(2)) are placed at x1 = (1,0), x2 = (−1,0), x3 = (0,1)
and x4 = (0,−1). You are given the classification x1, x2 belong to C1 and x3, x4 belong to C2. In addition. We also
add 100 − 4 random 2D patterns (giving us a total of 100 patterns with 50 patterns in each class)
obeying the criteria

If x(1) > 1 and x(2) < 1; then x 2 C1;
If x(1) < 1 and x(2) > 1; then x 2 C2;

Other locations are not permitted:
Implement a fully connected multi-layer backpropagation network from scratch using sigmoidal units
(including in the output layer and using cross-entropy as the output measure).
