import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
import pickle
import sys
import time

n_population = 100
n_generation = 50
crossover = 0.6
mutation = 0.2
n_order = 29
n_SKU = 85

filename = "Instance-"+str(n_order)+"-"+str(n_SKU)+"-"+"OBSP_scenario2"+".pkl"
outfile_name = "NSGA-multi-OBSP2-"+str(n_population)+"-"+str(crossover)+"-"+str(mutation)+"-"+str(n_generation)+"-"+str(n_order)+"-"+str(n_SKU)+".txt"

with open(filename, 'rb') as f:
    save_dict = pickle.load(f)

f = open(outfile_name,'w')

for n_instance in save_dict.keys() :
    start = time.time()
    f = open(outfile_name, 'a')

    instance = save_dict.get(n_instance)

    input_matrix = instance

    print("input_matrix : ")
    print(input_matrix)

    sum_order_num = int(0)

    n_lb = [0 for i in range(n_order)]
    n_ub = [n_order-1 for j in range(n_order)]

    for order in range(n_order) :
        sum_order_num += order

    vtype = []
    for a in range(n_order) :
        vtype.append("int")

    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=n_order,
                            n_obj=2,
                            xl=np.array(n_lb),
                            xu=np.array(n_ub),
                            mask=vtype)

        def _evaluate(self, x, out, *args, **kwargs):

            output_mat = []

            for num in range(n_order) :
                output_mat.append(input_matrix[x[num]])

            output_matrix = np.array(output_mat)

            param = []
            for row in range(len(output_matrix)):
                param.append(sum(input_matrix[x[row]]))

            # [1,0,0,1] -> [1,1,1,1]
            for column in range(len(output_matrix[0])) :
                for row1 in range(n_order) :
                    for row2 in range(row1+2, n_order) :
                        if output_matrix[row1][column] == 1 and output_matrix[row2][column] == 1 :
                            for row3 in range(row1+1, row2) :
                                output_matrix[row3][column] = 1

            sum_row = []
            for row in range(len(output_matrix)):
                sum_row.append(sum(output_matrix[row]))

            capacity = max(sum_row)

            for i in range(len(param)):
                for j in range(len(output_matrix[0])):
                    output_matrix[i][j] = param[i] * output_matrix[i][j]

            list_sum = []
            for row in range(len(output_matrix)) :
                list_sum.append(sum(output_matrix[row]))

            sum_element = sum(list_sum)

            f1 = sum_element
            f2 = capacity

            out["F"] = [f1,f2]

    problem = MyProblem()

    algorithm = NSGA2(
        pop_size=n_population,
        n_offsprings=n_population,
        sampling=get_sampling("perm_random"),
        crossover=get_crossover("perm_ox", prob= crossover),
        mutation=get_mutation("perm_inv", prob = mutation),
        eliminate_duplicates=True
    )

    from pymoo.factory import get_termination

    termination = get_termination("n_gen", n_generation)

    from pymoo.optimize import minimize

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    print("F :", F)

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    print('approx_ideal :', approx_ideal)
    print('approx_nadir :', approx_nadir)

    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    print('nF :', nF)

    fl = nF.min(axis=0)
    fu = nF.max(axis=0)

    print('fl :', fl)
    print('fu :', fu)

    weights = np.array([0.5, 0.5])

    from pymoo.decomposition.asf import ASF

    decomp = ASF()

    i = decomp.do(nF, 1 / weights).argmin()
    print("Best regarding ASF: Point \n i = %s \n F = %s" % (i, F[i]))

    solution = []

    f1 = F[i][0]
    f2 = F[i][1]

    solution.append(f1)
    solution.append(f2)

    end = time.time()

    computeTime = end - start

    f.write(str(n_instance) + "\t" + str(solution[0]) + "\t" + str(solution[1]) + "\t" + str(computeTime) + "\n")