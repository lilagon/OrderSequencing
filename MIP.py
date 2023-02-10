from gurobipy import *
import numpy as np
import pickle
import sys

n_order = 29
n_SKU = 85
n_seq = n_order

dummy = int(sys.argv[1])

filename = "Instance-"+str(n_order)+"-"+str(n_SKU)+"-"+"OBSP_scenario2"+".pkl"
outfile_name = "MIP-OrderPicking-Results-main"+"-"+str(n_order)+"-"+str(n_SKU)+"-"+"OBSP_scenario2"+".txt"

with open(filename, 'rb') as f:
    save_dict = pickle.load(f)

f = open(outfile_name,'w')

for n_instance in range(30) :
    f = open(outfile_name, 'a')
    instance = save_dict.get(n_instance)

    input_matrix = instance

    print("input_matrix : ", input_matrix)

    param = []
    for row in range(len(input_matrix)):
        param.append(sum(input_matrix[row]))

    m = Model('Machine-Part grouping1')
    m.setParam("TimeLimit", 3600)
    m.setParam("Threads", 2)

    # 파라미터
    C = m.addVars(n_order, vtype=GRB.CONTINUOUS, name='C')
    m.addConstrs(C[i] == param[i] for i in range(n_order))  # 7

    R = m.addVars(n_order, n_SKU, vtype=GRB.CONTINUOUS, name='R')
    m.addConstrs(R[i, a] == input_matrix[i][a] for i in range(n_order) for a in range(n_SKU))

    # 결정변수
    e = m.addVars(n_order, n_seq, vtype=GRB.BINARY, name='e')
    l = m.addVars(n_order, n_seq, n_SKU, vtype=GRB.BINARY, name='l')
    w = m.addVar(vtype=GRB.CONTINUOUS, name='w')
    g = m.addVar(vtype=GRB.CONTINUOUS, name='g')

    # 목적함수
    m.setObjective(50 * w + g, GRB.MINIMIZE)
    #m.setObjective(g, GRB.MINIMIZE)

    # 제약
    m.addConstrs(R[i, a] * e[i, h] <= l[i, h, a] for i in range(n_order) for a in range(n_SKU) for h in range(n_seq))  # 3

    m.addConstrs(quicksum(e[i, h] for i in range(n_order)) == 1 for h in range(n_seq))  # 5

    m.addConstrs(quicksum(e[i, h] for h in range(n_seq)) == 1 for i in range(n_order))  # 6

    m.addConstrs((2 - (quicksum(l[i, h, a] for i in range(n_order)) + quicksum(l[j, d, a] for j in range(n_order))) + quicksum(l[k, g, a] for k in range(n_order))) >= 1
                 for h in range(n_seq) for d in range(n_seq) for g in range(n_seq) for a in range(n_SKU) if h < g < d)  # 8

    m.addConstrs(l[i, h, a] <= e[i, h] for i in range(n_order) for h in range(n_seq) for a in range(n_SKU))  # 4

    m.addConstrs(quicksum(C[i] * l[i, h, a] for i in range(n_order) for h in range(n_seq) for a in range(n_SKU)) <= g for i in range(n_order) for h in range(n_seq))  # 10

    m.addConstrs(quicksum(l[i, h, a] for a in range(n_SKU)) <= w for i in range(n_order) for h in range(n_seq))

    m.optimize()

    computeTime = m.Runtime

    all_vars = []
    all_vals = []
    g_val = []
    w_val = []

    if m.solCount > 0:
        print('obj : %g' % m.objval)
        for v in m.getVars():
            if v.x != 0:
                print('%s %g' % (v.varName, v.x))

            all_vars.append(v.varName)
            all_vals.append(v.x)

        length = len(all_vars)

        for var in range(length):
            if all_vars[var].startswith('g'):
                g_val.append(all_vals[var])

            if all_vars[var].startswith('w'):
                w_val.append(all_vals[var])
            else:
                pass

        print('w_val = ', w_val)

        f.write(str(n_instance) + "\t" + str(g_val[0]) + "\t" + str(w_val[0]) + "\t" + str(m.objVal) + "\t" + str(
            computeTime) + "\n")
    else:
        f.write(str(n_instance) + "\t" + str(0) + "\t" + str(computeTime) + "\n")