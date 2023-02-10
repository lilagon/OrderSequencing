from gurobipy import *
import numpy as np
import pickle
import sys

n_order = 29
n_SKU = 85
n_seq = n_order
dummy = int(sys.argv[1])

filename = "Instance-"+str(n_order)+"-"+str(n_SKU)+"-"+"OBSP_scenario2"+".pkl"
outfile_name = "MINLP-OrderPicking-Results-main"+"-"+str(n_order)+"-"+str(n_SKU)+"-"+"OBSP_scenario2"+".txt"

with open(filename, 'rb') as f:
    save_dict = pickle.load(f)

f = open(outfile_name,'w')

for n_instance in range(30) :
    f = open(outfile_name, 'a')
    instance = save_dict.get(n_instance)

    input_matrix = instance

    print("input_matrix : ", input_matrix)

    # model
    m = Model('weighted-orderpicking')

    m.setParam("TimeLimit", 3600)
    m.setParam("Threads", 2)

    # 파라미터 설정
    C = m.addVars(n_order, vtype=GRB.CONTINUOUS, name="C")  # 주문 i에 속한 SKU의 피킹시간
    l = m.addVars(n_order, n_SKU, vtype=GRB.BINARY, name="l")

    m.addConstrs(C[i] == quicksum(l[i, a] for a in range(n_SKU)) for i in range(n_order))

    for i in range(n_order):
        for a in range(n_SKU):
            m.addConstr(l[i, a] == input_matrix[i][a])

    # 변수 추가
    r = m.addVars(n_seq, n_SKU, vtype=GRB.BINARY, name="r")
    e = m.addVars(n_order, n_seq, vtype=GRB.BINARY, name="e")
    w = m.addVar(vtype=GRB.CONTINUOUS, name='w')
    g = m.addVar(vtype=GRB.CONTINUOUS, name='g')
    q = m.addVars(n_seq, vtype=GRB.CONTINUOUS, name='q')

    # Objective Function
    m.setObjective(50 * w + g, GRB.MINIMIZE)

    # Constraints
    m.addConstrs(l[i, a] * e[i, s] <= r[s, a] for i in range(n_order) for a in range(n_SKU) for s in range(n_seq))

    m.addConstrs(quicksum(e[i, s] for i in range(n_order)) == 1 for s in range(n_seq))

    m.addConstrs(quicksum(e[i, s] for s in range(n_seq)) == 1 for i in range(n_order))

    m.addConstrs(2 - (r[s, a] + r[t, a]) + r[u, a] >= 1 for a in range(n_SKU) for s in range(n_seq) for t in range(n_seq) for u in range(n_seq) if s < u < t)

    m.addConstrs(quicksum(r[s, a] for a in range(n_SKU)) <= w for s in range(n_seq))

    m.addConstrs(quicksum(C[i] * e[i, s] for i in range(n_order)) == q[s] for s in range(n_seq))

    m.addConstr(quicksum(q[s] * r[s, a] for s in range(n_seq) for a in range(n_SKU)) <= g)

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
                var += 1
            if all_vars[var].startswith('w'):
                w_val.append(all_vals[var])
            else:
                var += 1

        f.write(str(n_instance) + "\t" + str(g_val[0]) + "\t" + str(w_val[0]) + "\t" + str(m.objVal) + "\t" + str(
            computeTime) + "\n")
    else:
        f.write(str(n_instance) + "\t" + str(0) + "\t" + str(computeTime) + "\n")