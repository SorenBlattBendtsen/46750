import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

#Assignment variables applicable across all tasks
n_bus = 24
wind_MWp = 300
S_base_3ph = 100

def read_data(data:str, path:str = 'data/', wind_hour: int = 9, wind_scenarios: list = list(range(100))):
    #path is the path to the folder containing the files for the assignment

    #data specifies which data should be read and processed:
    # 'gen_costs'
    # 'gen_data'
    # 'line_data'
    # 'system_demand'
    # 'load_distribution'
    # 'wind_data'
    # 'branch_matrix'

    gen_costs = pd.read_excel(path + 'gen_costs.xlsx')
    gen_costs[gen_costs.columns[1:]] = gen_costs[gen_costs.columns[1:]].values

    gen_data = pd.read_excel(path + 'gen_data.xlsx')
    line_data = pd.read_excel(path + 'line_data.xlsx')

    system_demand = pd.read_excel(path + 'system_demand.xlsx')
    load_distribution = pd.read_excel(path + 'load_distribution.xlsx')

    # Wind data processing
    wind_data_list = []
    for i in range(6):
        wt = pd.read_csv(path + 'wind %d.out' % (i + 1))
        wt = wt.drop(columns = ['Unnamed: 0']) #dropping the unnecessary index column
        wt['Expected'] = wt.mean(numeric_only=True, axis=1)

        wt = wt[list(wt.columns[wind_scenarios]) + ['Expected']] * wind_MWp

        wt = wt.loc[wt.index == wind_hour]

        wind_data_list.append(wt)

    wind_data = pd.concat(wind_data_list, axis=0).reset_index(drop=True)
    wind_data.index = np.arange(1,7)
    wind_data.index.name = 'Wind Farm'


    #Filling out branch susceptance matrix and adjusting branch data
    line_data['Susceptance pu'] = 1 / line_data['Reactance pu']
    line_data['Capacity pu'] = line_data['Capacity MVA'] / S_base_3ph
    
    branch_matrix = np.zeros((n_bus, n_bus))

    for n in range(1, n_bus + 1):
        branch_matrix[n - 1][n - 1] = line_data.loc[(line_data['From'] == n) | (line_data['To'] == n), 'Susceptance pu'].sum()
        for k in range(n, n_bus + 1): #start from n
            if k != n:
                branch_matrix[n - 1][k - 1] = -1 * (line_data.loc[(line_data['From'] == n) & (line_data['To'] == k), 'Susceptance pu'].sum())
                branch_matrix[k - 1][n - 1] = branch_matrix[n - 1][k - 1] #symmetric

    #Since the process is so fast, we might as well just handle all the data at every function call and return the desired data
    
    if data == 'gen_costs':
        return gen_costs
    elif data == 'gen_data':
        return gen_data
    elif data == 'line_data':
        return line_data
    elif data == 'system_demand':
        return system_demand
    elif data == 'load_distribution':
        return load_distribution
    elif data == 'wind_data':
        return wind_data
    elif data == 'branch_matrix':
        return branch_matrix
    else:
        print('Invalid input.')
        return None

def mapping_dictionaries(gen_data, wf_nodes:list = [2, 4, 6, 15, 20, 22]):
    #These dictionarys return the 0-indexed indices of generators or wind farms located at the node n

    #for example, gens_map.get(0) will return the list [0] because generator 1 is placed at node 1
    #gens_map.get(14) will return [4, 5] because generators 5 and 6 are at node 15

    #if no wind farms or generators are located at node n, it will return an empty list

    gens_map = {}

    for n in range(1, n_bus + 1):
        gens_map[n - 1] = (gen_data['Unit #'][gen_data['Node'] == n] - 1).tolist()

    wf_map = {}

    for n in range(n_bus):
        wf_map[n] = np.where(np.array(wf_nodes) == n)[0].tolist()

    return gens_map, wf_map