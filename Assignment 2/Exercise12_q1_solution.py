#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:06:45 2023

@author: lesiamitridati
"""


import gurobipy as gb
from gurobipy import GRB
import pandas as pd
from scipy.stats import norm
import numpy as np


# Define ranges and hyperparameters  
CONTROLLABLE_GENERATORS = ['G1','G3'] #range of controllable generators
WIND_GENERATORS = ['G2','G4'] #range of wind generators
GENERATORS = ['G1','G2','G3','G4'] #range of all generators
LOADS = ['D1'] #range of Loads
SCENARIOS = ['S1','S2','S3','S4'] # range of wind production scenarios
MODEL_TYPES = ['robust - expected','worst-case','chance-constrained','deterministic'] # types of ED models (robust - expected ; worst-case; uncertainty budget)

# Set values of input parameters
dispatch_cost = {'G1':75,'G2':6,'G3':80,'G4':4} # Generation costs (c^G_i) in DKK/MWh
adjustment_cost_up = {g:dispatch_cost[g]+2 for g in GENERATORS} # costs for upward adjustments in real time (c^up_i) in DKK/MWh
adjustment_cost_down = {g:dispatch_cost[g]-1 for g in GENERATORS} # costs for downward adjustments in real time (c^dw_i) in DKK/MWh
generation_capacity = {'G1':100,'G2':75,'G3':50,'G4':75} # Generators capacity (Q^G_i) in MW
adjustment_capacity_up = {'G1':10,'G2':75,'G3':50,'G4':75} # upward adjustment capacity (Q^up_i) in MW
adjustment_capacity_down = {'G1':10,'G2':75,'G3':50,'G4':75} # downward adjustment capacity (Q^dw_i) in MW
wind_availability_scenario = {('G2','S1'):0.6,('G2','S2'):0.7,('G2','S3'):0.75,('G2','S4'):0.85,
                              ('G4','S1'):0.6,('G4','S2'):0.7,('G4','S3'):0.75,('G4','S4'):0.85} # scenarios of available wind production -
scenario_probability = {'S1':0.25,'S2':0.25,'S3':0.4,'S4':0.1} # probability of scenarios of available wind production -
wind_availability_expected = {g:sum(scenario_probability[k]*wind_availability_scenario[g,k] for k in SCENARIOS) for g in WIND_GENERATORS}
wind_availability_standard_deviation = {g:0.09 for g in WIND_GENERATORS}
wind_availability_min ={g: wind_availability_expected[g] - 0.29 for g in WIND_GENERATORS} # min available wind production (normalized)
wind_availability_max = {g:wind_availability_expected[g] + 0.29 for g in WIND_GENERATORS} # max available wind production (normalized)
load_capacity = {'D1':200} # inflexible load consumption

#probability of violation in chance-constraints
epsilon = 0.01

#%% Build and solve robust ED models

def _solve_DA_model_(model_type):
    
    # Create a Gurobi model for the optimization problem
    DA_model = gb.Model(name='Day-ahead economic dispatch problem')
        
        
    # Set time limit
    DA_model.Params.TimeLimit = 100

    
    # Add variables to the Gurobi model
    generator_dispatch = {g:DA_model.addVar(lb=0,ub=generation_capacity[g],name='dispatch of generator {0}'.format(g)) for g in GENERATORS} # electricity production of generators (x^G_i)        
        
    if model_type == 'robust - expected' or model_type == 'worst-case' or model_type == 'chance-constrained':       
        # linear decision rules 
        generator_adjustment_up_0 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
        generator_adjustment_down_0 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
        generator_adjustment_up_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0}'.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
        generator_adjustment_down_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
        generator_adjustment_up_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0}'.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
        generator_adjustment_down_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)

    if model_type == 'chance-constrained':       
        # auxiliary variables for SOC constraints (non-negative)
        auxiliary_variable_1 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_2 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_3 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_4 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_5 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_6 = {g:DA_model.addVar(lb=0) for g in GENERATORS}
        auxiliary_variable_1_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_1_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_2_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_2_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_5_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_5_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in GENERATORS}
        auxiliary_variable_6_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in WIND_GENERATORS}
        auxiliary_variable_6_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY) for g in WIND_GENERATORS}
        
    
    # update gurobi model
    DA_model.update()
    
    # Set objective function and optimization direction of the Gurobi model
    if model_type == 'robust - expected' or model_type == 'chance-constrained':
        total_cost = gb.quicksum(dispatch_cost[g]*generator_dispatch[g] + adjustment_cost_up[g]*generator_adjustment_up_0[g] - adjustment_cost_down[g]*generator_adjustment_down_0[g] + wind_availability_expected['G2']*(adjustment_cost_up[g]*generator_adjustment_up_2[g] - adjustment_cost_down[g]*generator_adjustment_down_2[g]) + wind_availability_expected['G4']*(adjustment_cost_up[g]*generator_adjustment_up_4[g] - adjustment_cost_down[g]*generator_adjustment_down_4[g]) for g in GENERATORS) # expected electricity production cost (z)   
    DA_model.setObjective(total_cost, gb.GRB.MINIMIZE) #minimize cost

    # Add constraints to the Gurobi model
    DA_balance_constraint = DA_model.addConstr(
            gb.quicksum(generator_dispatch[g] for g in GENERATORS),
            gb.GRB.EQUAL,
            gb.quicksum(load_capacity[d] for d in LOADS),name='Day-ahead balance equation')

        
    if model_type == 'robust - expected' or model_type == 'worst-case' or model_type == 'chance-constrained':
       
        #refomrulation of robust RT_balance_constriant        
        RT_balance_constraint_0 = DA_model.addConstr(
                gb.quicksum(generator_adjustment_up_0[g] - generator_adjustment_down_0[g] for g in GENERATORS),
                gb.GRB.EQUAL,
                0,name='real-time balance equation (intercept of LDR)')
    
        RT_balance_constraint_2 = DA_model.addConstr(
                gb.quicksum(generator_adjustment_up_2[g] - generator_adjustment_down_2[g] for g in GENERATORS),
                gb.GRB.EQUAL,
                0,name='real-time balance equation (slope of LDR associated with wind power of G2)')
    
        RT_balance_constraint_4 = DA_model.addConstr(
                gb.quicksum(generator_adjustment_up_4[g] - generator_adjustment_down_4[g] for g in GENERATORS),
                gb.GRB.EQUAL,
                0,name='real-time balance equation (slope of LDR associated with wind power of G4)')


    if model_type == 'chance-constrained':
        # reformualtion of chance-constrained adjustment_up_min_constraint
        adjustement_up_min_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_1_2[g]**2+auxiliary_variable_1_4[g]**2, 
                                                           gb.GRB.LESS_EQUAL,
                                                           auxiliary_variable_1[g]**2,name='chance-constrained reformualtion of adjustment_up_min_constraint 1/2') for g in GENERATORS} 

        adjustement_up_min_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_1[g], 
                                                           gb.GRB.EQUAL,
                                                           generator_adjustment_up_0[g]+wind_availability_expected['G2']*generator_adjustment_up_2[g]+wind_availability_expected['G4']*generator_adjustment_up_4[g],name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        adjustement_up_min_constraint_3 = {g:DA_model.addConstr(auxiliary_variable_1_2[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G2']*generator_adjustment_up_2[g],name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        adjustement_up_min_constraint_4 = {g:DA_model.addConstr(auxiliary_variable_1_4[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G4']*generator_adjustment_up_4[g],name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        # reformualtion of chance-constrained adjustment_down_min_constraint
        adjustement_down_min_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_2_2[g]**2+auxiliary_variable_2_4[g]**2, 
                                                           gb.GRB.LESS_EQUAL,
                                                           auxiliary_variable_2[g]**2,name='chance-constrained reformualtion of adjustment_down_min_constraint 1/2') for g in GENERATORS} 

        adjustement_down_min_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_2[g], 
                                                           gb.GRB.EQUAL,
                                                           generator_adjustment_up_0[g]+wind_availability_expected['G2']*generator_adjustment_down_2[g]+wind_availability_expected['G4']*generator_adjustment_down_4[g],name='chance-constrained reformualtion of adjustment_down_min_constraint 2/2') for g in GENERATORS} 

        adjustement_down_min_constraint_3 = {g:DA_model.addConstr(auxiliary_variable_2_2[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G2']*generator_adjustment_down_2[g],name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        adjustement_down_min_constraint_4 = {g:DA_model.addConstr(auxiliary_variable_2_4[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G4']*generator_adjustment_down_4[g],name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        # reformualtion of chance-constrained adjustment_up_max_constraint
        adjustement_up_max_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_1_2[g]**2+auxiliary_variable_1_4[g]**2, 
                                                           gb.GRB.LESS_EQUAL,
                                                           auxiliary_variable_3[g]**2,name='chance-constrained reformualtion of adjustment_up_max_constraint 1/2') for g in GENERATORS} 

        adjustement_up_max_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_3[g], 
                                                           gb.GRB.EQUAL,
                                                           adjustment_capacity_up[g]-generator_adjustment_up_0[g]-wind_availability_expected['G2']*generator_adjustment_up_2[g]-wind_availability_expected['G4']*generator_adjustment_up_4[g],name='chance-constrained reformualtion of adjustment_up_max_constraint 2/2') for g in GENERATORS} 

        # reformualtion of chance-constrained adjustment_down_max_constraint
        adjustement_down_max_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_2_2[g]**2+auxiliary_variable_2_4[g]**2, 
                                                           gb.GRB.LESS_EQUAL,
                                                           auxiliary_variable_4[g]**2,name='chance-constrained reformualtion of adjustment_down_max_constraint 1/2') for g in GENERATORS} 

        adjustement_down_max_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_4[g], 
                                                           gb.GRB.EQUAL,
                                                           adjustment_capacity_down[g]-generator_adjustment_down_0[g]-wind_availability_expected['G2']*generator_adjustment_down_2[g]-wind_availability_expected['G4']*generator_adjustment_down_4[g],name='chance-constrained reformualtion of adjustment_down_max_constraint 2/2') for g in GENERATORS} 

        # reformulation of chance-constrained RT_min_production_constraint
        RT_min_production_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_5_2[g]**2+auxiliary_variable_5_4[g]**2, 
                                                           gb.GRB.LESS_EQUAL,
                                                           auxiliary_variable_5[g]**2,name='chance-constrained reformualtion of RT_min_production_constraint 1/2') for g in GENERATORS} 

        RT_min_production_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_5[g], 
                                                           gb.GRB.EQUAL,
                                                           generator_dispatch[g]+generator_adjustment_up_0[g]-generator_adjustment_down_0[g]+wind_availability_expected['G2']*(generator_adjustment_up_2[g]-generator_adjustment_down_2[g])+wind_availability_expected['G4']*(generator_adjustment_up_4[g]-generator_adjustment_down_4[g]),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') for g in GENERATORS} 

        RT_min_production_constraint_3 = {g:DA_model.addConstr(auxiliary_variable_5_2[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G2']*(generator_adjustment_up_2[g]-generator_adjustment_down_2[g]),name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        RT_min_production_constraint_4 = {g:DA_model.addConstr(auxiliary_variable_5_4[g], 
                                                           gb.GRB.EQUAL,
                                                           norm.ppf(1-epsilon)*wind_availability_standard_deviation['G4']*(generator_adjustment_up_4[g]-generator_adjustment_down_4[g]),name='chance-constrained reformualtion of adjustment_up_min_constraint 2/2') for g in GENERATORS} 

        # reformulation of chance-constrained RT_max_production_constraint 
        RT_max_production_constraint_1 = {}
        RT_max_production_constraint_2 = {}
        RT_max_production_constraint_3 = {}
        RT_max_production_constraint_4 = {}
        
        for g in CONTROLLABLE_GENERATORS:
            RT_max_production_constraint_1[g] = DA_model.addQConstr(auxiliary_variable_5_2[g]**2+auxiliary_variable_5_4[g]**2, 
                                                               gb.GRB.LESS_EQUAL,
                                                               auxiliary_variable_6[g]**2,name='chance-constrained reformualtion of RT_min_production_constraint 1/2') 

            RT_max_production_constraint_2[g] = DA_model.addConstr(auxiliary_variable_6[g], 
                                                               gb.GRB.EQUAL,
                                                               generation_capacity[g]-generator_dispatch[g]-generator_adjustment_up_0[g]+generator_adjustment_down_0[g]-wind_availability_expected['G2']*(generator_adjustment_up_2[g]-generator_adjustment_down_2[g])-wind_availability_expected['G4']*(generator_adjustment_up_4[g]-generator_adjustment_down_4[g]),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') 


        RT_max_production_constraint_1['G2'] = DA_model.addQConstr(auxiliary_variable_6_2['G2']**2+auxiliary_variable_6_4['G2']**2, 
                                                       gb.GRB.LESS_EQUAL,
                                                       auxiliary_variable_6['G2']**2,name='chance-constrained reformualtion of RT_min_production_constraint 1/2') 

        RT_max_production_constraint_2['G2'] = DA_model.addConstr(auxiliary_variable_6['G2'], 
                                                       gb.GRB.EQUAL,
                                                       -generator_dispatch['G2']-generator_adjustment_up_0['G2']+generator_adjustment_down_0['G2']-wind_availability_expected['G2']*(generator_adjustment_up_2['G2']-generator_adjustment_down_2['G2']-generation_capacity['G2'])-wind_availability_expected['G4']*(generator_adjustment_up_4['G2']-generator_adjustment_down_4['G2']),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') 


        RT_max_production_constraint_3['G2'] = DA_model.addConstr(auxiliary_variable_6_2['G2'], 
                                                       gb.GRB.EQUAL,
                                                       norm.ppf(1-epsilon)*wind_availability_standard_deviation['G2']*(generator_adjustment_up_2['G2']-generator_adjustment_down_2['G2']-generation_capacity['G2']),name='chance-constrained reformualtion of RT_min_production_constraint 1/2') 

        RT_max_production_constraint_4['G2'] = DA_model.addConstr(auxiliary_variable_6_4['G2'], 
                                                       gb.GRB.EQUAL,
                                                       norm.ppf(1-epsilon)*wind_availability_standard_deviation['G4']*(generator_adjustment_up_4['G2']-generator_adjustment_down_4['G2']),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') 


        RT_max_production_constraint_1['G4'] = DA_model.addQConstr(auxiliary_variable_6_2['G4']**2+auxiliary_variable_6_4['G4']**2, 
                                                       gb.GRB.LESS_EQUAL,
                                                       auxiliary_variable_6['G4']**2,name='chance-constrained reformualtion of RT_min_production_constraint 1/2') 

        RT_max_production_constraint_2['G4'] = DA_model.addConstr(auxiliary_variable_6['G4'], 
                                                       gb.GRB.EQUAL,
                                                       -generator_dispatch['G4']-generator_adjustment_up_0['G4']+generator_adjustment_down_0['G4']-wind_availability_expected['G2']*(generator_adjustment_up_2['G4']-generator_adjustment_down_2['G4'])-wind_availability_expected['G4']*(generator_adjustment_up_4['G4']-generator_adjustment_down_4['G4']-generation_capacity['G4']),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') 

        RT_max_production_constraint_3['G4'] = DA_model.addConstr(auxiliary_variable_6_2['G4'], 
                                                       gb.GRB.EQUAL,
                                                       norm.ppf(1-epsilon)*wind_availability_standard_deviation['G2']*(generator_adjustment_up_2['G4']-generator_adjustment_down_2['G4']),name='chance-constrained reformualtion of RT_min_production_constraint 1/2') 

        RT_max_production_constraint_4['G4'] = DA_model.addConstr(auxiliary_variable_6_4['G4'], 
                                                       gb.GRB.EQUAL,
                                                       norm.ppf(1-epsilon)*wind_availability_standard_deviation['G4']*(generator_adjustment_up_4['G4']-generator_adjustment_down_4['G4']-generation_capacity['G4']),name='chance-constrained reformualtion of RT_min_production_constraint 2/2') 

        
    # optimize ED problem (primal)
    DA_model.optimize()
    
    optimal_DA_objval = DA_model.ObjVal
    optimal_DA_cost = sum(dispatch_cost[g]*generator_dispatch[g].x for g in GENERATORS)
    optimal_DA_dispatch = {g:generator_dispatch[g].x for g in GENERATORS}
    optimal_DA_price = DA_balance_constraint.Pi
    
    return optimal_DA_objval, optimal_DA_cost, optimal_DA_dispatch, optimal_DA_price


#%%

# Build and solve RT adjustment models for fixed values of DA dispatch and a given scenario of wind realization
def _solve_RT_model_(optimal_DA_dispatch,scenario):
    
    # Create a Gurobi model for the optimization p[roblem]
    RT_model = gb.Model(name='Real-time adjustment problem')
        
        
    # Set time limit
    RT_model.Params.TimeLimit = 100
    
    
    # Add variables to the Gurobi model
    generator_adjustment_up = {g:RT_model.addVar(lb=0,ub=adjustment_capacity_up[g],name='Electricity upward adjustment of generator {0} in scenario {1}'.format(g,scenario)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_down = {g:RT_model.addVar(lb=0,ub=adjustment_capacity_down[g],name='Electricity downward of generator {0} in scenario {1}'.format(g,scenario)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    
    
    # update gurobi model
    RT_model.update()
    
    
    # Set objective function and optimization direction of the Gurobi model
    RT_cost = gb.quicksum((adjustment_cost_up[g]*generator_adjustment_up[g] - adjustment_cost_down[g]*generator_adjustment_down[g]) for g in GENERATORS) # expected electricity production cost (z)   
    RT_model.setObjective(RT_cost, gb.GRB.MINIMIZE) # Min expected electricity production cost (z): Eq. (1a)    
        
    RT_balance_constraint = RT_model.addConstr(
            gb.quicksum(generator_adjustment_up[g] - generator_adjustment_down[g] for g in GENERATORS),
            gb.GRB.EQUAL,
            0,name='real-time balance equation for scenario {0}'.format(scenario))
    
    adjustment_max_generation_constraint = {} # max generation after adjustement of generators
    for g in CONTROLLABLE_GENERATORS:
        adjustment_max_generation_constraint[g] = RT_model.addConstr(
                optimal_DA_dispatch[g] + generator_adjustment_up[g] - generator_adjustment_down[g],
                gb.GRB.LESS_EQUAL,
                generation_capacity[g],name='max adjusted generation constraint for generator {0} and scenario {1}'.format(g,scenario))
    for g in WIND_GENERATORS:
        adjustment_max_generation_constraint[g] = RT_model.addConstr(
                optimal_DA_dispatch[g] + generator_adjustment_up[g] - generator_adjustment_down[g],
                gb.GRB.LESS_EQUAL,
                generation_capacity[g]*wind_availability_scenario[g,scenario],name='max adjusted generation constraint for generator {0} and scenario {1}'.format(g,scenario))
    
    adjustment_min_generation_constraint = {g: RT_model.addConstr(
                    optimal_DA_dispatch[g] + generator_adjustment_up[g] - generator_adjustment_down[g],
                    gb.GRB.GREATER_EQUAL,
                    0,name='min adjusted generation constraint for generator {0} and scenario {1}'.format(g,scenario)) for g in GENERATORS}
           
    # optimize ED problem (primal)
    RT_model.optimize()
    
    optimal_RT_cost = RT_model.ObjVal
    optimal_RT_adjustment_up = {g:generator_adjustment_up[g].x for g in GENERATORS}
    optimal_RT_adjustment_down = {g:generator_adjustment_down[g].x for g in GENERATORS}
    optimal_RT_adjustment = {g:generator_adjustment_up[g].x - generator_adjustment_down[g].x for g in GENERATORS}
    optimal_RT_price = RT_balance_constraint.Pi
    
    return optimal_RT_cost, optimal_RT_adjustment_up,optimal_RT_adjustment_down,optimal_RT_adjustment, optimal_RT_price


#%% initialize lists of results for all models

optimal_DA_objval = {} # save objective value of DA dispatch optimization problem at optimality
optimal_DA_cost = {} # save day-ahead cost of optimization problem at optimality
optimal_RT_cost = {} # save real-time cost of optimization problem at optimality
optimal_total_cost = {} #save otal (day-ahead + rela-time) cost of model
optimal_DA_dispatch = {} # save values of pgenerators dispatch
optimal_DA_price = {} # save values of pgenerators dispatch
optimal_RT_adjustment_up = {} # save values of adjustment up
optimal_RT_adjustment_down = {} # save values of adjustment down
optimal_RT_adjustment = {} # save values of adjustment
optimal_DA_price = {} #save values of day-ahead prices
optimal_RT_price = {} #save values of real time prices
expected_RT_cost = {} #save expected RT costs
expected_total_cost = {} #save expected RT costs

#%% solve day-ahead economic dispatch for all model types

for model_type in MODEL_TYPES:
    optimal_DA_objval[model_type], optimal_DA_cost[model_type] , optimal_DA_dispatch[model_type], optimal_DA_price[model_type] = _solve_DA_model_(model_type)

#%% print day-ahead results

for model_type in MODEL_TYPES:
    print(model_type)   

    print('System Costs:')
    print('optimal objective value:', optimal_DA_objval[model_type])
    print('optimal DA cost:',optimal_DA_cost[model_type])

#%%  solve RT adjustmewnt models for all fixed values of day-ahead dispatch  

for model_type in MODEL_TYPES:
    for scenario in SCENARIOS:
        optimal_RT_cost[model_type,scenario] , optimal_RT_adjustment_up[model_type,scenario], optimal_RT_adjustment_down[model_type,scenario], optimal_RT_adjustment[model_type,scenario], optimal_RT_price[model_type,scenario] = _solve_RT_model_(optimal_DA_dispatch[model_type],scenario)
        optimal_total_cost[model_type,scenario] = optimal_DA_cost[model_type] + optimal_RT_cost[model_type,scenario]        
    expected_RT_cost[model_type] = sum(scenario_probability[s]*optimal_RT_cost[model_type,s] for s in SCENARIOS)
    expected_total_cost[model_type] = optimal_DA_cost[model_type] + expected_RT_cost[model_type]      

#%% print RT results

for model_type in MODEL_TYPES:    
    print(model_type)
    
    print('System Costs:')
    print('expected total cost:', expected_total_cost[model_type])
    print('max total cost over all scenarios:',max([optimal_total_cost[model_type,scenario] for scenario in SCENARIOS]))
