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

# Set values of input parameters
dispatch_cost = {'G1':75,'G2':6,'G3':80,'G4':4} # Generation costs in DKK/MWh
reserve_cost_up = {'G1':77,'G2':0,'G3':10,'G4':0} # costs for upward reserve in DKK/MW
reserve_cost_down = {'G1':74,'G2':0,'G3':1,'G4':0} # costs for downward reserve in DKK/MW
adjustment_cost_up = {g:dispatch_cost[g]+2 for g in GENERATORS} # costs for upward adjustments in real time in DKK/MWh
adjustment_cost_down = {g:dispatch_cost[g]-1 for g in GENERATORS} # costs for downward adjustments in real time in DKK/MWh
generation_capacity = {'G1':100,'G2':75,'G3':50,'G4':75} # Generators capacity (Q^G_i) in MW
adjustment_capacity_up = {'G1':10,'G2':75,'G3':50,'G4':75} # upward adjustment capacity (Q^up_i) in MW
adjustment_capacity_down = {'G1':10,'G2':75,'G3':50,'G4':75} # downward adjustment capacity (Q^dw_i) in MW
wind_availability_scenario = {('G2','S1'):0.6,('G2','S2'):0.7,('G2','S3'):0.75,('G2','S4'):0.85,
                              ('G4','S1'):0.6,('G4','S2'):0.7,('G4','S3'):0.75,('G4','S4'):0.85} # scenarios of available wind production -
scenario_probability = {'S1':0.25,'S2':0.25,'S3':0.4,'S4':0.1} # probability of scenarios of available wind production -
wind_availability_expected = {g:sum(scenario_probability[k]*wind_availability_scenario[g,k] for k in SCENARIOS) for g in WIND_GENERATORS}
wind_availability_standard_deviation = {g:0.09 for g in WIND_GENERATORS}
wind_availability_min ={g: wind_availability_expected[g] - 0.2 for g in WIND_GENERATORS} # min available wind production (normalized)
wind_availability_max = {g:wind_availability_expected[g] + 0.2 for g in WIND_GENERATORS} # max available wind production (normalized)
load_capacity = {'D1':200} # inflexible load consumption

#probability of violation in chance-constraints
epsilon = 0.1

#%% Build and solve robust ED models

def _solve_reserve_dimensioning_model_():
    
    # Create a Gurobi model for the optimization problem
    DA_model = gb.Model(name='Day-ahead economic dispatch and reserve dimensioning problem')
        
        
    # Set time limit
    DA_model.Params.TimeLimit = 100
    
    # Add variables to the Gurobi model
    # first-stage variables
    generator_dispatch = {g:DA_model.addVar(lb=0,ub=generation_capacity[g],name='dispatch of generator {0}'.format(g)) for g in GENERATORS} # electricity production of generators (x^G_i)        
    generator_reserve_up = {g:DA_model.addVar(lb=0,ub=adjustment_capacity_up[g],name='upward reserve of generator {0}'.format(g)) for g in GENERATORS} # upward reserves of generators (x^G_i)        
    generator_reserve_down = {g:DA_model.addVar(lb=0,ub=adjustment_capacity_down[g],name='downward reserve of generator {0}'.format(g)) for g in GENERATORS} # downward reserves of generators (x^G_i)        
     
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
        
    # linear decision rules for second-stage variables
    generator_adjustment_up_0 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_down_0 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_up_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0}'.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_down_2 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_up_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='upward adjustment LDR parameter of generator {0}'.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)
    generator_adjustment_down_4 = {g:DA_model.addVar(lb=-gb.GRB.INFINITY,name='downward adjustemnt LDR parameter of generator {0} '.format(g)) for g in GENERATORS} # electricity production adjustment of generators in real time (\Delta x^G_i)

    # update gurobi model
    DA_model.update()
    
    # Set objective function and optimization direction of the Gurobi model
    total_cost = gb.quicksum(dispatch_cost[g]*generator_dispatch[g] + reserve_cost_up[g]*generator_reserve_up[g] + reserve_cost_down[g]*generator_reserve_down[g] + adjustment_cost_up[g]*generator_adjustment_up_0[g] - adjustment_cost_down[g]*generator_adjustment_down_0[g] + wind_availability_expected['G2']*(adjustment_cost_up[g]*generator_adjustment_up_2[g] - adjustment_cost_down[g]*generator_adjustment_down_2[g]) + wind_availability_expected['G4']*(adjustment_cost_up[g]*generator_adjustment_up_4[g] - adjustment_cost_down[g]*generator_adjustment_down_4[g]) for g in GENERATORS) # expected electricity production cost (z)   
    DA_model.setObjective(total_cost, gb.GRB.MINIMIZE) #minimize cost

    # Add constraints to the Gurobi model
    # DA balance equation
    DA_balance_constraint = DA_model.addConstr(
            gb.quicksum(generator_dispatch[g] for g in GENERATORS),
            gb.GRB.EQUAL,
            gb.quicksum(load_capacity[d] for d in LOADS),name='Day-ahead balance equation')
 
    # DA_dispatch_min_constraint
    DA_dispatch_min_constraint = {g:DA_model.addConstr(generator_dispatch[g]-generator_reserve_down[g], 
                                                       gb.GRB.GREATER_EQUAL,
                                                       0,name='day-ahead dispatch and reserved capacity lower bound') for g in GENERATORS}
    # DA_dispatch_max_constraint for synchronous generators
    DA_dispatch_max_constraint = {}
    for g in CONTROLLABLE_GENERATORS:
        DA_dispatch_max_constraint[g] = DA_model.addConstr(generator_dispatch[g]+generator_reserve_up[g], 
                                                           gb.GRB.LESS_EQUAL,
                                                           generation_capacity[g],name='day-ahead dispatch and reserved capacity upper bound')
    
    # reformualtion of chance-constrained DA_dispatch_max_constraint for wind generators
    for g in WIND_GENERATORS:
        DA_dispatch_max_constraint[g] = DA_model.addConstr(generator_dispatch[g]+generator_reserve_up[g]+norm.ppf(1-epsilon)*wind_availability_standard_deviation[g]*generation_capacity[g], 
                                                           gb.GRB.LESS_EQUAL,
                                                           wind_availability_expected[g]*generation_capacity[g],name='day-ahead dispatch and reserved capacity upper bound')

    #reformulation of robust RT_balance_constriant        
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
                                                       generator_reserve_up[g]-generator_adjustment_up_0[g]-wind_availability_expected['G2']*generator_adjustment_up_2[g]-wind_availability_expected['G4']*generator_adjustment_up_4[g],name='chance-constrained reformualtion of adjustment_up_max_constraint 2/2') for g in GENERATORS} 

    # reformualtion of chance-constrained adjustment_down_max_constraint
    adjustement_down_max_constraint_1 = {g:DA_model.addQConstr(auxiliary_variable_2_2[g]**2+auxiliary_variable_2_4[g]**2, 
                                                       gb.GRB.LESS_EQUAL,
                                                       auxiliary_variable_4[g]**2,name='chance-constrained reformualtion of adjustment_down_max_constraint 1/2') for g in GENERATORS} 

    adjustement_down_max_constraint_2 = {g:DA_model.addConstr(auxiliary_variable_4[g], 
                                                       gb.GRB.EQUAL,
                                                       generator_reserve_down[g]-generator_adjustment_down_0[g]-wind_availability_expected['G2']*generator_adjustment_down_2[g]-wind_availability_expected['G4']*generator_adjustment_down_4[g],name='chance-constrained reformualtion of adjustment_down_max_constraint 2/2') for g in GENERATORS} 

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
    optimal_reserve_cost = sum(reserve_cost_up[g]*generator_reserve_up[g].x + reserve_cost_down[g]*generator_reserve_down[g].x for g in GENERATORS)
    optimal_DA_dispatch = {g:generator_dispatch[g].x for g in GENERATORS}
    optimal_reserve_up = {g:generator_reserve_up[g].x for g in GENERATORS}
    optimal_reserve_down = {g:generator_reserve_down[g].x for g in GENERATORS}
    
    return optimal_DA_objval, optimal_DA_cost, optimal_reserve_cost, optimal_DA_dispatch, optimal_reserve_up, optimal_reserve_down


#%% initialize lists of results for all models

optimal_DA_objval = {} # save objective value of DA dispatch optimization problem at optimality
optimal_DA_cost = {} # save day-ahead cost of optimization problem at optimality
optimal_reserve_cost = {} # save reserve cost of optimization problem at optimality
optimal_DA_dispatch = {} # save values of generators dispatch
optimal_reserve_up = {} # save values of generators upward reserves
optimal_reserve_down = {} # save values of generators downward reserves

#%% solve day-ahead economic dispatch for all model types

optimal_DA_objval, optimal_DA_cost, optimal_reserve_cost, optimal_DA_dispatch, optimal_reserve_up, optimal_reserve_down = _solve_reserve_dimensioning_model_()

#%% print day-ahead results

print('System Costs:')
print('optimal objective value:', optimal_DA_objval)
print('optimal DA dispatch cost:',optimal_DA_cost)
print('optimal DA reserve cost:',optimal_reserve_cost)
print('optimal DA dispatch:',optimal_DA_dispatch)
print('optimal upward reserve:',optimal_reserve_up)
print('optimal downward reserve:',optimal_reserve_down)

