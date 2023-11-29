#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:15:24 2023

@author: lesiamitridati
"""

import gurobipy as gp
from gurobipy import GRB

# Create the Gurobi models for the two subproblems
original_problem = gp.Model("original")

# Variables for subproblem 1
x1_original = original_problem.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1 _ original")

# Variables for subproblem 2
x2_original = original_problem.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2 _ original")

# Update subproblems 1 and 2
original_problem.update()

c1_original = original_problem.addConstr(x1_original+x2_original,gp.GRB.GREATER_EQUAL,4)

# Set objective functions for original problem
original_problem.setObjective(x1_original**2 + x2_original**2, GRB.MINIMIZE)

original_problem.optimize()
                
#%% Lagrangian relaxation algorithm

def lagrangian_relaxation_decomposition(lambda_0,a,b,epsilon,iterations):

    # initialize Lagrange multipliers:
    lambda_values = [lambda_0]
    
    #initialize solutions of subproblems
    obj_values = []
    x1_values = []
    x2_values = []
    
    # Create the Gurobi models for the two subproblems
    subproblem1 = gp.Model("subproblem1")
    subproblem2 = gp.Model("subproblem2")

    # Variables for subproblem 1
    x1 = subproblem1.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1")
    
    # Variables for subproblem 2
    x2 = subproblem2.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2")

    # Update subproblems 1 and 2
    subproblem1.update()
    subproblem2.update()

    # initialize iteration
    iteration=0

    while iteration<=iterations:    

            # Set objective functions for subproblems
            subproblem1.setObjective(x1**2 - lambda_values[-1]*x1, GRB.MINIMIZE)
            subproblem2.setObjective(x2**2 - lambda_values[-1]*x2, GRB.MINIMIZE)
            
            # Solve subproblem 1 with updated lambda
            subproblem1.optimize()
            subproblem2.optimize()
    
            # Get the solutions from subproblems
            x1_values.append(x1.x)
            x2_values.append(x2.x)
    
            # Update the Lagrange multiplier using the subgradient method
            if -x1_values[-1]-x2_values[-1]+4 !=0:
                lambda_update = max(0,lambda_values[-1]+1/(a+b*iteration)*((-x1_values[-1]-x2_values[-1]+4)/abs(-x1_values[-1]-x2_values[-1]+4)))  
                lambda_values.append(lambda_update)
                if abs(lambda_values[-1]-lambda_values[-2])/abs(lambda_values[-2]) <= epsilon:
                    iteration = iterations+1
                    print("stop: convergence criterion achieved")
                else:    
                    iteration = iteration+1
            else: 
                iteration = iterations+1
                print("stop: update impossible divide by 0")
    
            # Calculate the final objective value
            obj_values.append(x1_values[-1]**2 + x2_values[-1]**2)
        
    return obj_values, x1_values, x2_values, lambda_values


lambda_0 = 10 # Initial value of Lagrange multipliers
iterations = 10000 # Number of iterations
epsilon = 0.001  # Sensitivity
a=1 # update parameters
b=0.1 # update parameters

obj_values, x1_values, x2_values, lambda_values = lagrangian_relaxation_decomposition(lambda_0,a,b,epsilon,iterations)
print(f"Number of iterations={len(x1_values)-1}")
print(f"Optimal solution: x1={x1_values[-1]}, x2={x2_values[-1]}, obj={obj_values[-1]}")


#%% Augmented lagrangian algorithm (ADMM)

def augmented_lagrangian_relaxation_decomposition(lambda_0,x1_0,x2_0,gamma,epsilon,iterations):

    # initialize Lagrange multipliers:
    lambda_values = [lambda_0]
    x1_values = [x1_0]
    x2_values = [x2_0]
    
    #initialize solutions of subproblems
    obj_values = [x1_0**2+x2_0**2]

    
    # Create the Gurobi models for the two subproblems
    subproblem1 = gp.Model("subproblem1")
    subproblem2 = gp.Model("subproblem2")

    # Variables for subproblem 1
    x1 = subproblem1.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1")
    
    # Variables for subproblem 2
    x2 = subproblem2.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2")

    # Update subproblems 1 and 2
    subproblem1.update()
    subproblem2.update()

    # initialize iteration
    iteration=0

    while iteration<=iterations:    

            # Set objective functions for subproblems
            subproblem1.setObjective(x1**2 - lambda_values[-1]*x1+gamma/2*(-x1-x2_values[-1]+4)**2, GRB.MINIMIZE)
            subproblem2.setObjective(x2**2 - lambda_values[-1]*x2+gamma/2*(-x1_values[-1]-x2+4)**2, GRB.MINIMIZE)
            
            # Solve subproblem 1 with updated lambda
            subproblem1.optimize()
            subproblem2.optimize()
    
            # Get the solutions from subproblems
            x1_values.append(x1.x)
            x2_values.append(x2.x)
    
            # Update the Lagrange multiplier using the subgradient method
            lambda_update = lambda_values[-1]+gamma*(-x1_values[-1]-x2_values[-1]+4)
            lambda_values.append(lambda_update)
            if abs(lambda_values[-1]-lambda_values[-2])/abs(lambda_values[-2]) <= epsilon:
                iteration = iterations+1
                print("stop: convergence criterion achieved")

            else: 
                iteration = iteration+1
    
            # Calculate the final objective value
            obj_values.append(x1_values[-1]**2 + x2_values[-1]**2)
        
    return obj_values, x1_values, x2_values, lambda_values


lambda_0 = 10 # Initial value of Lagrange multipliers
x1_0 = 0 # Initial value of x1
x2_0 = 0 # Initial value of x2
iterations = 10000 # Number of iterations
epsilon = 0.000001  # Sensitivity
gamma=1 # penalty parameter

obj_values, x1_values, x2_values, lambda_values = augmented_lagrangian_relaxation_decomposition(lambda_0,x1_0,x2_0,gamma,epsilon,iterations)
print(f"Number of iterations={len(x1_values)-1}")
print(f"Optimal solution: x1={x1_values[-1]}, x2={x2_values[-1]}, obj={obj_values[-1]}")
