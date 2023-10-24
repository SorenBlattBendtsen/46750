## 46750 - Assigment 1
Optimization in Modern Power Systems 
Julia Fabris, Markus Hvid Monin & Søren Blatt Bendtsen

---------------------------------------------
### Function Library Assignment 1
This Py file creates different functions that are used across all the following Jupyter Notebooks. The functions are:
* Read Data
This function reads and processes all the input data.
* Mapping Dictionaries
This function maps the node location of each generator and wind farm in the power system.

---------------------------------------------
### Task 0, Wind data processing
This Jupyter Notebook processes the data from the wind farms.
* First it plots all the scenarios for the 6 different windfarms together with the average of each scenario across the available 37 hours
* It then saves and plots the average values from hour 13 to 37 which will be used for the optimization tasks

---------------------------------------------
### Task 1
This Jupyter Notebook solves Task 1 for the Single Time Step Optimal Power Flow.
* First it creates and solves the optimization problem
* Then it stores all the results of the model
* At last it creates visualizations of all the relevant results

---------------------------------------------
### Task 2
This Jupyter Notebook solves Task 2 for the Decomposition using Augmented Lagrangian Relaxation
* First it creates hyper parameters, sets up the 24 subproblems solves these over 100 iterations while it tries to converge
* Then it stores all the results of the model
* At last it creates visualizations of the relevant results

---------------------------------------------
### Task 3
This Jupyter Notebook solves Task 3 for the Multi Time Step Optimal Power Flow with three Battery Energy Storage Systems (BESS) introduced.
* First it creates a function that is used to solve the optimization problem and store and plot simple results.
* This function can either take BESS = False or BESS = True as input in order to compare results with and without BESS in the system.
* At last it creates visualizations of all the relevant results

---------------------------------------------
### Task 4
This Jupyter Notebook solves Task 4 for the optimal location of BESS
* First it creates a function that is used to solve the optimization problem and store and plot simple results.
* At last it creates visualizations of all the relevant results
