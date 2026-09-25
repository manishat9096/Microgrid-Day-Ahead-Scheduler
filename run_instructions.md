specifications.py contains the components sizes, rating, efficiency curves of converters
architechture.py - here, to build the microgrid architechture. add a new conversion stage using components.Stage --> Thenm build the components, buses, and add it to design = components.SystemSpec.
In a new python file, load the design from architechture, and then run modular_MG_model.build_and_solve() to run the optimisation model for 1 day.
Examples -
main.py runs the optimisation model for one day. 
old_case_study.py runs the optimisation for 20 random days over the year for two model AC and DC


Important flags:
1. under architechture.py > components.State contains the flag sos2 - (False by default) --> Setting it to True add the SOS2 binary constraints, so the model will take 2-3 mins per solve.
2. components.Sink contains the flag exact - (True by default) --> True means load is constant, False means Load can be controlled (shifting and shedding).
3. components.Source contains the flag exact - (True by default) --> True means No curtailment, False means can be curtailed.




### Notes
1. add degradatation cost to battery cycles
2. add load shedding cost if load is not exact. 