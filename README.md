# CoilSignalSim
Simple simulation of coil-based magnetometer

(Preliminary)
by Qing Lin @ 2022-01-25


*******
 Code Usage
*******

python3 CoilSignalSim.py <Mode> <config file> <output filename> <(optional) Number of simulated events>

Mode=0: Field simulation mode. Generate pkl file containing fine-element simulation results.

Mode=1: Signal simulation model. Use the pkl file generated in Mode0 to calculate induced signal.



example:

First to generate a monopole field:

python3 CoilSignalSim.py 0 configs/generator/monopole.yaml outputs/test_monopole_field.pkl


Then use this field to calculate induction voltage:

python3 CoilSignalSim.py 1 configs/induction_calculator/test_monopole1.yaml outputs/test_monopole_signal.pkl


Check out the yaml files for more details.
