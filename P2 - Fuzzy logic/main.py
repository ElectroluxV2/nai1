"""
==========================================
Fuzzy Control Systems: The Lunar Lander
==========================================

This code demonstrates a fuzzy control system for simulating the landing of a lunar lander. The control system uses the skfuzzy library to model the landing process. The purpose of this simulation is to determine the required thrust to safely land the lunar lander based on its current altitude, acceleration, and fuel levels.

Problem Formulation
-------------------
The problem is formulated as follows:

- Antecedents (Inputs):
    - `altitude`
        - Universe (crisp value range): Represents the altitude of the lunar lander (ranging from 0 to 10).
        - Fuzzy set (fuzzy value range): good, average, poor
    - `acceleration`
        - Universe: Represents the acceleration of the lunar lander (ranging from 0 to 10).
        - Fuzzy set: good, average, poor
    - `fuel`
        - Universe: Represents the remaining fuel of the lunar lander (ranging from 0 to 10).
        - Fuzzy set: good, average, poor

- Consequents (Outputs):
    - `thrust`
        - Universe: Represents the thrust required for a successful landing (ranging from 0 to 25).
        - Fuzzy set: low, medium, high
Usage
-----
To use this control system, you can input specific values for `altitude`, `acceleration`, and `fuel`. The system will compute and suggest the required `thrust` to safely land the lunar lander based on the fuzzy rules.

Dependencies
------------
Make sure to install the required libraries using the following commands:
- pip install scikit-fuzzy
- pip install matplotlib

Control System Creation and Simulation
---------------------------------------
The control system is created using the defined rules, and a `ControlSystemSimulation` object is employed to simulate the lunar lander's descent for specific input values. The resulting `thrust` is computed and can be visualized.

Note
----
The fuzzy logic-based control system can handle imprecise and uncertain conditions during the landing process, making it suitable for real-time decision-making.

Authors: [Mateusz Budzisz, Aleksander Guzik]

"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
altitude = ctrl.Antecedent(np.arange(0, 11, 1), 'altitude')
acceleration = ctrl.Antecedent(np.arange(0, 11, 1), 'acceleration')
fuel = ctrl.Antecedent(np.arange(0, 11, 1), 'fuel')

thrust = ctrl.Consequent(np.arange(0, 26, 1), 'thrust')

# Auto-membership function population is possible with .automf(3, 5, or 7)
altitude.automf(3)
acceleration.automf(3)
fuel.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
thrust['low'] = fuzz.trimf(thrust.universe, [0, 0, 13])
thrust['medium'] = fuzz.trimf(thrust.universe, [0, 13, 25])
thrust['high'] = fuzz.trimf(thrust.universe, [13, 25, 25])

"""
To help understand what the membership looks like, use the ``view`` methods.
"""

# You can see how these look with .view()
altitude['average'].view()
"""
.. image:: PLOT2RST.current_figure
"""
acceleration.view()
"""
.. image:: PLOT2RST.current_figure
"""
fuel.view()
"""
.. image:: PLOT2RST.current_figure
"""
thrust.view()
"""
.. image:: PLOT2RST.current_figure

"""
"""
Fuzzy Rules
-----------
The fuzzy control system is governed by a set of fuzzy rules:

1. If the `altitude` is high and the `acceleration` is high, then the required `thrust` is high.
2. If the `altitude` is low, then the required `thrust` is low.
3. If the `fuel` is high or the `acceleration` is high, then the required `thrust` is medium.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""

rule1 = ctrl.Rule(altitude['good'], thrust['low'])
rule2 = ctrl.Rule(altitude['average'], thrust['medium'])
rule3 = ctrl.Rule(altitude['poor'], thrust['high'])

rule4 = ctrl.Rule(fuel['good'] | fuel['average'], thrust['high'])
rule5 = ctrl.Rule(fuel['poor'], thrust['low'])

rule6 = ctrl.Rule(acceleration['good'], thrust['high'])
rule7 = ctrl.Rule(acceleration['average'] | acceleration['poor'], thrust['medium'])

"""
Control System Creation and Simulation
---------------------------------------

Now that we have our rules defined, we can simply create a control system
via:
"""

thrust_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

"""
In order to simulate this control system, we will create a ``ControlSystemSimulation``. 
Think of this object as representing our controller applied to a specific set of circumstances. 
For our lunar lander simulation, this might be simulating the landing of a lunar lander with certain input values. 
We would create another ``ControlSystemSimulation`` when we want to apply our control system to a different scenario with different input values.

"""

thrust_simulation = ctrl.ControlSystemSimulation(thrust_ctrl)

"""
Now we set the input values for the simulation as follows:
"""
altitude_input = float(input("Enter altitude: "))
acceleration_input = float(input("Enter acceleration: "))
fuel_input = float(input("Enter fuel quantity: "))

thrust_simulation.input['altitude'] = altitude_input
thrust_simulation.input['acceleration'] = acceleration_input
thrust_simulation.input['fuel'] = fuel_input

"""
Once computed, we can view the result as well as visualize it.
"""
thrust_simulation.compute()
print(thrust_simulation.output['thrust'])
thrust.view(sim=thrust_simulation)
plt.show()

"""
Final thoughts
--------------

The power of fuzzy systems is allowing complicated, intuitive behavior based
on a sparse system of rules with minimal overhead. Note our membership
function universes were coarse, only defined at the integers, but
``fuzz.interp_membership`` allowed the effective resolution to increase on
demand. This system can respond to arbitrarily small changes in inputs,
and the processing burden is minimal.
"""
