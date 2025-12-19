# Design of Fuzzy Inference System (FIS)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temp = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
fan = ctrl.Consequent(np.arange(0, 11, 1), 'fan')

temp['cold'] = fuzz.trimf(temp.universe, [0, 0, 15])
temp['warm'] = fuzz.trimf(temp.universe, [10, 20, 30])
temp['hot'] = fuzz.trimf(temp.universe, [25, 40, 40])

fan['low'] = fuzz.trimf(fan.universe, [0, 0, 15])
fan['med'] = fuzz.trimf(fan.universe, [3, 5, 7])
fan['high'] = fuzz.trimf(fan.universe, [6, 10, 10])

rule1 = ctrl.Rule(temp['cold'], fan['low'])
rule2 = ctrl.Rule(temp['warm'], fan['med'])
rule3 = ctrl.Rule(temp['hot'], fan['high'])

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

if __name__ == '__main__':
    fan_sim.input['temperature'] = 20
    fan_sim.compute()
    print(f'Fan speed (crisp): {fan_sim.output['fan']}')

# OUTPUT

# Fan speed (crisp): 5.000000000000001
