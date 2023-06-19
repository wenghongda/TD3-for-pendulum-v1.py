import rl
from energy_model import Fuel_cell,Electrolyzer,PV,Battery,Hydrogen_tank

state = env.reset()
for t in range(24 * 4):
    action = env.choose_action(state)
    next_state,reward = env.step(action)
    state = next_state