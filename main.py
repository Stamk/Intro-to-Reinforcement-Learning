import json
import os
from datetime import datetime


from functions.utils import make_envs, make_agents

if __name__ == '__main__':
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)

    config_file = 'data/storage_agent.json'

    with open(config_file, 'rb') as f:
        my_dict = json.load(f)

    envs = make_envs(my_dict)
    envs_agents = dict()
    for env in envs:
        envs_agents[env] = make_agents(env, my_dict)

    for env, agents in envs_agents.items():
        for agent in agents:
            agent.train()
            agent.save()
        # agent.evaluate()

    print("I m here")
