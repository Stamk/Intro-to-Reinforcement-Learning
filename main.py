from functions.utils import get_config_file,get_exp_dir,load_input_data,create_envs_agents_combinations,run,plot_performance
from shutil import copy

if __name__ == '__main__':
    config_file = get_config_file()

    exp_path = get_exp_dir()

    copy(config_file, exp_path)

    data = load_input_data(config_file)

    envs_agents = create_envs_agents_combinations(data)

    run(envs_agents, exp_path)

    plot_performance(envs_agents, exp_path)
