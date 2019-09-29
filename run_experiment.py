import subprocess

datasets = ['fsai', 'synthetic', 'statics', 'assist2009', 'assist2015']
memory_sizes = [1, 2, 5, 10, 20, 50, 100]
state_dims = [10, 50, 100, 200]

model_profiles = []
for dataset in datasets:
    for memory_size in memory_sizes:
        for state_dim in state_dims:
            model_profiles.append(
                {
                    'dataset': dataset,
                    'memory_size': memory_size,
                    'value_memory_state_dim': state_dim,
                    'key_memory_state_dim': state_dim
                }
            )

for model_profile in model_profiles:
    # base
    command = ["python", "main.py"]

    # add dataset
    command.append("--dataset")
    command.append("{}".format(model_profile['dataset']))

    # add memory_size
    command.append("--memory_size")
    command.append("{}".format(model_profile['memory_size']))

    # add value_memory_state_dim
    command.append("--value_memory_state_dim")
    command.append("{}".format(model_profile['value_memory_state_dim']))

    # add memokey_memory_state_dimry_size
    command.append("--key_memory_state_dim")
    command.append("{}".format(model_profile['key_memory_state_dim']))

    # run command
    print("run:", command)
    subprocess.run(command)