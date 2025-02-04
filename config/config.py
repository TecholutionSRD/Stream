import yaml

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    print(config)