import argparse

from config.config import AppConfig, load_config
from app.agent import agent_run

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str, default="./config/config.yaml", help="Config path")

    args = parser.parse_args()

    config = load_config(args.config)

    agent_run(config)

if __name__ == "__main__":
    main()