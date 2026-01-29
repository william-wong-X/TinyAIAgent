import argparse

from config.config import load_config
from app.agent import Agent
from ui.cli import chat_cli

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str, default="./config/config.yaml", help="Config path")

    args = parser.parse_args()

    config = load_config(args.config)

    agent = Agent(config)

    chat_cli(config, agent)

if __name__ == "__main__":
    main()