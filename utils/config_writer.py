import argparse
import json

def update_config(config_content, path, value):
    keys = path.split('/')
    temp = config_content
    for key in keys[:-1]:
        temp = temp.setdefault(key, {})
    temp[keys[-1]] = json.loads(value)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Update JSON config file")
    parser.add_argument("-c", "--config", help="Path to config JSON file", required=True)
    parser.add_argument("-o", "--output", help="Path to output JSON file", required=True)
    parser.add_argument("params", nargs='+',
                        help="List of parameters in format cat1/param1/val1=2")
    args = parser.parse_args()

    # Load config JSON file
    with open(args.config, 'r') as f:
        config_content = json.load(f)

    # Update config with parameters
    for param in args.params:
        path, value = param.split('=')
        update_config(config_content, path, value)

    # Save updated config to output JSON file
    with open(args.output, 'w') as f:
        json.dump(config_content, f, indent=4)

if __name__ == "__main__":
    main()
