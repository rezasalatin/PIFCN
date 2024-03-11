import yaml

class ConfigLoader:
    def __init__(self, yaml_path, section=None, key=None):
        # Load and store the entire configuration
        with open(yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # If both section and key are specified, return a specific value
        if section and key:
            self.value = self.config[section].get(key)
        # If only section is specified, return the whole section
        elif section:
            self.value = self.config.get(section)
        # Otherwise, store the entire config
        else:
            self.value = self.config

    def get_value(self):
        # Return the stored value (specific value, section, or entire config)
        return self.value