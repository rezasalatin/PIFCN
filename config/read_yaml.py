import yaml

class ConfigLoader:
    def __init__(self, yaml_path, section=None, sub_1=None, sub_2=None):
        # Load and store the entire configuration
        with open(yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # If section, sub1 and sub2 are specified, return a specific value
        if section and sub_1 and sub_2:
            self.value = self.config[section][sub_1].get(sub_2)
        # If both section and key are specified, return a specific value
        elif section and sub_1:
            self.value = self.config[section].get(sub_1)
        # If only section is specified, return the whole section
        elif section:
            self.value = self.config.get(section)
        # Otherwise, store the entire config
        else:
            self.value = self.config

    def get_value(self):
        # Return the stored value (specific value, section, or entire config)
        return self.value