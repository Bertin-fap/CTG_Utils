__all__ = ['GLOBAL']

def _config_ctg():
    # Reads the congiguration file

    # Standard library imports
    import os.path
    from pathlib import Path

    # 3rd party imports
    import yaml
    
    # Reads the default PVcharacterization.yaml config file
    path_config_file = Path(__file__).parent / Path('CTG_RefFiles/CTG.yaml')
    with open(path_config_file) as file:
        global_ = yaml.safe_load(file)
        
       
    return global_

GLOBAL = _config_ctg()