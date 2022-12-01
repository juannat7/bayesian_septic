import json

from . import bayes_models as BM


def train_bayes_model(**kwargs):
    model_name = kwargs['model_name']
    mod = kwargs['mod']
    
    with open(f"configs/{model_name}.json", "r") as f:
        pars = json.load(f)
        pars["name"] = model_name
    
    aval_model_name = ["HierarchicalBayesian", "MultilevelHierarchicalBayesian"]
    assert pars['model_type'] in aval_model_name, f"model_type parameter unrecognized, must be one of the {aval_model_name}. If new model type is added, please update the permission list via ./src/base_models.py SettingParser __init__"
    
    Dock = getattr(BM, pars['model_type'])(pars)
    Dock.run_scripts(mod)
    return Dock
