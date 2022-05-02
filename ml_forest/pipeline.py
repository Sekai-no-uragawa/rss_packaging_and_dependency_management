from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import ast


def create_pipeline(
    clf_type: str,
    use_scaler: bool,
    random_state: int,
    model_param
):
    mapping_dict = {
        'ExtraTreesClassifier' : ExtraTreesClassifier,
        'DecisionTreeClassifier' : DecisionTreeClassifier,
        'RandomForestClassifier' : RandomForestClassifier,
    }


    params = ast.literal_eval('{' + model_param + '}')

    pipeline_steps = []
    if use_scaler:
         pipeline_steps.append(("scaler", StandardScaler()))
    
    clf = mapping_dict[clf_type]
    pipeline_steps.append(
        (
            "clf_type",
            clf(random_state=random_state, **params),
        )
    )
    return Pipeline(steps=pipeline_steps)