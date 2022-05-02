from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def create_pipeline(
    clf_type: str,
    use_scaler: bool,
    random_state: int,
):
    mapping_dict = {
        'ExtraTreesClassifier' : ExtraTreesClassifier,
        'DecisionTreeClassifier' : DecisionTreeClassifier,
        'RandomForestClassifier' : RandomForestClassifier,
    }

    pipeline_steps = []
    if use_scaler:
         pipeline_steps.append(("scaler", StandardScaler()))
    
    clf = mapping_dict[clf_type]
    pipeline_steps.append(
        (
            "clf_type",
            clf(random_state=random_state),
        )
    )
    return Pipeline(steps=pipeline_steps)