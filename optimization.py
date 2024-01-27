import numpy as np
from sklearn.linear_model import LinearRegression

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

MAPPING = {"Natural Rubber RSS": "Polymer",
           "Natural Rubber TSR 10 60 CV": "Polymer",
           "Polybutadiene (NR) - Cisamer 1220": "Polymer",
           "Polybutadiene High Cys": "Polymer",
           "SBR": "Polymer",
           "Peptizer": "Process aid",
           "N660": "Black Filler",
           "N550": "Black Filler",
           "N339": "Black Filler",
           "N330": "Black Filler",
           "N326": "Black Filler",
           "N220": "Black Filler",
           "N234": "Black Filler",
           "N115": "Black Filler",
           "N121": "Black Filler",
           "Silica": "White Filler",
           "MR material": "Nano Filler",
           "Non Reinforcing Filler": "Non R Filler",
           "Silane": "Surface Modifier",
           "Cabot Endure D63": "Black Filler",
           "MWCNT from MRO Gen II": "Nano Filler",
           "Pasticiser": "Plasticiser",
           "Aromatic Oil": "Plasticiser",
           "Naftenic Oil": "Plasticiser",
           "Paraffinic Process Oil": "Plasticiser",
           "Pine Tar": "Plasticiser",
           "C5 Aliphatic Hydrocarbon": "Plasticiser",
           "Process aid": "Process aid",
           "ZnO": "Catalizer",
           "Stearic Acid": "Catalizer",
           "6 ppd": "Antidegradant",
            "TMQ": "Antidegradant",
            "Micro cristaline wax": "Antidegradant",
            "TBBS": "Accelerators",
            "MBS": "Accelerators",
            "CBS": "Accelerators",
            "DCBS": "Accelerators",
            "MBTS": "Accelerators",
            "MBT": "Accelerators",
            "DPG": "Accelerators",
            "TBzTD": "Accelerators",
            "TMTD": "Accelerators",
            "Sulfur": "Curing Agent",
            "Perkalink 900": "Anti reversion Agent",
            "PVI": "Retarder",
            "Sum Curing System": "Others",
            "A/S Ratio": "Others",
            "Rubbers Sum": "Others",
}

def generate_relations(df, input_features, output_features):
    relations = {}
    r2_scores = {}
    for target in input_features:
        X = df[output_features]
        y = df[target]
        
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        print(f"> {target}: R2 score =", lr_model.score(X, y))
        r2_scores[target] = lr_model.score(X, y)
        relations[target]  = [lr_model.coef_, lr_model.intercept_]
    
    return relations, r2_scores


def define_problem(df, input_features, output_features, lower_bounds, upper_bounds, sample_input, mapping=MAPPING):
    relations, r2_scores = generate_relations(df, input_features, output_features)

    polymer_features = []
    for index, feature in enumerate(output_features):
        if mapping[feature].lower() == "polymer":
            polymer_features.append(index)

    additional_constraints = 0
    if polymer_features:
        additional_constraints += 1

    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(output_features),
                            n_obj=1,
                            n_ieq_constr=0,
                            n_eq_constr=len(relations)+additional_constraints,
                            xl=np.array(lower_bounds),
                            xu=np.array(upper_bounds))

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = 0
            for x_term in x:
                f1 -= x_term
            
            H = []
            index = 0
            for target in relations:
                h = 0
                for idx, x_term in enumerate(x):
                    h += x_term * relations[target][0][idx]
                h -= (sample_input[index] - relations[target][1])
                index += 1
                H.append(h)
            

            if polymer_features:
                h = 0 # h = x[0] + x[1] - 100, sum to 100
                for index in polymer_features:
                    h += x[index]
                h -= 100
                H.append(h)
            
            out["F"] = [f1]
            out["H"] = H

    problem = MyProblem()
    return (problem, r2_scores)

def minimize_problem(problem, n_gen=100):
    algorithm = NSGA2()
    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   verbose=True,
                   save_history=True)
    pop = res.pop

    X = []
    for i, x in enumerate(pop.get("X")[0]):
        X.append(x)
    
    F = []
    for f in pop.get("F"):
        F.append(f[0])

    return (X, F)