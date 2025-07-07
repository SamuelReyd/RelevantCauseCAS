import pickle, sys

from flocking import *
from cause_identification import FlockVarSCM, FlockParamSCM, FlockBooleanSCM, GranularVarSCM, GranularParamSCM, Metrics
from numpy import pi

## Fixes for solving pickle errors
sys.modules['__main__'].FlockVarSCM = FlockVarSCM
sys.modules['__main__'].FlockParamSCM = FlockParamSCM
sys.modules['__main__'].FlockBooleanSCM = FlockBooleanSCM
sys.modules['__main__'].GranularVarSCM = GranularVarSCM
sys.modules['__main__'].GranularParamSCM = GranularParamSCM

## Result plotting
def load_scms(scenario, prefix="../"):
    try:
        with open(
            f"{prefix}results/{scenario.value}.pkl", 
            "rb"
        ) as file:
            scms = pickle.load(file)
    except (FileNotFoundError, AttributeError) as e:
        print("Unpickling failed", e)
        scms = []
        for cls in (FlockVarSCM, FlockParamSCM, FlockBooleanSCM, GranularVarSCM, GranularParamSCM):
            actual_run, hp = getters[scenario]()
            flock_mapping = get_flock_mapping(actual_run, hp)
            scm = cls(actual_run, hp, flock_mapping=flock_mapping)
            scms.append(scm)
    return scms

def do_causal_analysis(scms, scenario, prefix="../"):
    for scm in scms:
        if "causes" not in scm.__dict__:
            print(f"Searching causes for SCM: '{scm.__class__.__name__}'")
            scm.find_causes(sim_verbose=1, max_step=3, beam_size=10)
            print(f" -> Found {len(scm.causes)} causes")
        else:
            print(f"SCM: '{scm.__class__.__name__}' -> {len(scm.causes)} causes")
        if "scores" not in scm.__dict__:
            scm.compute_scores()
        with open(
            f"{prefix}results/{scenario.value}.pkl", 
            "wb"
        ) as file:
            pickle.dump(scms, file)

def analyze_causes(causes, variables):
    entities, timesteps, var_names = [], [], []
    for cause in causes:
        for dim in cause[3]:
            causal_var = variables[dim]
            entity_type, i, v, t = break_var(causal_var)
            entities.append(i)
            timesteps.append(t)
            var_names.append(v)
    return entities, timesteps, var_names

def get_causes_scenario(scenario):
    scms = load_scms(scenario, prefix="../")
    causes, scores, scm_refs = get_causes(scms)
    return causes, scores, scm_refs, scms

def get_causes(scms):
    causes = []
    metrics = [m.value for m in Metrics]
    scores = []
    scm_refs = []
    for i, scm in enumerate(scms):
        if "causes" not in scm.__dict__:
            continue
        causes += scm.causes
        scm.compute_scores()
        scores.append([scm.scores[metric] for metric in metrics])
        scm_refs += [i] * len(scm.causes)
    scores = np.hstack(scores)
    return causes, scores, scm_refs

# Scenarii
def get_scenario_1():
    hp = {
        **base_hp,
        **{
            "width": 200,
            "height": 200,
            "n_steps": 70,
            "do_padding": True,
            "obstacle_x": 100,
            "obstacle_y": 100,
            "padding": 20,
            "max_speed": 3,
            "view_radius": 18,
            "minimum_separation": 9,
            "alignment_factor": 5  * pi / 180,
            "cohesion_factor": 6 * pi / 180,
            "separation_factor": 12  * pi / 180,
            "avoid_factor": 10  * pi / 180,
        }
    }
    # Initialize boids
    boids = init_boids_uniform(hp, min_y=(2*hp["height"])//3)
    history = make_run_flocking(0, boids, hp)
    full_actual_run = np.stack(history)
    return full_actual_run, hp

def get_scenario_2():
    hp = {
        **base_hp,
        **{
            "init_width": 50,
            "init_height": 50,
            "n_steps": 50,
            "do_padding": True,
            "width": 200,
            "height": 200,
            "obstacle_x": 100,
            "obstacle_y": 30,
            "obstacle_radius": 25,
            "padding": 10,
            "max_speed": 3,
            "view_radius": 18,
            "minimum_separation": 7,
            "alignment_factor": 8  * pi / 180,
            "cohesion_factor": 8 * pi / 180,
            "separation_factor": 9  * pi / 180,
            "avoid_factor": 8  * pi / 180,
        }
    }
    # Initialize boids
    boids = init_boids_bottom(hp)
    history = make_run_flocking(0, boids, hp)
    full_actual_run = np.stack(history)
    return full_actual_run, hp

def get_scenario_3():
    hp = {
        **base_hp,
        **{
            "init_width": 30,
            "init_height": 30,
            "n_steps": 35,
            "do_padding": True,
            "width": 200,
            "height": 200,
            "obstacle_x": 100,
            "obstacle_y": 30,
            "obstacle_radius": 25,
            "padding": 10,
            "max_speed": 4,
            "view_radius": 15,
            "minimum_separation": 7,
            "alignment_factor": 8  * pi / 180,
            "cohesion_factor": 8 * pi / 180,
            "separation_factor": 9  * pi / 180,
            "avoid_factor": 6  * pi / 180,
        }
    }
    # Initialize boids
    boids = init_boids_clash(hp)
    history = make_run_flocking(0, boids, hp)
    full_actual_run = np.stack(history)
    return full_actual_run, hp

getters = {
    Scenarii.FREE: get_scenario_1,
    Scenarii.ONE_FLOCK: get_scenario_2,
    Scenarii.FUSION: get_scenario_3
}

if __name__ == "__main__":
    for scenario in Scenarii:
        scms = load_scms(scenario, prefix="")
        do_causal_analysis(scms, scenario, prefix="")
