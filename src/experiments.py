from flocking import Scenarii, base_hp
from cause_identification import FlockVarSCM, FlockParamSCM, FlockBooleanSCM, GranularVarSCM, GranularParamSCM
from numpy import pi

## Result plotting
def load_scms(scenario, prefix="../"):
    try:
        with open(
            f"{prefix}results_multi_scale/{scenario.value}.pkl", 
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
            f"{prefix}results_multi_scale/{scenario.value}.pkl", 
            "wb"
        ) as file:
            pickle.dump(scms, file)

def illustrate_scms(scms):
    axes = get_3_axes()
    for ax, scm in zip(axes,scms):
        scm.init_structure()
        var = np.random.choice(scm.variables)
        ent, i, _, t = break_var(var)
        boid_ids = get_boid_ids_once(var, scm.flock_mapping)
        show_boids(
            scm.actual_run[t], scm.hp, 
            highlight_ids=boid_ids,ax=ax, title=f'{scm.label}\n{var} (/{len(scm.variables)})'
        )
    plt.tight_layout()
    plt.show()


def show_opposite_scores(scm, metric):
    print(f"SCM: {scm.label}")
    print(f"Metric: {metric}")
    cause_labels = ["Worst cause", "Best cause"]
    scores = scm.scores[metric]
    videos = []
    titles = []
    for idx in (0,-1):
        i = np.argsort(scores)[idx]
        videos.append(render_simulation(scm.causes[i][-1], scm.hp))
        titles.append(
            show_cause(scm.causes[i], 
                       scm.variables, 
                       scm.instance, 
                       score=scores[i], 
                       label=f"{cause_labels[idx]}:", 
                       show=False,
                       force_line_break=True
                      )
        )
        
    return HTML(make_animation(videos, titles=titles))

def show_scenarii(scenarii):
    videos = []
    titles = []
    for label, (run, hp) in scenarii.items():
        titles.append(label)
        videos.append(render_simulation(run, hp))
    return HTML(make_animation(videos, titles))

def show_cause(cause, variables, instance, score=None, label="Cause: ", 
               show=True, force_line_break=False):
    # Label
    s = f"{label}"
    # Line break for long causes
    if (len(cause[3])>1 or force_line_break) and label:
        s += "\n"
    # Repr cause
    s += '\n'.join([f'{variables[dim]}={instance[dim]:.2f}' for dim in cause[3]])
    # Line break for long causes
    if score is not None:
        if len(cause[3])>1 or force_line_break:
            s += '\n'
        s += f" -> {score=:.2f}"
    if show:
        print(s) 
    else:
        return s

def show_causation_results(causes, all_scores, variables, instance, actual_run, flock_mapping, hp, granularity):
    print(f"Found {len(causes)} causes.")
    entities, timesteps, var_names = analyze_causes(causes, variables)
    plot_causes_analysis(entities, timesteps, var_names)
    plot_frequent_entities(entities, actual_run[0], flock_mapping, hp, granularity)
    for name, scores in all_scores.items():
        best_score_id = np.argmax(scores)
        show_cause(causes[best_score_id], variables, instance, scores[best_score_id], f"{name}: ")

def compute_results_scaling(factors, actual_run, freq_data, hp, verbose=1):
    scms = {}
    # Granular SCMs
    for cls in (GranularVarSCM, GranularParamSCM):
        scm = cls(actual_run, hp, freq_data=freq_data)
        scm.find_causes()
        scm.compute_scores()
        scms[scm.get_label()] = scm
        if verbose:
            print(f"{scm.get_label()} -> {len(scm.variables)=}, {len(scm.causes)=}")
        
    # Coarse SCMs
    for cls in (FlockVarSCM, FlockParamSCM):
        for factor in factors:
            eps = factor * hp["view_radius"]
            flock_mapping = get_flock_mapping(actual_run, hp, eps)
            scm = cls(actual_run, hp, 
                            freq_data=freq_data, 
                            flock_mapping=flock_mapping, eps=eps)
            scm.find_causes()
            scm.compute_scores()
            scms[scm.get_label()+f"({factor=})"] = scm
            if verbose:
                print(f"{scm.get_label()+f"({factor=})"} -> {len(scm.variables)=}, {len(scm.causes)=}")
    return scms

def get_best_causes(scms):
    data = []
    for label, scm in scms.items():
        scm_data = {}
        for metric, scores in scm.scores.items():
            scm_data[metric] = max(scores)
        data.append(scm_data)
    return data

def plot_causes_analysis(entities, timesteps, var_names):
    axes = get_3_axes()
    plot_entities(entities, axes[0])
    plot_timesteps(timesteps, axes[1])
    plot_variables(var_names, axes[2])
    axes[0].set_ylabel("#cause")
    plt.tight_layout()
    plt.show()

def plot_frequent_entities(entities, boids, flock_mapping, hp, granularity):
    x, y = zip(*Counter(entities).items())
    ids = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
    axes = get_3_axes()
    for i, ax in zip(ids, axes.flatten()):
        ax.set_title(f"Entity {x[i]} (x{y[i]})")
        if granularity == Granularity.FLOCK:
            f_ent = flock_mapping[x[i]]
        else:
            f_ent = [x[i]]
        show_boids(boids, hp, f_ent, ax)
    plt.tight_layout()
    plt.show()

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

def plot_entities(entitities, ax):
    x, y = zip(*Counter(entitities).items())
    ids = sorted(range(len(y)), key=lambda i: -y[i])
    x = [str(x[i]) for i in ids]
    y = [y[i] for i in ids]
    ax.bar(x,y)
    ax.set_xlabel("Entitities")

def plot_timesteps(timesteps, ax):
    x, y = zip(*Counter(timesteps).items())
    ids = sorted(range(len(x)), key=lambda i: int(x[i]))
    x = [str(x[i]) for i in ids]
    y = [y[i] for i in ids]
    ax.bar(x,y)
    ax.set_xlabel("Timesteps")

def plot_variables(var_names, ax):
    x, y = zip(*Counter(var_names).items())
    ax.tick_params('x', labelrotation=45)
    x_labels = []
    for elt in x:
        x_label = "_".join([s[:3] for s in elt.split("_")])
        x_labels.append(x_label)
    ax.bar(x_labels,y)
    ax.set_xlabel("Variables")

def plot_all_scores(scms):
    metrics = ("CF score", "Strong Necessity", "Rencency", "Abnormality", "Cost")
    _, axes = plt.subplots(2,3,figsize=(18,9), sharex=True, sharey=True)
    
    for ax, metric in zip(axes.flatten(), metrics):
        ax.set_title(metric) 
        for label, scm in scms.items(): 
            if metric in scm.scores:
                ax.plot(sorted(scm.scores[metric]), label=label)
        ax.legend(loc="upper left")
    for ax in axes.flatten()[len(metrics):]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("../results_multi_scale/scaling-scores.pdf")
    plt.show()


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
    history = make_run_flocking(boids, hp)
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
    history = make_run_flocking(boids, hp)
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