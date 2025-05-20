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

def show_causation_results(causes, all_scores, variables, instance, actual_run, flock_mapping, hp, granularity):
    print(f"Found {len(causes)} causes.")
    entities, timesteps, var_names = analyze_causes(causes, variables)
    plot_causes_analysis(entities, timesteps, var_names)
    plot_frequent_entities(entities, actual_run[0], flock_mapping, hp, granularity)
    for name, scores in all_scores.items():
        best_score_id = np.argmax(scores)
        show_cause(causes[best_score_id], variables, instance, scores[best_score_id], f"{name}: ")

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
    plt.savefig("../results/scaling-scores.pdf")
    plt.show()

def render_cause(cause, variables, instance, flock_mapping, actual_run, hp):
    _, axes = plt.subplots(1,len(cause[0]), squeeze=False)
    axes = axes[0]
    for ax, (dim, cf_value) in zip(axes, cause[0]):
        causal_var = variables[dim]
        actual_value = instance[dim]
        cf_info = f", cf={cf_value:.2f}" if actual_value != cf_value else ""
        ent_type, ent_id, obs_var_label, t = break_var(causal_var)
        causal_label = f"{ent_type}={ent_id}, t={t}\n{obs_var_label}={actual_value:.2f}"
        title = causal_label + cf_info
        # ax.set_title(title)
        boid_ids = get_boid_ids_once(causal_var, flock_mapping)
        show_boids(actual_run[t], hp, highlight_ids=boid_ids, ax=ax, title=title)
    plt.tight_layout()
    plt.show()

def plot_score_distibution(scores, metrics, cost_bin_counts, cost_bin_edges):
    axes = get_3_axes(sharey=True)
    axes[0].set_yscale('log')
    plot_cost(cost_bin_edges, cost_bin_counts, axes[0])
    axes[0].tick_params('x', rotation=45)
    
    for i in (1,2):
        counter = Counter(scores[i])
        X, Y = zip(*counter.items())
        ids = np.argsort(X)
        axes[i].bar([str(round(X[i])) for i in ids], [Y[i] for i in ids])
            
    for i in range(3):
        axes[i].set_title(metrics[i])
    plt.suptitle("Metric distribution over the 5 SCMs")
    plt.tight_layout()
    plt.show()

def plot_cost(bin_edges, bin_counts, ax):
    ax.bar([f'{elt:.0%}' for elt in bin_edges], bin_counts)

def plot_score_2D(scores, metrics):
    _, axes = plt.subplots(2,2)
    for i in range(3):
        for j in range(i+1, 3):
            axes[j-1,i-1].scatter(scores[i], scores[j], alpha=.1)
            axes[j-1,i-1].set_xlabel(metrics[i])
            axes[j-1,i-1].set_ylabel(metrics[j])
    axes[0,0].set_axis_off()
    plt.suptitle("Metric distribution 2D")
    plt.tight_layout()
    plt.show()

def show_filtered_causes(causes, scores, scm_refs, scms, ids):
    # print(f"Number of cause considered {len(ids)} ({len(ids)/len(causes):.2e})")
    for i in ids:
        score = scores[:,i]
        scm = scms[scm_refs[i]]
        cause = causes[i]
        for m, s in zip(metrics, score):
            if m == Metrics.COST.value:
                print(f"{m}={s:.0%}", end=" ")
            else:
                print(f"{m}={s:.0f}", end=" ")
        print()
        render_cause(cause, scm.variables, scm.instance, scm.flock_mapping, scm.actual_run, scm.hp)


def show_priority(causes, scores, s, scms, scm_refs, priority, n=5):
    ids = sort_causes_priority(s, priority)[:n]
    for i in ids:
        cause = causes[i]
        scm = scms[scm_refs[i]]
        show_cause(cause, scm.variables, scm.instance, 
                   score=scores[priority,i].round(2).tolist(),
                   pred_sep=" & ", force_line_break=False, label="")

def show_results(scenario, show_sim=False, show_distr=False, 
                 show_distr_2D=False, show_priorities=[],
                n=5, show_priorities_graphic=False):
    scms = load_scms(scenario)
    causes, scores, scm_refs = get_causes(scms)
    s, bin_edges, bin_counts = digitalize_cost(scores)
    if show_sim: show_simulation(scms[0].actual_run, scms[0].hp)
    if show_distr: plot_score_distibution(s, metrics, bin_counts, bin_edges)
    if show_distr_2D: plot_score_2D(s, metrics)
    for priority in show_priorities:
        print(f"Priotities: {' '.join([list(Metrics)[i].value for i in priority])}")
        show_priority(causes, scores, s, scms, scm_refs, priority, n)
        if show_priorities_graphic:
            ids = sort_causes_priority(s, priority)[:n]
            show_filtered_causes(causes, scores, scm_refs, scms, ids)
        print()