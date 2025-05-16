from flocking import dim_labels
from beam_search import beam_search
from enum import Enum


CAUSAL_VAR_REGEX = r"(flock|boid)_([\d]+)_((?:[a-z]|_)+)_([\d]+)"

TIME_GRANULARITY = 5
SAMPLING_GRANULARITY = 10
TIME_RANGE = 40

class Granularity(Enum):
    FLOCK = "flock"
    BOID = "boid" 

class CausalObservation(Enum):
    BOOLEAN = "boolean"
    VARS = "variables"
    PARAMS = "parameters"

class Metrics(Enum):
    COST = "Cost"
    OLDNESS = "Oldness"
    COMPLEXITY = "Complexity"

def map_vars(C, variables):
    return [variables[c] for c in C]

def get_boid_ids(causal_vars, flock_mapping):
    boid_ids = set()
    for causal_var in causal_vars:
        boid_ids |= set(get_boid_ids_once(causal_var, flock_mapping))
    return list(boid_ids)

def get_boid_ids_once(causal_var, flock_mapping):
    agent_type, i, _, t = break_var(causal_var)
    if agent_type == Granularity.FLOCK.value:
        return flock_mapping[i]
    return [i]

def break_var(causal_var, dim2int=None): 
    entity, i, dim, t = re.match(CAUSAL_VAR_REGEX, causal_var).groups()
    if dim2int is not None:
        if isinstance(dim2int, list):
            dim2int = {var: i for i, var in enumerate(dim2int)}
        dim = int(dim2int[dim])
    return entity, int(i), dim, int(t)

## Boolean interventions
def check_single_flock(flock, hp):
    return not any(get_labels(flock, hp))

def check_several_flock(flock, hp):
    return any(get_labels(flock, hp))

def check_no_flock(flock, hp):
    return all(get_labels(flock, hp) == -1)

def get_flock_direction(boids, target_headings, max_turn):
    turns = target_headings - boids[:, 2]
    neg_filter = np.logical_or(np.logical_and(turns < 0, turns > -pi), turns > pi)
    signs = np.ones_like(turns)
    signs[neg_filter] = -1
    boids[:, 2] += signs * np.minimum(np.abs(turns), max_turn)
    boids[:, 2] %= 2 * np.pi
    return boids[:, 2]

def move_flock(boids, boid_ids, flock_id, orientation):
    flock = boids[boid_ids,:]
    # TODO -> use subflocks for better reliability
    flock_center = np.mean(flock[:,:2], axis=0)
    oposite_vect = flock[:,:2] - flock_center
    oposite_dir = angle(oposite_vect)
    boids[boid_ids,:2] += orientation * direction(oposite_dir) * .1

    if orientation == 1:
        heading = oposite_dir
    else:
        heading = np.mean(flock[:,2])
    boids[boid_ids,2] = get_flock_direction(
        boids[boid_ids,:], 
        heading, 
        .1
    )

def set_flock_state(boids, boid_ids, hp, flock_id, state):
    origin_boids = boids.copy()
    conditions = [check_no_flock, check_several_flock, check_single_flock]
    for i, condition in enumerate(conditions):
        if condition(boids[boid_ids,:], hp):
            orientation = np.sign(i-state)
            break
    if not orientation: 
        return
    stop_condition = conditions[state]
    if i == 0 and state == 1:
        stop_condition = lambda a, b: check_several_flock(a,b) or check_single_flock(a,b)
    for _ in range(2000):
        move_flock(boids, boid_ids, flock_id, orientation)
        if stop_condition(boids[boid_ids,:], hp):
            break
    else:
        print("!!! Failed intervention... !!!")
        # print(f"{i=}, {state=}, {orientation=:.0f}")
        # print("Original flock labels", 
        #       get_labels(origin_boids[boid_ids,:], hp), 
        #       check_no_flock(origin_boids[boid_ids,:], hp))
        # print("Final flock labels", get_labels(boids[boid_ids,:], hp))
        # show_boids(origin_boids, hp, boid_ids)
        # plt.show()
        # show_boids(boids, hp)
        # plt.show()
        # raise Exception("Intervention failed")

## SCM
class SCM:
    def __init__(self, actual_run, hp, 
                 freq_data=None, flock_mapping=None,
                 sampling_granularity=None,
                 eps=None, time_range=None,
                 time_granularity=None, verbose=0):
        if time_granularity is None:
            time_granularity = TIME_GRANULARITY
        if sampling_granularity is None:
            sampling_granularity = SAMPLING_GRANULARITY
        if time_range is None:
            time_range = TIME_RANGE
        self.eps = eps
        self.actual_run = actual_run[-time_range:]
        self.freq_data = freq_data
        self.flock_mapping = flock_mapping
        self.hp = hp
        self.verbose = verbose
        self.time_granularity = time_granularity
        self.sampling_granularity = sampling_granularity
        self.variables = []
        self.domains = []
        self.instance = []
        self.initialized = False

    def find_causes(self, max_step=3, early_stop=False, beam_size=30, bs_verbose=0, sim_verbose=0):
        v = self.verbose
        self.verbose = sim_verbose
        self.init_structure()
        full_causes = beam_search(**self.get_dict(), max_steps=max_step, early_stop=early_stop, 
                                  beam_size=beam_size, verbose=bs_verbose)
        self.verbose = v
        self.causes = full_causes

    def compute_scores(self, verbose=0):
        self.scores = {}
        for metric in list(Metrics):
            scores = []
            for cause in self.causes:
                scores.append(self.compute_score(cause, metric))
            self.scores[metric.value] = scores

    def compute_score(self, cause, metric):
        match metric:
            case Metrics.COST:
                return cost_key(cause)
            case Metrics.OLDNESS:
                first_hit = np.argwhere(
                    compute_obst_dist(self.actual_run, self.hp).
                    reshape(self.actual_run.shape[:2]) < 0
                )[:,0].min()
                return oldness_key(cause, self.variables, first_hit)
            case Metrics.COMPLEXITY:
                return complexity_key(cause)

    def show_causation_results(self):
        show_causation_results(self.causes, self.scores, self.variables, 
                               self.instance, self.actual_run, self.flock_mapping, 
                               self.hp, 
                               Granularity.FLOCK if isinstance(self, FlockSCM) else Granularity.BOID
                              )

    def compare_best_cf(self, metric):
        i = np.argmax(self.scores[metric])
        return compare_run(self.causes[i], self.variables, self.actual_run, self.flock_mapping, self.hp)

    def get_dict(self):
        return {"variables": self.variables, 
                "instance": self.instance, 
                "domains": self.domains, 
                "simulation": self.get_simulation()}

    def set_rule(self, rule):
        self.interventions = {}
        self.cost = 0
        for dim, value in rule:
            var2int = {var: i for i, var in enumerate(self.dim_labels)}
            _, i, dim, t = break_var(self.variables[dim], var2int)
            add_dict_value(self.interventions, (i, dim, value), [t])

    def get_label(self):
        return self.obs_type + "-" + self.entity_type

    def get_ranges(self):
        return self.dim_ranges(self.hp)

    @ staticmethod
    def make_domain(dim_deltas, dim_range, dim_label, actual_value):
        values = np.array(sorted(dim_deltas + actual_value))
        if dim_label == "angle":
            return np.unique(((values + 2*pi) % (2*pi)).round(6))
        return values[(dim_range[0]<= values) & (values <= dim_range[1])]

    def get_intervention_steps(self):
        first_hit = np.argwhere(
                        compute_obst_dist(self.actual_run, self.hp).
                        reshape(self.actual_run.shape[:2]) < 0
                    )[:,0].min()
        steps = np.geomspace(1, first_hit, self.time_granularity, dtype=int)
        _, idx = np.unique(steps, return_index=True)
        return (first_hit - steps[idx])[::-1]

    def get_intervention_deltas(self):
        deltas = []
        for (r_min, r_max), dim_label in zip(self.get_ranges(), self.dim_labels):
            if dim_label == "angle":
                radius = (r_max - r_min)/2
                values = np.geomspace(1,radius,self.sampling_granularity//2)
            else:
                radius = (r_max - r_min)
                values = np.geomspace(1,radius,self.sampling_granularity//2)
            deltas.append(np.array(sorted(
                [0] + values.tolist() + (-values).tolist()
            )))
        return deltas

    def get_entities(self, t):
        raise NotImplementedError
        
    def get_vars(self, t, i):
        raise NotImplementedError
        
    def compute_actual_value(self, t, i, dim):
        raise NotImplementedError
        
    def get_variable(self, t, i, dim):
        raise NotImplementedError
        
    def complete_data(self, actual_value, t, i, dim):
        pass

    def __call__(self, boids, params, t):
        ref_boids = boids.copy()
        ref_params = params.copy()
        self.do_intervention(boids, params, t)
        self.add_cost(ref_boids, ref_params, boids, params)

    def add_cost(self, ref_boids, ref_params, boids, params):
        pass
    
    def init_structure(self):
        if self.initialized: return 
        # Create variables, domains and actual values for each timestep, entity and associated variable
        intervention_steps = self.get_intervention_steps()
        deltas = self.get_intervention_deltas()
        for t in intervention_steps:
            for i in self.get_entities(t):
                for dim in self.get_vars(t, i):
                    actual_value = self.compute_actual_value(t,i,dim)
                    variable = self.get_variable(t, i, dim)
                    self.variables.append(variable)
                    self.domains.append(
                        self.make_domain(deltas[dim], 
                                         self.get_ranges()[dim],
                                         self.dim_labels[dim],
                                         actual_value)
                    )
                    self.instance.append(actual_value)
                    self.complete_data(actual_value, t, i, dim)
        self.initialized = True

    def get_simulation(self):
        return lambda rules: \
            simulation_flocking(rules, 
                                self.actual_run, self.hp, 
                                self.verbose, 
                                self)

class GranularSCM(SCM):
    entity_type = Granularity.BOID
    flock_mapping = None
    
    def get_entities(self, t):
        return range(self.actual_run.shape[1])

class FlockSCM(SCM):
    entity_type = Granularity.FLOCK
    
    def __init__(self, actual_run, hp, 
                 freq_data=None, flock_mapping=None,
                 sampling_granularity=None,
                 eps=None, time_range=None,
                 time_granularity=None, verbose=0):
        if eps is None:
            eps = hp["view_radius"]
        SCM.__init__(self, actual_run, hp, freq_data, flock_mapping,
                     sampling_granularity, eps, time_range, time_granularity, verbose)
        self.current_lbls = None
        self.gaps = {}
        self.flock_map = dict(enumerate(self.flock_mapping))
    
    def get_entities(self, t):
        self.current_lbls = get_labels(self.actual_run[t], self.hp, self.eps, self.flock_mapping)
        return set(self.current_lbls) - {-1}

class VarSCM(SCM):
    obs_type = CausalObservation.VARS
    dim_labels = ["x", "y", "angle"]
    var_label_template = "{entity_type}_{entity_id}_{dim_label}_{t}"

    @staticmethod
    def dim_ranges(hp):
        return [(0, hp["width"]), (0, hp["height"]), (0, 2*pi)]

    def get_vars(self, t, i):
        return range(self.actual_run.shape[-1])

    def get_variable(self, t, i, dim):
        return self.var_label_template.format(
            entity_type=self.entity_type.value, 
            entity_id=i, 
            dim_label=self.dim_labels[dim], 
            t=t)

    def add_cost(self, ref_boids, ref_params, cf_boids, cf_params):
        ranges = self.get_ranges()
        rgs = np.array(ranges)
        rgs = rgs[:,1]-rgs[:,0]
        dif = np.abs(ref_boids - cf_boids)/rgs
        self.cost += dif.sum()
        

class ParamSCM(SCM):
    obs_type = CausalObservation.PARAMS
    dim_labels = dim_labels
    var_label_template = "{entity_type}_{entity_id}_{dim_label}_{t}"
    freq_data = None

    @staticmethod
    def dim_ranges(hp):
        ranges = ((1,6),(5,25),(3,12))
        ranges += ((0, 20*pi/180),) * 4
        return ranges

    def get_variable(self, t, i, dim):
        return self.var_label_template.format(
            entity_type = self.entity_type.value, entity_id = i,
            dim_label = self.dim_labels[dim], t = t)
    
    def get_vars(self, t, i):
        return range(len(self.dim_labels))

    def make_domain(self, dim_deltas, dim_range, dim_label, actual_value):
        values = np.linspace(*dim_range, self.sampling_granularity)
        all_values = set(values) | {actual_value}
        return np.array(sorted(all_values))

    def add_cost(self, ref_boids, ref_params, cf_boids, cf_params):
        ranges = self.get_ranges()
        rgs = np.array(ranges)
        rgs = rgs[:,1]-rgs[:,0]
        dif = np.abs(ref_params - cf_params)/rgs
        self.cost += dif.sum()

class GranularVarSCM(GranularSCM, VarSCM):
    def compute_actual_value(self, t, i, dim):
        return float(self.actual_run[t,i,dim])

    def do_intervention(self, boids, params, t):
        for boid, dim, value in self.interventions.get(t,[]):
            boids[boid, dim] = value

class FlockVarSCM(FlockSCM, VarSCM):
        
    def compute_actual_value(self, t, i, dim):
        flock = self.actual_run[t, self.current_lbls==i]
        return float(flock[:,dim].mean())
        
    def get_intervention(self):
        return FlockVarIntervention(self.variables, self.flock_mapping, self.gaps)

    def complete_data(self, actual_value, t, i, dim):
        lbls = self.current_lbls
        flock = self.actual_run[t,lbls==i]
        gap = flock[:,dim] - actual_value
        set_dict_value(self.gaps, gap, [t,i,dim])
        
    def do_intervention(self, boids, params, t):
        for l, var, value in self.interventions.get(t, []):
            boids[self.flock_map[l], var] = value + self.gaps[t][l][var]

class GranularParamSCM(GranularSCM, ParamSCM):
    
    def compute_actual_value(self, t, i, dim):
        return self.hp[self.dim_labels[dim]]

    def do_intervention(self, boids, params, t):
        for i, var, value in self.interventions.get(t, []):
            params[i,var] = value

class FlockParamSCM(FlockSCM, ParamSCM):
    
    def compute_actual_value(self, t, i, dim):
        return self.hp[self.dim_labels[dim]]

    def do_intervention(self, boids, params, t):
        for l, var, value in self.interventions.get(t, []):
            params[self.flock_map[l], var] = value

class FlockBooleanSCM(FlockSCM):
    dim_labels = ["step"]
    obs_type = CausalObservation.BOOLEAN

    dim_ranges = VarSCM.dim_ranges
    
    def compute_actual_value(self, t, i, dim):
        return 2

    def get_base_domains(self):
        return [{0,1,3}]

    def get_vars(self, t, i):
        return [0]

    def get_variable(self, t, i, dim):
        return f"flock_{i}_step_{t}"
        
    def do_intervention(self, boids, params, t):
        # TODO -> compute orientation based on actual causal variable value vs intervention
        for l, var, value in self.interventions.get(t, []):
            set_flock_state(boids, self.flock_map[l], self.hp, l, value)

    def add_cost(self, ref_boids, ref_params, cf_boids, cf_params):
        ranges = self.get_ranges()
        rgs = np.array(ranges)
        rgs = rgs[:,1]-rgs[:,0]
        dif = np.abs(ref_boids - cf_boids)/rgs
        self.cost += dif.sum()

    def get_ranges(self):
        return VarSCM.dim_ranges(self.hp)

    def get_intervention_deltas(self):
        return [None]

    def make_domain(self, dim_deltas, dim_range, dim_label, actual_value):
        return [0,1,2]