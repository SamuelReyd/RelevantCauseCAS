import numpy as np, bisect, time
from tqdm import tqdm
from itertools import count

def render_step(verbose, causes, non_causes, instance, variables):
    if len(non_causes):
        if verbose == 2:
            print("Number of causes found:", len(causes))
            print("Number of non-causes remaining:", len(non_causes))
            print("Best non-cause:")
            show_rule(non_causes[0], variables)
            print("Worst non-cause:")
            show_rule(non_causes[-1], variables)
        if verbose >= 3:
            print("Causes found this step:")
            show_rules(causes, variables)
            print("Rules passed for next step:")
            show_rules(non_causes, variables)
    else:
        print("No rule available")

def render_time(time):
    n_hours = time//(60*60)
    ret = ""
    if int(n_hours): ret += f"{n_hours:.0f}h"
    if int(n_hours) or int((time%3600)//60): ret += f"{(time%3600)//60:.0f}min"
    if not int(n_hours): ret += f"{time%60:.0f}s"
    if not (int(n_hours) or int((time%3600)//60)): ret += f"{(time-int(time))*100:.0f}ms"
    return ret

def show_raw_rule(rule):
    C, W = get_sets(rule, instance)
    s, p = 0,0
    show_rule((rule, s, p, C, W, None), variables)

def is_minimal(C, W, causes, contingencies):
    return not any([c < C or (C==c and w < W) for (c,w) in zip(causes, contingencies)])

def filter_minimality(rule_values, Cs, Ws, verbose=0):
    o_rule_values, o_Cs, o_Ws = [], [], []
    for rule_value in tqdm(rule_values, disable=not verbose):
        C, W = rule_value[3:5]
        if is_minimal(C, W, Cs, Ws):
            o_rule_values.append(rule_value)
            o_Cs.append(C)
            o_Ws.append(W)
    return o_rule_values, o_Cs, o_Ws

def get_sets(rule, instance):
    C = set()
    W = set()
    for feature, value in rule:
        if instance[feature] != value:
            C.add(feature)
        else:
            W.add(feature)
    return C, W

def sort_key(rule_values):
    _, rule_output, rule_score, C, W, _ = rule_values
    return (rule_output, rule_score, len(C), C, len(W), W)

def get_rule_desc(rule_values, variables, show_score=False):
    rule, output, score, C, W, _ = rule_values
    dim2value = dict(rule)
    C = {variables[c]:f'{dim2value[c]:.2f}' for c in C}
    W = {variables[w]:f'{dim2value[w]:.2f}' for w in W}
    if show_score: 
        if isinstance(score, tuple):
            return f"{C=}, {W=}, {output=}, {score=}"
        return f"{C=}, {W=}, {output=}, {score=:.3f}"
    return f"{C=}, {W=}"

def show_rule(rule_values, variables):
    print(get_rule_desc(rule_values, variables, True))
    
def show_rules(rule_values, variables):
    for r_values in rule_values:
        show_rule(r_values, variables)

def get_initial_rules(instance, domains):
    rules = []
    for feature, domain in enumerate(domains):
        for value in domain:
            if instance[feature] != value:
                rules.append(((feature, value),))
    return rules


# TODO: keep track of the provenance of each rule to remove ancestors when a cause is found (based only on C, not W)
# I.e. filter minimality in constant time
def get_rules(previous_rules, domains, instance, valid_causal_sets, verbose=False):
    # Build new rules on top of the previous ones
    # The previous rules are not valid (i.d. they do not define causes)
    new_rules = set()

    # Iterate through the previous rules
    for rule in tqdm(previous_rules, disable=not verbose): # Complexity: O(1)
        C, W = get_sets(rule, instance)
        for feature, domain in enumerate(domains): # Complexity: O(n)
            # Don't consider features already in rule
            if feature in C|W:
                continue
                
            # Don't consider the rule if it is not minimal
            if any([c <= C|{feature} for c in valid_causal_sets]): # Complexity O(n)
                continue

            # Add new rules with the feature
            for value in domain:
                # Build the rule
                new_rule = rule + ((feature, value),)
                # Add the new rule to the next rules
                new_rules.add(tuple(sorted(new_rule)))
    return sorted(new_rules)

def beam_search(instance, domains, simulation, variables, 
                          max_steps=5, beam_size=10, epsilon=.05, early_stop=True, verbose=0, max_time=None):
    # verbose: 
    #  = 1 -> best cause at the end, tqdm for steps
    #  >= 2 -> removes step tqdm, adds step header + number of cause found + best and worse non causes
    #  >= 3 -> adds all causes + tqdm for get_rules
    
    all_causes = []
    causal_sets = []
    init_time = time.time()
    if max_steps is -1 or max_steps is None: iterator = count(start=1, step=1)
    else: iterator = range(1,max_steps+1)
    
    for t in tqdm(iterator, disable=(verbose!=1)):
        # Render the step
        if verbose >= 2: print(f"{f'Step {t}':=^30}")
            
        # Create the rules for step t base on the ones from t-1, we use the initial ones if t==1
        if not (t-1):
            beams = get_initial_rules(instance, domains)
        else:
            beams =  get_rules(
                beams, 
                domains, 
                instance, 
                causal_sets, 
                verbose >= 3)
        
        # Check for early stopping
        if not len(beams) \
        or (early_stop and len(all_causes)) \
        or (max_time is not None and time.time()-init_time > max_time):
            break

        if verbose >= 2: print(f"Evaluating {len(beams)} rules")
        # Evaluate the rules using the simulation 
        cf_values = simulation(beams)
        
        # Build the tuples of rule values: 
        # (rule, output, score, C length, C inclusion, W length, W inclusion)
        causes, Cs, Ws = [], [], []
        non_causes = []
        for rule, (cf_state, cf_output, cf_score) in zip(beams, cf_values):
            C, W = get_sets(rule, instance)
            rule_value = (rule, cf_output, cf_score, C, W, cf_state)

            # Save causes and keep n best non-causes for next step
            if cf_output < epsilon: 
                causes.append(rule_value)
                Cs.append(C)
                Ws.append(W)
            else:
                non_causes.append(rule_value)

        # Filter causes to keep only minimal ones
        causes, Cs, Ws = filter_minimality(causes, Cs, Ws)
        
        all_causes += causes
        causal_sets += Cs

        # Build next beams
        # !!! Added !!!
        non_causes = [rule_value for rule_value in non_causes if 
                     not any([c <= rule_value[3] for c in Cs])]
        
        if beam_size == -1:
            non_causes = sorted(non_causes, key=sort_key)
        else:
            non_causes = sorted(non_causes, key=sort_key)[:beam_size]
        beams = [rule_value[0] for rule_value in non_causes]

        # Render step output
        render_step(verbose, causes, non_causes, instance, variables)

    # Sort final rule set
    all_causes.sort(key=sort_key)

    # Render final result
    if verbose:
        print(f"----> Found {len(all_causes)} causes.")
        print(f"{'Overall best rule:':=^30}")
        show_rule(all_causes[0], variables)
        
    return all_causes