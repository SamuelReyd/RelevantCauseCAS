



# Relevance
def key(rule_value):
    rule, p, s, C, W, _ = rule_value
    return len(C), C, len(W), W, -s[1], -s[0]

def filter_causes(full_causes):
    sorted_causes = sorted(full_causes, key=key)
    i = 0
    while i < len(sorted_causes)-1:
        if sorted_causes[i][3] == sorted_causes[i+1][3]:
            del sorted_causes[i+1]
        else:
            i += 1
    return sorted_causes

def rencency_key(rule_value, variables, ref_time):
    rule, p, s, C, W, _ = rule_value
    recent_t = float('inf')
    for dim in C:
        _,_,_,t = break_var(variables[dim])
        recent_t = min(recent_t, t)
    return recent_t / ref_time

def oldness_key(rule_value, variables, ref_time):
    recent_t = float('inf')
    for dim in rule_value[3]:
        _,_,_,t = break_var(variables[dim])
        recent_t = min(recent_t, t)
    return ref_time - recent_t

def abnormality_key(rule_value, variables, flock_mapping, actual_run, freq_data, hp):
    rule, p, s, C, W, _ = rule_value
    score = 1
    for c in C:
        norm = get_normality(variables[c], actual_run, freq_data, flock_mapping, hp)
        score += get_normality(variables[c], actual_run, freq_data, flock_mapping, hp)
    return 1-(score/len(C))

def cost_key(rule_values):
    return rule_values[2][1]

def complexity_key(rule_value):
    return len(rule_value[0])

def strong_sufficiency_key(rule_value, causes, variables, instance, simulation):
    ref_rule = rule_value[0]
    actual_rule = [(var, instance[var]) for var, _ in ref_rule]
    actual_rules = [actual_rule + list(cause[0]) for cause in causes if cause[0] != ref_rule]

    res = simulation(actual_rules)
    return np.mean([r[1] for r in res])

def relevance_key(rule_value, variables, flock_mapping, actual_run, freq_data, ref_time, hp):
    return (
        rencency_key(rule_value, variables, ref_time) * 
        abnormality_key(rule_value, variables, flock_mapping, actual_run, freq_data, hp)
    )

def cf_key(rule_value, max_dist = 10):
    d = min(-rule_value[2], max_dist)
    return d / max_dist

def best_cause(causes, key):
    return sorted(causes, key=key)[-1]



