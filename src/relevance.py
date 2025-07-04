# Relevance
import numpy as np, re

CAUSAL_VAR_REGEX = r"(flock|boid)_([\d]+)_((?:[a-z]|_)+)_([\d]+)"

def break_var(causal_var, dim2int=None): 
    entity, i, dim, t = re.match(CAUSAL_VAR_REGEX, causal_var).groups()
    if dim2int is not None:
        if isinstance(dim2int, list):
            dim2int = {var: i for i, var in enumerate(dim2int)}
        dim = int(dim2int[dim])
    return entity, int(i), dim, int(t)

    
def key(rule_value):
    rule, p, s, C, W, _ = rule_value
    return len(C), C, len(W), W, -s[1], -s[0]

def oldness_key(rule_value, variables, ref_time):
    recent_t = float('inf')
    for dim in rule_value[3]:
        _,_,_,t = break_var(variables[dim])
        recent_t = min(recent_t, t)
    return ref_time - recent_t

def cost_key(rule_values):
    return rule_values[2][1]

def complexity_key(rule_value, variables):
    desc_elts = set()
    for dim in rule_value[3]:
        var = variables[dim]
        desc_elts |= set(break_var(var))
    return len(desc_elts)