import matplotlib.colors as colors, matplotlib.animation as animation, matplotlib.pyplot as plt
import pygame, numpy as np

from sklearn.cluster import DBSCAN
try:
    from IPython.display import HTML
except ModuleNotFoundError:
    pass
from tqdm import tqdm
from enum import Enum
from scipy.spatial.distance import squareform, pdist, cdist

from utils import *

pygame.init()
pi = np.pi

base_hp = {
    "n_boids": 40,
    "boid_radius": 2,
    "init_width": 50,
    "init_height": 100,
    "max_speed": 4,
    "view_radius": 15,
    "minimum_separation": 6,
    "alignment_factor": 5  * pi / 180,
    "cohesion_factor": 6 * pi / 180,
    "separation_factor": 12  * pi / 180,
    "avoid_factor": 10  * pi / 180,
    "width": 300,
    "height": 300,
    "padding": 40,
    "obstacle_x": 150,
    "obstacle_y": 100,
    "obstacle_radius": 10,
    "n_steps": 40,
    "do_padding": False,
    "seed": 42
  }

dim_labels = (
        "max_speed", 
        "view_radius",
        "minimum_separation",
        "alignment_factor",
        "cohesion_factor",
        "separation_factor",
        "avoid_factor",
    )

# Constants
class Scenarii(Enum):
    FREE = "free_movement"
    ONE_FLOCK = "one_flock"
    FUSION = "flock_fusion"

# Utils
## Utils general

def avoid_round_obstacle(boids, hp):
    obstacle_pos = np.array([hp["obstacle_x"], hp["obstacle_y"]])
    vect = obstacle_pos - boids[:,:2]
    dirs = np.empty(len(boids))
    dirs = angle(vect)
    # filter = np.linalg.norm(vect, axis=1) < hp["obstacle_radius"] + hp["view_radius"]
    # filter &= np.abs(boids[:, 2] - dirs) < np.pi / 2
    filter = np.logical_and(
        np.linalg.norm(vect, axis=1) < hp["obstacle_radius"] + hp["view_radius"],
        np.abs(boids[:,2]-dirs) < pi/2
    )

    dirs[~filter] = np.nan
    dirs[filter] += pi/2 * np.sign(boids[filter,2]-dirs[filter])
    return dirs

def compute_obst_dist(history, hp):
    obstacle_pos = np.array([hp["obstacle_x"], hp["obstacle_y"]])
    dist_to_obstacle = cdist(
        np.vstack(history)[:,:2], 
        obstacle_pos.reshape(1,-1)
    ) - hp["obstacle_radius"] - hp["boid_radius"]
    return dist_to_obstacle

def get_labels(boids, hp, eps=None, flock_mapping=None):
    if eps is None: eps = hp["view_radius"]
    boids = boids.copy()
    boids[:,2] *= hp["width"] / (2*pi)
    frame_labels = DBSCAN(eps=eps, min_samples=3).fit_predict(boids)
    return map_labels(frame_labels, flock_mapping)

def get_flock_mapping(run, hp, eps=None):
    if eps is None:
        eps = hp["view_radius"]
    flocks = set()
    for t, boids in enumerate(run):
        labels = get_labels(boids, hp, eps)
        for l in np.unique(labels):
            if l == -1: continue
            flock = tuple(sorted(np.argwhere(labels == l).flatten()))
            flocks.add(flock)
    return list(flocks)

def map_labels(frame_labels, flock_mapping):
    if flock_mapping is None:
        return frame_labels
    mapped_labels = -np.ones_like(frame_labels)
    new_flock_id = len(flock_mapping)
    for l in np.unique(frame_labels):
        if l == -1: continue
        flock_filter = frame_labels == l
        flock = tuple(sorted(np.argwhere(flock_filter).flatten()))
        if flock in flock_mapping:
            mapped_labels[flock_filter] = flock_mapping.index(flock)
        else:
            mapped_labels[flock_filter] = new_flock_id
            new_flock_id += 1
    return mapped_labels

def get_first_hit(run, hp):
    return np.argwhere(
                        compute_obst_dist(run, hp).
                        reshape(run.shape[:2]) < 0
                    )[:,0].min()

## Simulation
def get_filters(s, params, hp):
    distances = squareform(pdist(s[:,:2]))
    other_filter = ~np.eye(len(s), dtype=bool)
    close_filter = distances <= params[:,2]
    close_neighbour_filter = np.logical_and(
        close_filter, 
        other_filter
    )
    regular_neighbour_filter = np.logical_and(
        distances <= params[:,1], 
        ~close_filter
    )
    regular_neighbour_filter = np.logical_and(
        regular_neighbour_filter, 
        np.tile(~close_neighbour_filter.any(axis=1),(s.shape[0],1)).T
    )
    return close_neighbour_filter, regular_neighbour_filter

def angle(x):
    return (np.arctan2(x.reshape(-1,2)[:,1], x.reshape(-1,2)[:,0]) + 2*pi) % (2*pi)

def direction(angle):
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)

def avoid_edges(boids, hp):
    dirs = np.full(len(boids), np.nan)
    dirs[boids[:,1] > hp["height"] - hp["padding"]] = 3*pi/2
    dirs[boids[:,1] < hp["padding"]] = pi/2
    dirs[boids[:,0] > hp["width"] - hp["padding"]] = pi
    dirs[boids[:,0] < hp["padding"]] = 0
    dirs[np.logical_and(
        boids[:,0] < hp["padding"], 
        boids[:,1] < hp["padding"]
    )] = pi/4
    dirs[np.logical_and(
        boids[:,0] < hp["padding"], 
        boids[:,1] > hp["height"] - hp["padding"]
    )] = 7*pi/4
    dirs[np.logical_and(
        boids[:,0] > hp["width"] - hp["padding"], 
        boids[:,1] < hp["padding"]
    )] = 3*pi/4
    dirs[np.logical_and(
        boids[:,0] > hp["width"] - hp["padding"], 
        boids[:,1] > hp["height"] - hp["padding"]
    )] = 5*pi/4
    return dirs

def turn(ns, s, close_filter, regular_filter, params, hp):
    if hp["do_padding"]:
        turn_towards(ns, avoid_edges(s, hp), params[:,6])
    turn_towards(ns, avoid_round_obstacle(s, hp), params[:,6])
    turn_towards(ns, separation(s, close_filter), params[:,5])
    turn_towards(ns, cohesion(s, regular_filter), params[:,4])
    turn_towards(ns, alignment(s, regular_filter), params[:,3])

def turn_towards(boids, target_headings, max_turn):
    nan_filter = ~np.isnan(target_headings)
    if not nan_filter.any(): return
    turns = target_headings[nan_filter] - boids[nan_filter, 2]
    neg_filter = np.logical_or(np.logical_and(turns < 0, turns > -pi), turns > pi)
    signs = np.ones_like(turns)
    signs[neg_filter] = -1  
    boids[nan_filter, 2] += signs * np.minimum(np.abs(turns), max_turn[nan_filter])
    boids[nan_filter, 2] %= 2 * np.pi

def separation(boids, neighbour_filter):
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbours = np.divide(
            neighbour_filter @ boids[:, :2],
            neighbour_filter.sum(axis=1, keepdims=True)
        )
    vects = - (mean_neighbours - boids[:,:2])
    return angle(vects)

def alignment(boids, neighbour_filter):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(neighbour_filter @ boids[:, 2], neighbour_filter.sum(axis=1))

def cohesion(boids, neighbour_filter):
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbours = np.divide(
            neighbour_filter @ boids[:, :2],
            neighbour_filter.sum(axis=1, keepdims=True)
        )
    vects = mean_neighbours - boids[:,:2]
    return angle(vects)

def update_flocks(s, params, hp):
    ns = s.copy()
    
    close_filter, regular_filter = get_filters(s, params, hp)
    turn(ns, s, close_filter, regular_filter, params, hp)
    
    ns[:,:2] += direction(ns[:,2]) * params[:,0, np.newaxis]
    return ns

def make_run_flocking(init_t, boids, hp, intervention=None):
    history = [boids]
    boid_params = np.array([hp[param] for param in dim_labels])
    boid_params = np.tile(boid_params, (boids.shape[0],1))
        
    for t in range(init_t, hp["n_steps"]):
        if intervention is not None:
            intervention(boids, boid_params, t)
        boids = update_flocks(boids, boid_params, hp)
        
        history.append(boids)
    return history

def simulation_flocking(rules, actual_run, hp, verbose, 
                        intervention):
    hp = hp | {"n_steps": actual_run.shape[0]}
    outputs = []
    for rule in tqdm(rules, disable=not verbose):
        intervention.set_rule(rule)
        init_t = min(intervention.interventions.keys())
        n_steps = hp["n_steps"] - init_t
        history = make_run_flocking(init_t, actual_run[init_t].copy(), hp, intervention)
        run = np.stack(history)
        # print(run.shape, actual_run.shape)
        run = np.vstack([actual_run[:init_t],run])
        d = compute_obst_dist(run, hp)
        outputs.append(
            (
                run, 
                bool(d.min() < 0), 
                (- d.min(), intervention.cost)
            )
        )
    return outputs

def make_animation(videos, titles=None, n_axes=None):
    def init():
        for video, img in zip(videos, imgs):
            img.set_array(video[0])
        return imgs

    def animate(i):
        for video, img in zip(videos, imgs):
            i = min((i*N)//N_frames, len(video)-1)
            img.set_array(video[i].copy())
        
        return imgs

    N = max(map(len,videos))
    N_frames = N

    plt.show()
    if n_axes is None:
        n_axes = (1,len(videos))
    if titles is None:
        titles = [None for _ in range(len(videos))]
        
    fig, axes = plt.subplots(*n_axes, squeeze=False)
    imgs = [ax.imshow(video[0], animated=True) 
            for ax, video in zip(axes.flatten(), videos)]
    for ax, title in zip(axes.flatten(), titles):
        ax.set_axis_off()
        if title is not None: ax.set_title(title)
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=N_frames, init_func=init, blit=True)
    plt.close()
    return anim.to_jshtml(default_mode="once", fps=10)

def render_boids(boids, screen, hp, highlight_ids=[]):
    screen.fill((255,255,255))

    # Draw padding
    if hp["do_padding"]:
        # Top
        pygame.draw.rect(screen, (200,200,200),
                         (
                             0,
                             0,
                             hp["width"],
                             hp["padding"]
                        )
                        )
                         
        # Left
        pygame.draw.rect(screen, (200,200,200),
                         (
                             0,
                             0,
                             hp["padding"],
                             hp["height"]
                        )
                        )
                         
        # Bottom
        pygame.draw.rect(screen, (200,200,200),
                         (
                             0,
                             hp["height"] - hp["padding"],
                             hp["width"],
                             hp["padding"]
                        )
                        )
                         
        # Right
        pygame.draw.rect(screen, (200,200,200),
                         (
                             hp["width"] - hp["padding"],
                             0,
                             hp["padding"],
                             hp["height"]
                        )
                        )
                         
        

    # Render obstacle
    obstacle_pos = np.array([hp["obstacle_x"], hp["obstacle_y"]])
    collide_obstacle = cdist(
        boids[:,:2], obstacle_pos.reshape(1,-1)
    ) < hp["obstacle_radius"] + hp["boid_radius"]
    
    pygame.draw.circle(
        screen, 
        (255,0,0) if any(collide_obstacle) else (125, 125, 125), 
        obstacle_pos, 
        hp["obstacle_radius"]
    )

    # Render boids
    labels = get_labels(boids, hp)
    u_labels = np.unique(labels)
    hsv = np.ones(shape=(len(boids),3))
    if len(u_labels) > 0:
        hsv[:,0] = labels/len(u_labels)
    rgb = colors.hsv_to_rgb(hsv)
    c = 255 - np.round(rgb*255)
    for i, l in enumerate(labels):
        if l == -1:
            c[i] = (125,125,125)
    for highlight_id in highlight_ids:
        c[highlight_id] = (0,0,0)
    for i, boid in enumerate(boids):
        if i in highlight_ids:
            pygame.draw.circle(screen, c[i], 
                boid[:2], hp["boid_radius"]+2, 2)
    
            end_line = boid[:2] + direction(boid[2]) * 10
            pygame.draw.line(screen, c[i], boid[:2], end_line, 2)
        else:
            pygame.draw.circle(screen, c[i], 
                boid[:2], hp["boid_radius"], 1)
    
            end_line = boid[:2] + direction(boid[2]) * 7
            pygame.draw.line(screen, c[i], boid[:2], end_line, 1)

def show_boids(boids, hp, highlight_ids=[], ax=None, title=None):
    show = False
    if ax is None:
        show = True
        ax = plt.gca()
    screen = pygame.display.set_mode((hp["width"], hp["height"]))
    render_boids(boids, screen, hp, highlight_ids)
    im = pygame.surfarray.pixels3d(screen).transpose(1,0,2).copy()
    ax.imshow(im)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()

def render_simulation(history, hp, highlight_ids=[]):
  states = []
  labels = []
  screen = pygame.display.set_mode((hp["width"], hp["height"]))
  for boids in history:
      pygame.display.flip()
      render_boids(boids, screen, hp, highlight_ids)
      states.append(pygame.surfarray.pixels3d(screen).transpose(1,0,2).copy())
  return states

def compare_run(cause, variables, actual_run, flock_mapping, hp):
    cf_run = np.stack(cause[-1])
    causal_vars = [variables[dim] for dim in cause[3]]
    highlight_boids = get_boid_ids(causal_vars, flock_mapping)
    actual_video = render_simulation(actual_run, hp, highlight_boids)
    cf_video = render_simulation(cf_run, hp, highlight_boids)
    return HTML(make_animation([actual_video, cf_video], ['Actual run', 'CF run']))

def show_simulation(run, hp):
    return HTML(make_animation([render_simulation(run, hp)]))


## Runs
def init_boids_uniform(hp, max_x=None, min_x=None, max_y=None, min_y=None):
    if max_x is None:
        max_x = hp["width"] - hp["padding"]
    if min_x is None:
        min_x = hp["padding"]
    if max_y is None:
        max_y = hp["height"] - hp["padding"]
    if min_y is None:
        min_y = hp["padding"]
    np.random.seed(hp["seed"])
    boids = np.zeros((hp["n_boids"], 3))
    boids[:,0] = np.random.randint(min_x, max_x, size=hp["n_boids"])
    boids[:,1] = np.random.randint(min_y, max_y, size=hp["n_boids"])
    boids[:,2] = np.random.uniform(0,2*pi, size=hp["n_boids"])
    return boids

def grid_coords(p, s):
    return int(p[0] / s), int(p[1] / s)

def fits(p, grid, r, s, w, h):
    gx, gy = grid_coords(p, s)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            ngx, ngy = gx + dx, gy + dy
            if 0 <= ngx < w and 0 <= ngy < h:
                neighbor = grid[ngy][ngx]
                if neighbor is not None and np.linalg.norm(np.array(p) - np.array(neighbor)) < r:
                    return False
    return True

def generate_poisson_disc_samples(width, height, r, k=30):
    s = r / np.sqrt(2)
    w = int(np.ceil(width / s))
    h = int(np.ceil(height / s))
    grid = [[None for _ in range(w)] for _ in range(h)]

    process_list = []
    sample_points = []

    # Generate the first sample
    first_point = (np.random.uniform(0, width), np.random.uniform(0, height))
    process_list.append(first_point)
    sample_points.append(first_point)
    gx, gy = grid_coords(first_point, s)
    grid[gy][gx] = first_point

    while process_list:
        idx = np.random.randint(0, len(process_list))
        parent_point = process_list[idx]
        found = False

        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            candidate = np.array(parent_point) + direction * np.random.uniform(r, 2 * r)

            if 0 <= candidate[0] < width and 0 <= candidate[1] < height and \
                    fits(candidate, grid, r, s, w, h):
                process_list.append(tuple(candidate))
                sample_points.append(tuple(candidate))
                gx, gy = grid_coords(candidate, s)
                grid[gy][gx] = tuple(candidate)
                found = True
                break

        if not found:
            process_list.pop(idx)

    return np.array(sample_points)

def init_boids_bottom(hp, k=30):
    np.random.seed(hp["seed"])
    points = generate_poisson_disc_samples(
        hp["init_width"], 
        hp["init_height"], 
        int(hp["minimum_separation"]*1.2), 
        k
    )
    points[:,0] += hp["width"]/2 - 1/2 * hp["init_width"]
    points[:,1] += hp["height"] - 2 * hp["padding"] - hp["init_height"]
    hp["n_boids"] = len(points)
    boids = np.zeros((hp["n_boids"], 3))
    boids[:,:2] = points
    boids[:, 2] = np.random.uniform(
        3*np.pi / 2 - pi / 10, 
        3*np.pi / 2 + pi / 10, 
        size=hp["n_boids"])
    return boids

def init_boids_clash(hp, k=30):
    np.random.seed(hp["seed"])
    # Bottom boids
    points = generate_poisson_disc_samples(
        hp["init_width"], 
        hp["init_height"], 
        int(hp["minimum_separation"]*1.2), 
        k
    )
    points[:,0] += hp["width"]/3 - 1/2 * hp["init_width"]
    points[:,1] += hp["height"] - 2 * hp["padding"] - hp["init_height"]
    
    boids = np.zeros((points.shape[0], 3))
    boids[:,:2] = points
    boids[:, 2] = 3*np.pi / 2

    # Clash boids
    clash_points = generate_poisson_disc_samples(
        hp["init_width"]/2, 
        hp["init_height"]/2, 
        int(hp["minimum_separation"]*1.2), 
        k
    )
    clash_points[:,0] += hp["padding"]
    clash_points[:,1] += hp["height"]*.75
    clash_boid = np.zeros((clash_points.shape[0], 3))
    clash_boid[:,:2] = clash_points
    clash_boid[:,2] -= pi/3
    
    boids = np.vstack([boids, clash_boid])
    hp["n_boids"] = boids.shape[0]
    return boids
