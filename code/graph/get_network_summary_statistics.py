import numpy as np
import pickle

# Assume we have 5 edge dictionaries indexed by RINPERSOON
# family_edges = {}
# household_edges = {}
# colleague_edges = {}
# neighbor_edges = {}
# education_edges = {}

root = '/gpfs/ostor/ossc9424/homedir/Dakota_Network/'

with open(root + "path_to_family_edges.pkl", 'rb') as pkl_file:
    family_edges = pickle.load(pkl_file)

with open(root + "path_to_household_edges.pkl", 'rb') as pkl_file:
    household_edges = pickle.load(pkl_file)

with open(root + "path_to_neighbor_edges.pkl", 'rb') as pkl_file:
    neighbor_edges = pickle.load(pkl_file)

with open(root + "path_to_collague_edges.pkl", 'rb') as pkl_file:
    colleague_edges = pickle.load(pkl_file)

with open(root + "path_to_education_edges.pkl", 'rb') as pkl_file:
    education_edges = pickle.load(pkl_file)

num_family_connections = []
num_household_connections = []
num_colleague_connections = []
num_neighbor_connections = []
num_education_connections = []

edge_types = ['family', 'household', 'colleague', 'neighbor', 'education']

family_degree = []
household_degree = []
neighbor_degree = []
colleague_degree = []
education_degree = []

family_household_intersections = []
family_neighbor_intersections = []
family_colleague_intersections = []
family_education_intersections = []

household_neighbor_intersections = []
household_colleague_intersections = []
household_education_intersections = []

neighbor_colleague_intersections = []
neighbor_education_intersections = []

colleague_education_intersections = []

# Use family as the canonical layer for person inclusion
for person in family_edges:

    # Get family connections
    if person in family_edges:
        family_connections = set(family_edges[person])
    else:
        family_connections = set()

    # Get household connections if applicable
    if person in household_edges:
        household_connections = set(household_edges[person])
    else:
        household_connections = set()

    # Get neighbor connections if applicable
    if person in neighbor_edges:
        neighbor_connections = set(neighbor_edges[person])
    else:
        neighbor_connections = set()

    # Get colleague connections if applicable
    if person in colleague_edges:
        colleage_connections = set(colleague_edges[person])
    else:
        colleage_connections = set()

    # Get education connections if applicable
    if person in education_edges:
        education_connections = set(education_edges[person])
    else:
        education_connections = set()

    # Append the degree of each layer for this person
    family_degree.append(len(family_connections))
    household_degree.append(len(household_connections))
    neighbor_degree.append(len(neighbor_connections))
    colleague_degree.append(len(colleage_connections))
    education_degree.append(len(education_connections))

    family_household_intersection = len(family_connections.intersection(household_connections))
    family_neighbor_intersection = len(family_connections.intersection(neighbor_connections))
    family_colleague_intersection = len(family_connections.intersection(colleage_connections))
    family_education_intersection = len(family_connections.intersection(education_connections))
    # Append
    family_household_intersections.append(family_household_intersection)
    family_neighbor_intersections.append(family_neighbor_intersection)
    family_colleague_intersections.append(family_colleague_intersection)
    family_education_intersections.append(family_education_intersection)

    household_neighbor_intersection = len(household_connections.intersection(neighbor_connections))
    household_colleague_intersection = len(household_connections.intersection(colleage_connections))
    household_education_intersection = len(household_connections.intersection(education_connections))
    # Append
    household_neighbor_intersections.append(household_neighbor_intersection)
    household_colleague_intersections.append(household_colleague_intersection)
    household_education_intersections.append(household_education_intersection)

    neighbor_colleague_intersection = len(neighbor_connections.intersection(colleage_connections))
    neighbor_education_intersection = len(neighbor_connections.intersection(education_connections))
    # Append
    neighbor_colleague_intersections.append(neighbor_colleague_intersection)
    neighbor_education_intersections.append(neighbor_education_intersection)

    colleague_education_intersection = len(colleage_connections.intersection(education_connections))
    # Append
    colleague_education_intersections.append(colleague_education_intersection)

# Average across collections, get mean and std
save_data = {}

# Family stuff
save_data['family'] = {}
save_data['family']['degree_mean'] = np.mean(family_degree)
save_data['family']['degree_std'] = np.std(family_degree)
save_data['family']['degree_max'] = np.max(family_degree)
save_data['family']['degree_min'] = np.min(family_degree)

# Household stuff
save_data['household'] = {}
save_data['household']['degree_mean'] = np.mean(household_degree)
save_data['household']['degree_std'] = np.std(household_degree)
save_data['household']['degree_max'] = np.max(household_degree)
save_data['household']['degree_min'] = np.min(household_degree)

# Neighbor stuff
save_data['neighbor'] = {}
save_data['neighbor']['degree_mean'] = np.mean(neighbor_degree)
save_data['neighbor']['degree_std'] = np.std(neighbor_degree)
save_data['neighbor']['degree_max'] = np.max(neighbor_degree)
save_data['neighbor']['degree_min'] = np.min(neighbor_degree)

# Colleague stuff
save_data['colleague'] = {}
save_data['colleague']['degree_mean'] = np.mean(colleague_degree)
save_data['colleague']['degree_std'] = np.std(colleague_degree)
save_data['colleague']['degree_max'] = np.max(colleague_degree)
save_data['colleague']['degree_min'] = np.min(colleague_degree)

# Education stuff
save_data['education'] = {}
save_data['education']['degree_mean'] = np.mean(education_degree)
save_data['education']['degree_std'] = np.std(education_degree)
save_data['education']['degree_max'] = np.max(education_degree)
save_data['education']['degree_min'] = np.min(education_degree)

# Intersection stuff
save_data['intersection'] = {}
save_data['intersection']['family_household_mean'] = np.mean(family_household_intersections)
save_data['intersection']['family_household_std'] = np.std(family_household_intersections)
save_data['intersection']['family_household_min'] = np.min(family_household_intersections)
save_data['intersection']['family_household_max'] = np.max(family_household_intersections)

save_data['intersection']['family_neighbor_mean'] = np.mean(family_neighbor_intersections)
save_data['intersection']['family_neighbor_std'] = np.std(family_neighbor_intersections)
save_data['intersection']['family_neighbor_min'] = np.min(family_neighbor_intersections)
save_data['intersection']['family_neighbor_max'] = np.max(family_neighbor_intersections)

save_data['intersection']['family_colleague_mean'] = np.mean(family_colleague_intersections)
save_data['intersection']['family_colleague_std'] = np.std(family_colleague_intersections)
save_data['intersection']['family_colleague_min'] = np.min(family_colleague_intersections)
save_data['intersection']['family_colleague_max'] = np.max(family_colleague_intersections)

save_data['intersection']['family_education_mean'] = np.mean(family_education_intersections)
save_data['intersection']['family_education_std'] = np.std(family_education_intersections)
save_data['intersection']['family_education_min'] = np.min(family_education_intersections)
save_data['intersection']['family_education_max'] = np.max(family_education_intersections)


save_data['intersection']['household_neighbor_mean'] = np.mean(household_neighbor_intersections)
save_data['intersection']['houshold_neighbor_std'] = np.std(household_neighbor_intersections)
save_data['intersection']['household_neighbor_min'] = np.min(household_neighbor_intersections)
save_data['intersection']['houshold_neighbor_max'] = np.max(household_neighbor_intersections)

save_data['intersection']['household_colleague_mean'] = np.mean(household_colleague_intersections)
save_data['intersection']['houshold_colleague_std'] = np.std(household_colleague_intersections)
save_data['intersection']['household_colleague_min'] = np.min(household_colleague_intersections)
save_data['intersection']['houshold_colleague_max'] = np.max(household_colleague_intersections)

save_data['intersection']['household_education_mean'] = np.mean(household_education_intersections)
save_data['intersection']['houshold_education_std'] = np.std(household_education_intersections)
save_data['intersection']['household_education_min'] = np.min(household_education_intersections)
save_data['intersection']['houshold_education_max'] = np.max(household_education_intersections)


save_data['intersection']['neighbor_colleague_mean'] = np.mean(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_std'] = np.std(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_min'] = np.min(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_max'] = np.max(neighbor_colleague_intersections)

save_data['intersection']['neighbor_education_mean'] = np.mean(neighbor_education_intersections)
save_data['intersection']['neighbor_education_std'] = np.std(neighbor_education_intersections)
save_data['intersection']['neighbor_education_min'] = np.min(neighbor_education_intersections)
save_data['intersection']['neighbor_education_max'] = np.max(neighbor_education_intersections)


save_data['intersection']['colleague_education_mean'] = np.mean(colleague_education_intersections)
save_data['intersection']['colleague_education_std'] = np.std(colleague_education_intersections)
save_data['intersection']['colleague_education_min'] = np.min(colleague_education_intersections)
save_data['intersection']['colleague_education_max'] = np.max(colleague_education_intersections)

# Save everything
with open("'/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/network_summary_statistics.pkl", 'wb') as pkl_file:
    pickle.dump(save_data, pkl_file)