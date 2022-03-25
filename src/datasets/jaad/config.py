import os

JAAD_PATH = os.path.join('..', 'resources', 'JAAD')

PROJECTION_MAPPING = {'gait_attention': ['look', 'action']}

def get_projection(projection):
    return PROJECTION_MAPPING[projection]
