import gym
from gym.wrappers import TimeLimit
from brax.envs import wrappers
import numpy as np
from brax.envs.halfcheetah import Halfcheetah
from brax.envs.grasp import Grasp
from brax.envs.ant import Ant
from brax.envs.fetch import Fetch

from brax.envs.halfcheetah import Halfcheetah
from google.protobuf import text_format
from brax.envs.halfcheetah import _SYSTEM_CONFIG as halfcheetah_config
from brax.envs.grasp import _SYSTEM_CONFIG as grasp_config
import brax

body_keys = {"torso":[0],
             "thig":[1,4],        
             "shin":[2,5],    
             "foot":[3,6]}

def modify_cheetah(cfg,specs):

    for spec,coeff in specs.items():
        if spec in body_keys:
            for i in body_keys[spec]:
                cfg.bodies[i].mass *= coeff
                cfg.bodies[i].colliders[0].capsule.radius *= coeff
                cfg.bodies[i].mass *= coeff
                cfg.bodies[i].colliders[0].capsule.radius *= coeff
        if spec == "gravity":
            cfg.gravity.z *= coeff
        if spec == "friction":
            cfg.friction *= coeff

    return cfg

class CustomHalfcheetah(Halfcheetah):
    def __init__(self, **kwargs):
        config = text_format.Parse(halfcheetah_config, brax.Config())
        if "env_spec" in kwargs:
            config = modify_cheetah(config,kwargs["env_spec"])
        self.sys = brax.System(config)

class CustomGrasp(Halfcheetah):
    def __init__(self, **kwargs):
        config = text_format.Parse(grasp_config, brax.Config())
        #if "env_spec" in kwargs:
        #    config = modify_cheetah(config,kwargs["env_spec"])
        self.sys = brax.System(config)

__envs__ = {
    'CustomHalfcheetah': CustomHalfcheetah,
    'CustomGrasp': CustomGrasp
}

def create_gym_env(env_name,
                   seed = 0,
                   batch_size = None,
                   episode_length = 1000,
                   action_repeat = 1,
                   backend = None,
                   auto_reset = True,
                   **kwargs):

    env = __envs__[env_name](**kwargs)
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)

test_cfgs = {
    "Halfcheetah":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahBigFoot":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.25,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahSmallFoot":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.,
      "foot": 0.75,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahBigThig":{"env_spec":{
      "torso": 1.,
      "thig": 1.25,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahSmallThig":{"env_spec":{
      "torso": 1.,
      "thig": 0.75,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahBigShin":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1.25,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahSmallShin":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 0.75,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahBigTorso":{"env_spec":{
      "torso": 1.25,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahSmallTorso":{"env_spec":{
      "torso": 0.75,
      "thig": 1.,
      "shin": 1.,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.,
    }},
    "HalfcheetahSmallGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 0.75,
      "friction": 1.,
    }},
    "HalfcheetahBigGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.25,
      "friction": 1.,
    }},
    "HalfcheetahSmallFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 0.75,
    }},
    "HalfcheetahBigFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.25,
    }},
    "HalfcheetahTinyGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 0.5,
      "friction": 1.,
    }},
    "HalfcheetahHugeGravity":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.5,
      "friction": 1.,
    }},
    "HalfcheetahTinyFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 0.5,
    }},
    "HalfcheetahHugeFriction":{"env_spec":{
      "torso": 1.,
      "thig": 1.,
      "shin": 1,
      "foot": 1.,
      "gravity": 1.,
      "friction": 1.5,
    }},
}