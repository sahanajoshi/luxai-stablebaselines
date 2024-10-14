import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
from kaggle_environments import make
from lux.game import Game
from kaggle_environments import make
from collections import OrderedDict
from enum import Enum

class ActionCT(Enum):
    bw = 0
    bc = 1
    r = 2
valToActionCT = {actionct.value: actionct.name for actionct in ActionCT}

class LuxAIEnv(gym.Env):
    """Custom Environment that follows gym interface for Lux AI Season 1"""

    def __init__(self, ):
        super().__init__()
        # init OpenAI variables
        # we create self.observation_state(OpenAI convention) to encapsulate the game state which is a common representation of the state of the game map, and each
        # players stats, i.e, observation_space encapsulates ALL the data present in self.state, except in different formats, conducive for training. This class adds
        # extra methods which help convert self.state into self.observation_space.
        self.observation_space = spaces.Dict({
            "game_map": spaces.Box(low=np.zeros((4, 32, 32), dtype=int), high=np.concatenate((4000*np.ones((3, 32, 32), dtype=int), 6*np.ones((1, 32, 32), dtype=int)))), # 3 reources + 1 road level 
            "player_units": spaces.Box(low=np.zeros((5, 32, 32), dtype=int), high=np.concatenate((np.ones((1, 32, 32), dtype=int), 3*np.ones((1, 32, 32), dtype=int), 4000*np.ones((3, 32, 32), dtype=int)))), # always player 0
            "player_citytiles": spaces.Box(low=np.zeros((2, 32, 32), dtype=int), high=np.concatenate((10*np.ones((1, 32, 32), dtype=int), 1024*np.ones((1, 32, 32), dtype=int)))),
            "player_rp": spaces.Discrete(201,start=0),
            "opponent_units": spaces.Box(low=np.zeros((5, 32, 32), dtype=int), high=np.concatenate((np.ones((1, 32, 32), dtype=int), 3*np.ones((1, 32, 32), dtype=int), 4000*np.ones((3, 32, 32), dtype=int)))),
            "opponent_citytiles": spaces.Box(low=np.zeros((2, 32, 32), dtype=int), high=np.concatenate((10*np.ones((1, 32, 32), dtype=int), 1024*np.ones((1, 32, 32), dtype=int)))),
            "opponent_rp": spaces.Discrete(201,start=0)
        })
        # sizect = 3 * np.ones([32, 32])
        sizect = 3 * np.ones(1024)
        self.action_space = spaces.MultiDiscrete(sizect)

        # init Game variables
        self.game_env = None

        # init LuxAI variables
        self.kaggle_env = None
        self.trainer = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}

        # init LuxAI variables
        self.kaggle_env = make("lux_ai_2021")
        self.trainer = self.kaggle_env.train([None, "random_agent"])
        kaggle_state = self.trainer.reset()  # note self.trainer.reset() gives same data as self.kaggle_env.state

        # init Game variables using kaggle_env
        self.game_env = Game()
        self.game_env._initialize(self.kaggle_env.state[0]["observation"]["updates"])
        self.game_env._update(self.kaggle_env.state[0]["observation"]["updates"][2:])
        self.game_env.id = 0

        # There is an inconsistency between how Lux AI stores the game state and how OpenAI baselines expects it. Specifically, self.env.state (Lux conventions)
        # is a list of dicts, one for each player, dict format being dict_keys(['action', 'reward', 'info', 'observation', 'status']), while both players have attr.
        # like action, status always filled, the game map information is available only in player 0's (first item in list) observation.updates.
        observation = self._get_obs_from_state(kaggle_state)
        return observation, info


    def step(self, action):
        # convert from OpenAI version to LuxAI
        action_luxai = self._get_action(action)

        # use action_luxai to update self.trainer and self.game_env
        # calling self.trainer.step() also updates self.kaggle_env.state
        obs, reward, done, info = self.trainer.step(action_luxai)

        # update self.game_env using self.kaggle_env.state
        self.game_env._update(self.kaggle_env.state[0]["observation"]["updates"][2:])

        # conversion from LuxAI obs to OpenAI (gym) obs
        obs = self._get_obs_from_state(obs)

        return obs, reward, done, False, info

    def _get_obs_from_state(self, state):
        updates = state['updates']
        _map = np.zeros((4, 32, 32), dtype=np.float32)
        _player_citytiles = np.zeros((2, 32, 32), dtype=np.float32)
        _player_units = np.zeros((5, 32, 32), dtype=np.float32)
        _opponent_units = np.zeros((5, 32, 32), dtype=np.float32)
        _opponent_citytiles = np.zeros((2, 32, 32), dtype=np.float32)
        _player_rp = np.ones((1, ), dtype=np.float32)
        _opponent_rp = np.ones((1, ), dtype=np.float32)

        citytile_dict = {}
        
        for u in updates:
            strs = u.split(" ")
            input_identifier = strs[0]
            if input_identifier == 'r': # resource
                if strs[1] == 'wood':
                    _map[0, int(strs[2]), int(strs[3])] = int(strs[4])
                elif strs[1] == 'coal':
                    _map[1, int(strs[2]), int(strs[3])] = int(strs[4])
                else: # uranium
                    _map[2, int(strs[2]), int(strs[3])] = int(strs[4])
            elif input_identifier == 'ccd': # road
                _map[3, int(strs[1]), int(strs[2])] = int(strs[3])
            elif input_identifier == 'ct': # citytile
                team_id = int(strs[1])
                if team_id == 0:
                    _player_citytiles[0, int(strs[3]), int(strs[4])] = int(strs[5]) # set cooldown
                    citytile_dict[(int(strs[3]), int(strs[4]))] = 0 # add to dict with val as team
                else:
                    _opponent_citytiles[0, int(strs[3]), int(strs[4])] = int(strs[5])
                    citytile_dict[(int(strs[3]), int(strs[4]))] = 1
            elif input_identifier == 'u':
                if (int(strs[4]), int(strs[5])) in citytile_dict:
                    team_id == int(strs[2])
                    if team_id == 0:
                        _player_citytiles[1, int(strs[4]), int(strs[5])] += 1 # keeps track of how many units on a citytile
                    else:
                        _opponent_citytiles[1, int(strs[4]), int(strs[5])] += 1
                else:
                    team_id == int(strs[2])
                    if team_id == 0:
                        _player_units[0, int(strs[4]), int(strs[5])] = int(strs[1]) # unittype
                        _player_units[1, int(strs[4]), int(strs[5])] = int(strs[6]) # cooldown
                        _player_units[2, int(strs[4]), int(strs[5])] = int(strs[7]) # wood
                        _player_units[3, int(strs[4]), int(strs[5])] = int(strs[8]) # coal
                        _player_units[4, int(strs[4]), int(strs[5])] = int(strs[9]) # uranium
                    else: # opponent
                        _opponent_units[0, int(strs[4]), int(strs[5])] = int(strs[1]) # unittype
                        _opponent_units[1, int(strs[4]), int(strs[5])] = int(strs[6]) # cooldown
                        _opponent_units[2, int(strs[4]), int(strs[5])] = int(strs[7]) # wood
                        _opponent_units[3, int(strs[4]), int(strs[5])] = int(strs[8]) # coal
                        _opponent_units[4, int(strs[4]), int(strs[5])] = int(strs[9]) # uranium
            elif input_identifier == 'rp':
                team_id = int(strs[1])
                if team_id == 0:
                    _player_rp[0] = int(strs[2])
                else:
                    _opponent_rp[0] = int(strs[2])

        obs = OrderedDict({
            "game_map": _map, 
            "player_units": _player_units,
            "player_citytiles": _player_citytiles,
            "opponent_units": _opponent_units,
            "opponent_citytiles": _opponent_citytiles,
            "player_rp": _player_rp,
            "opponent_rp": _opponent_rp
        })

        return obs

    def _get_action(self, action):
        # get citytile actions
        ret_actions = []

        action = np.reshape(action, (32, 32))

        for x in range(32):
            for y in range(32):
                ret_actions.append("{} {} {}".format(valToActionCT[action[x][y]], x, y))

        return ret_actions
