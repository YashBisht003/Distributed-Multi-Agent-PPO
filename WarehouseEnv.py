import numpy as np
from typing import Dict, Optional
from pettingzoo import ParallelEnv
from gymnasium import spaces
from world import (
    World,
    STAY, UP, DOWN, LEFT, RIGHT,
    PICK, DROP_ACT
)



class WarehouseEnv(ParallelEnv):


    metadata = {
        "name": "warehouse_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
            self,
            height: int = 12,
            width: int = 12,
            n_robots: int = 3,
            n_shelves: int = 8,
            n_drops: int = 2,
            max_steps: int = 500,
            view_size: int = 5,
            render_mode: Optional[str] = None
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.view_size = view_size
        self.render_mode = render_mode


        self.world = World(height, width, n_robots, n_shelves, n_drops)


        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents = self.possible_agents[:]


        self._action_spaces = {
            agent: spaces.Discrete(7) for agent in self.possible_agents
        }


        self._observation_spaces = {
            agent: spaces.Box(
                low=0, high=1,
                shape=(5, view_size, view_size),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.current_step = 0

    @property
    def observation_space(self):
        return self._observation_spaces[self.agents[0]]

    @property
    def action_space(self):
        return self._action_spaces[self.agents[0]]

    def observation_spaces(self, agent):
        return self._observation_spaces[agent]

    def action_spaces(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        if seed is not None:
            np.random.seed(seed)

        self.world.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.current_step = 0


        observations = {
            agent: self.world.get_local_observation(agent, self.view_size)
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):

        self.current_step += 1


        rewards, infos = self.world.step(actions)


        observations = {
            agent: self.world.get_local_observation(agent, self.view_size)
            for agent in self.agents
        }


        terminations = {agent: False for agent in self.agents}


        truncated = self.current_step >= self.max_steps
        truncations = {agent: truncated for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):

        if self.render_mode == "human":
            print("\n" + "=" * 50)
            print(f"Step: {self.current_step}")
            print(self.world.render_ascii())
            print(f"Deliveries: {self.world.total_deliveries}")
            print(f"Collisions: {self.world.collision_count}")
            print("=" * 50)
        elif self.render_mode == "rgb_array":
            # TODO: implement pygame rendering
            pass
        return None

    def close(self):

        pass

    def get_global_state(self):

        return self.world.get_global_state()



if __name__ == "__main__":

    env = WarehouseEnv(
        height=10,
        width=10,
        n_robots=2,
        max_steps=100,
        render_mode="human"
    )

    print("Testing WarehouseEnv...")
    print(f"Agents: {env.possible_agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")


    obs, info = env.reset(seed=42)
    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"Observation shape: {obs['robot_0'].shape}")


    for step in range(20):

        actions = {
            agent: env.action_space.sample()
            for agent in env.agents
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        env.render()

        print(f"Rewards: {rewards}")
        print(f"Infos: {infos}")

        
        if any(terms.values()) or any(truncs.values()):
            print("Episode ended!")
            break

    print("\n" + "=" * 50)
    print("Final metrics:", env.world.get_metrics())
    print("=" * 50)