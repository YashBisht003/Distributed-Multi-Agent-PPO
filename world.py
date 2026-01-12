import numpy as np
import random
from typing import Dict, Tuple, Optional

# Cell types
EMPTY = 0
WALL = 1
SHELF = 2
DROP = 3

# Actions
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PICK = 5
DROP_ACT = 6


class World:
    def __init__(self, height=10, width=10, n_robots=2, n_shelves=8, n_drops=2):
        self.height = height
        self.width = width
        self.n_robots = n_robots
        self.n_shelves = n_shelves
        self.n_drops = n_drops

        self.grid = np.zeros((height, width), dtype=int)

        self.robots = {}
        self.carrying = {}
        self.robot_targets = {}

        self.drop_points = []
        self.shelves = []
        self.shelf_inventory = {}

        self.total_deliveries = 0
        self.collision_count = 0
        self.step_count = 0

        self._build_world()

    # -----------------------
    # World construction
    # -----------------------

    def _build_world(self):
        self._add_walls()
        self._place_shelves()
        self._place_drop_points()
        self._spawn_robots()

    def _add_walls(self):
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

    def _place_shelves(self):
        placed = 0
        while placed < self.n_shelves:
            x = random.randint(1, self.height - 2)
            y = random.randint(1, self.width - 2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = SHELF
                self.shelves.append((x, y))
                self.shelf_inventory[(x, y)] = random.randint(5, 20)
                placed += 1

    def _place_drop_points(self):
        placed = 0
        while placed < self.n_drops:
            x = random.randint(1, self.height - 2)
            y = random.randint(1, self.width - 2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = DROP
                self.drop_points.append((x, y))
                placed += 1

    def _spawn_robots(self):
        for i in range(self.n_robots):
            while True:
                x = random.randint(1, self.height - 2)
                y = random.randint(1, self.width - 2)
                if self.grid[x, y] == EMPTY and (x, y) not in self.robots.values():
                    rid = f"robot_{i}"
                    self.robots[rid] = (x, y)
                    self.carrying[rid] = False
                    self.robot_targets[rid] = None
                    break

    # -----------------------
    # Movement
    # -----------------------

    def is_valid_move(self, x, y):
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        if self.grid[x, y] in [WALL, SHELF]:
            return False
        return True

    def step(self, actions: Dict[str, int]):
        self.step_count += 1

        # Movement actions only
        movement = {}
        for rid, act in actions.items():
            movement[rid] = STAY if act in [PICK, DROP_ACT] else act

        # Propose moves
        proposed = {}
        for rid, act in movement.items():
            x, y = self.robots[rid]
            proposed[rid] = self._compute_next_position(x, y, act)

        # Resolve collisions
        final_pos, collisions = self._resolve_collisions(proposed)
        self.robots = final_pos

        rewards = {}
        info = {}

        for rid in self.robots:
            reward = -0.01
            agent_info = {
                "collision": rid in collisions,
                "picked": False,
                "dropped": False,
                "invalid_pick": False,
                "invalid_drop": False
            }

            if rid in collisions:
                reward -= 5

            pos = self.robots[rid]
            action = actions[rid]

            # PICK (from adjacent shelf)
            if action == PICK:
                shelf_pos = self._adjacent_shelf(pos)
                if shelf_pos and not self.carrying[rid]:
                    if self.shelf_inventory[shelf_pos] > 0:
                        self.carrying[rid] = True
                        self.shelf_inventory[shelf_pos] -= 1
                        reward += 1
                        agent_info["picked"] = True
                    else:
                        reward -= 0.1
                        agent_info["invalid_pick"] = True
                else:
                    reward -= 0.1
                    agent_info["invalid_pick"] = True

            # DROP
            elif action == DROP_ACT:
                if self.grid[pos] == DROP and self.carrying[rid]:
                    self.carrying[rid] = False
                    self.total_deliveries += 1
                    reward += 10
                    agent_info["dropped"] = True
                else:
                    reward -= 0.1
                    agent_info["invalid_drop"] = True

            rewards[rid] = reward
            info[rid] = agent_info

        return rewards, info

    def _compute_next_position(self, x, y, a):
        if a == UP:
            nx, ny = x - 1, y
        elif a == DOWN:
            nx, ny = x + 1, y
        elif a == LEFT:
            nx, ny = x, y - 1
        elif a == RIGHT:
            nx, ny = x, y + 1
        else:
            return x, y

        return (nx, ny) if self.is_valid_move(nx, ny) else (x, y)

    # -----------------------
    # Collision logic
    # -----------------------

    def _resolve_collisions(self, proposed):
        collided = set()

        # Same-cell collisions
        cell_map = {}
        for rid, pos in proposed.items():
            cell_map.setdefault(pos, []).append(rid)

        for pos, ids in cell_map.items():
            if len(ids) > 1:
                for r in ids:
                    collided.add(r)
                    self.collision_count += 1

        # Head-on swaps (ROBUST)
        ids = list(proposed.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                r1, r2 = ids[i], ids[j]
                if (proposed[r1] == self.robots[r2] and
                        proposed[r2] == self.robots[r1]):
                    collided.add(r1)
                    collided.add(r2)
                    self.collision_count += 1

        final_pos = {}
        for rid in proposed:
            final_pos[rid] = self.robots[rid] if rid in collided else proposed[rid]

        return final_pos, collided

    # -----------------------
    # Helpers
    # -----------------------

    def _adjacent_shelf(self, pos):
        x, y = pos
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.shelf_inventory:
                return (nx, ny)
        return None

    # -----------------------
    # Observations
    # -----------------------

    def get_local_observation(self, rid, view_size=5):
        x, y = self.robots[rid]
        half = view_size // 2
        obs = np.zeros((5, view_size, view_size), dtype=np.float32)

        for i in range(view_size):
            for j in range(view_size):
                wx, wy = x-half+i, y-half+j
                if wx<0 or wx>=self.height or wy<0 or wy>=self.width:
                    obs[0,i,j]=1
                    continue

                cell = self.grid[wx,wy]
                if cell in [WALL,SHELF]:
                    obs[0,i,j]=1
                if cell==DROP:
                    obs[1,i,j]=1

                for oid,(ox,oy) in self.robots.items():
                    if oid!=rid and (ox,oy)==(wx,wy):
                        obs[2,i,j]=1

        obs[3,half,half]=1

        if self.carrying[rid]:
            obs[4,:,:]=1

        return obs

    def get_global_state(self):
        state=[]
        for i in range(self.n_robots):
            rid=f"robot_{i}"
            if rid in self.robots:
                x,y=self.robots[rid]
                state += [x/self.height,y/self.width,float(self.carrying[rid])]
            else:
                state += [0,0,0]

        for s in self.shelves:
            state += [s[0]/self.height,s[1]/self.width,
                      self.shelf_inventory[s]/20]

        for d in self.drop_points:
            state += [d[0]/self.height,d[1]/self.width]

        return np.array(state,dtype=np.float32)

    # -----------------------
    # Utility
    # -----------------------

    def reset(self, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.__init__(self.height,self.width,
                      self.n_robots,self.n_shelves,self.n_drops)

    def get_metrics(self):
        return {
            "total_deliveries": self.total_deliveries,
            "collision_count": self.collision_count,
            "step_count": self.step_count,
            "deliveries_per_step":
                self.total_deliveries/max(1,self.step_count)
        }

    def render_ascii(self):
        disp=np.copy(self.grid).astype(str)
        disp[disp=='0']='.'
        disp[disp=='1']='#'
        disp[disp=='2']='S'
        disp[disp=='3']='D'

        for i,(rid,(x,y)) in enumerate(self.robots.items()):
            disp[x,y]=str(i) if not self.carrying[rid] else str(i).upper()

        return "\n".join("".join(r) for r in disp)


# -----------------------
# Manual test
# -----------------------

if __name__=="__main__":
    w=World(n_robots=2)
    print(w.render_ascii())

    for _ in range(10):
        acts={k:random.randint(0,6) for k in w.robots}
        r,i=w.step(acts)
        print("\n",w.render_ascii())
        print("Rewards:",r)
