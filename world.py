import numpy as np
from typing import Dict


EMPTY = 0
WALL = 1
SHELF = 2
DROP = 3


STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PICK = 5
DROP_ACT = 6


class World:

    def __init__(self, height=12, width=12, n_robots=3):

        self.height = height
        self.width = width
        self.n_robots = n_robots

        self.grid = np.zeros((height, width), dtype=int)

        self.robots = {}
        self.carrying = {}

        self.drop_points = []
        self.shelves = []
        self.shelf_inventory = {}

        self.total_deliveries = 0
        self.collision_count = 0
        self.step_count = 0

        self._build_fixed_world()


    def _build_fixed_world(self):

        self._add_walls()
        self._add_corridors()
        self._place_shelves()
        self._place_drops()
        self._spawn_robots()

    def _add_walls(self):
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

    def _add_corridors(self):
        """
        Create internal walls to form warehouse lanes
        """
        for r in [4, 7]:
            for c in range(2, self.width-2):
                self.grid[r, c] = WALL

        
        for r in range(2, self.height-2):
            self.grid[r, 6] = WALL

       
        self.grid[4, 3] = EMPTY
        self.grid[7, 8] = EMPTY
        self.grid[6, 6] = EMPTY

    def _place_shelves(self):
        """
        Top zone: shelves
        """
        self.shelves = [
            (1,2),(1,4),(1,8),(1,10),
            (2,3),(2,7),(3,2),(3,9)
        ]

        for s in self.shelves:
            self.grid[s] = SHELF
            self.shelf_inventory[s] = 30

    def _place_drops(self):
        """
        Bottom zone: drop stations
        """
        self.drop_points = [(10,3),(10,9)]

        for d in self.drop_points:
            self.grid[d] = DROP

    def _spawn_robots(self):
        """
        Spawn robots in central region
        """
        starts = [(6,2),(6,9),(8,5)]

        for i in range(self.n_robots):
            rid = f"robot_{i}"
            self.robots[rid] = starts[i]
            self.carrying[rid] = False


    def step(self, actions: Dict[str, int]):

        self.step_count += 1

        proposed = {}

        for rid, act in actions.items():

            x, y = self.robots[rid]

            if act == UP:    nx, ny = x-1, y
            elif act == DOWN:nx, ny = x+1, y
            elif act == LEFT:nx, ny = x, y-1
            elif act == RIGHT:nx, ny = x, y+1
            else:            nx, ny = x, y

            if self._valid(nx, ny):
                proposed[rid] = (nx, ny)
            else:
                proposed[rid] = (x, y)

        final = {}
        collisions = set()

        for rid, pos in proposed.items():
            if list(proposed.values()).count(pos) > 1:
                collisions.add(rid)
                final[rid] = self.robots[rid]
                self.collision_count += 1
            else:
                final[rid] = pos

        self.robots = final

        rewards = {}
        info = {}

        for rid in self.robots:

            r = -0.02  # living cost
            x, y = self.robots[rid]

            picked = False
            dropped = False

            
            if actions[rid] == PICK:
                shelf = self._adjacent_shelf((x,y))

                if shelf and not self.carrying[rid]:
                    self.carrying[rid] = True
                    self.shelf_inventory[shelf] -= 1
                    r += 3.0
                    picked = True
                else:
                    r -= 0.3

            if actions[rid] == DROP_ACT:

                if (x,y) in self.drop_points and self.carrying[rid]:
                    self.carrying[rid] = False
                    self.total_deliveries += 1
                    r += 20.0
                    dropped = True
                else:
                    r -= 0.3

            # collision penalty
            if rid in collisions:
                r -= 2.0

            rewards[rid] = r
            info[rid] = {
                "picked": picked,
                "dropped": dropped,
                "collision": rid in collisions
            }

        return rewards, info


    def get_local_observation(self, rid, view_size=5):

        x, y = self.robots[rid]
        half = view_size // 2

        obs = np.zeros((5, view_size, view_size), dtype=np.float32)

        for i in range(view_size):
            for j in range(view_size):

                wx = x-half+i
                wy = y-half+j

                if wx<0 or wx>=self.height or wy<0 or wy>=self.width:
                    obs[0,i,j] = 1
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

   
    def _adjacent_shelf(self, pos):

        x,y = pos
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            p=(x+dx,y+dy)
            if p in self.shelves:
                return p
        return None

    def _valid(self,x,y):
        return self.grid[x,y] not in [WALL,SHELF]

   

    def reset(self):
        self.__init__(self.height,self.width,self.n_robots)

    def get_global_state(self):

        state=[]

        for i in range(self.n_robots):
            rid=f"robot_{i}"
            x,y=self.robots[rid]
            state+=[x/self.height,y/self.width,float(self.carrying[rid])]

        for s in self.shelves:
            state+=[s[0]/self.height,s[1]/self.width,
                    self.shelf_inventory[s]/30]

        for d in self.drop_points:
            state+=[d[0]/self.height,d[1]/self.width]

        return np.array(state,dtype=np.float32)

    def get_metrics(self):

        return {
            "total_deliveries":self.total_deliveries,
            "collision_count":self.collision_count,
            "step_count":self.step_count,
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
            disp[x,y]=str(i)

        return "\n".join("".join(r) for r in disp)

