"""Minimal standalone PyBullet example.
"""
import pybullet as p
import pybullet_data
from time import sleep
from pybullet_utils import urdfEditor

def main():
    PYB_CLIENT = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8, physicsClientId=PYB_CLIENT)
    p.setRealTimeSimulation(0, physicsClientId=PYB_CLIENT)
    p.setTimeStep(1/240, physicsClientId=PYB_CLIENT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=PYB_CLIENT)
    p.loadURDF("plane.urdf", physicsClientId=PYB_CLIENT)
    #p.loadURDF("duck_vhacd.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("teddy_vhacd.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [0, 0, 1.032], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [0.1, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [0.2, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [0.2, 0, 1], [90,90,0,0], physicsClientId=PYB_CLIENT)
    #p.loadURDF("cube_small.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("samurai.urdf", physicsClientId=PYB_CLIENT)
    #p.loadURDF("cube_no_rotation.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("sphere2.urdf", [0, 0, 1], physicsClientId=PYB_CLIENT)

    minx = -1.5
    maxx = 1.5
    miny = -1.5
    maxy = 1.5
    #p.loadURDF("block.urdf", [-1.45,1.59,1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [-1.59,1.45,1], useFixedBase=True, physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [-1.45,1.59,1], physicsClientId=PYB_CLIENT)
    #p.loadURDF("block.urdf", [-1.45,1.48,1], [90,90,0,0], physicsClientId=PYB_CLIENT)
    
    y = -1.59
    while y < 1.6:
        x = -1.45
        while x < 1.5:
            p.loadURDF("block.urdf", [x, y, 1], physicsClientId=PYB_CLIENT)
            x += 0.1
        y += 3.18

    x = -1.59
    while x < 1.6:
        y = -1.45
        while y < 1.5:
            p.loadURDF("block.urdf", [x, y, 1], [90,90,0,0], physicsClientId=PYB_CLIENT)
            y += 0.1
        x += 3.18
    
    for _ in range(240*300):
        p.stepSimulation(physicsClientId=PYB_CLIENT)
        sleep(1/(240*10))
    
if __name__ == "__main__":
    main()
