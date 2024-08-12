import pygame as py
import numpy as np
import math
import nibabel as nib
import os

global nodes, nodesOriginal, sensativity

py.init()

# Caption
py.display.set_caption('Sick 3D Simulation')

# set up screen dimensions and origin
screen_Width = 1000
screen_Height = 800
size = (screen_Width, screen_Height)
screen = py.display.set_mode(size)

#scalars
sensitivity = 20  # out of 10 aim for
speed = 10
coord_scale = 20 #for changing nifti coords to be bigger to be closer to camera type sheh
down_sample_factor = 3
node_threshold = 70
node_size = 1


# Camera
camera_Position = [0, 0, -2000]

#axis
axis = [0,0,0]

#File Paths ________________________________________________________________________________________________
data_dir = "/Users/theochiang/Desktop/Radiology Internship 2024/python code/Registration_Torch_Model_Chiang/Training Data/Reference Images (w3_)"
image_nifti = nib.load(os.path.join(data_dir, 'w3_321_FLAIR<=T2_Flair_Axial_p2_5.nii'))
image_data = image_nifti.get_fdata()


# ____________________________________________________________________________________________________
# COORD CREATING FUNCTIONS
def makePrism(topRight, w, h, l):
    prismPoints = [
        topRight,
        [topRight[0] - w, topRight[1], topRight[2]],
        [topRight[0] - w, topRight[1] - h, topRight[2]],
        [topRight[0], topRight[1] - h, topRight[2]],
        [topRight[0], topRight[1], topRight[2] - l],
        [topRight[0] - w, topRight[1], topRight[2] - l],
        [topRight[0] - w, topRight[1] - h, topRight[2] - l],
        [topRight[0], topRight[1] - h, topRight[2] - l],
    ]
    return prismPoints


def makePolyPrism(base, h, direction):
    topBase = []
    for i in base:
        topBase.append(i)
    if direction == 'y':
        for i in range(len(base)):
            topBase.append([base[i][0], base[i][1] + h, base[i][2]])
    elif direction == 'z':
        for i in range(len(base)):
            topBase.append([base[i][0], base[i][1], base[i][2] + h])
    elif direction == 'x':
        for i in range(len(base)):
            topBase.append([base[i][0], base[i][1], base[i][2] + h])

    return (topBase)

#scales intensities also
def convert_to_nodes(image_data):
    image_nodes = []
    for x_dim in range(len(image_data)):
        for y_dim in range(len(image_data[x_dim])):
            for z_dim in range(len(image_data[x_dim][y_dim])):
                nodes_max = np.max(np.array(image_data))
                #normalized:
                node_intensity = round(image_data[x_dim][y_dim][z_dim] * (255/nodes_max), 5) 
                #scale up coords by coord_scale
                image_nodes.append((x_dim * coord_scale,y_dim*coord_scale,z_dim*coord_scale, node_intensity))
    return image_nodes

#already included in convert_to_nodes
def scale_intensity_values(nodes):
    nodes_max = np.max(np.array(nodes))

    for i in range(len(nodes)):
        #scale intensity
        nodes[i][3] = round(nodes[i][3] * (255/nodes_max), 5) 
        
        #scale up coords
        nodes[i][0:3] = [x * coord_scale for x in nodes[i][0:3]]
    return nodes

# ____________________________________________________________________________________________________


# deadzone


# NODES AND SHAPES______________________________________________________________________________________
nodes = [
    [50, 50, -150,110],  # 1
    [-50, 50, -150,100],
    [-50, -50, -150,200],
    [50, -50, -150,150],
    [50, 50, -250,200],
    [-50, 50, -250,200],
    [-50, -50, -250,180],
    [50, -50, -250,150],

]

def get_nodes(image_data):
    down_sampled_image_data = image_data[::down_sample_factor, ::down_sample_factor, ::down_sample_factor]
    nodes_converted = np.array(convert_to_nodes(down_sampled_image_data))
    nodes = [row for row in nodes_converted if row[3] >= node_threshold]
    return nodes

nodes = get_nodes(image_data)


colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'light_gray': (192, 192, 192),
    'dark_gray': (64, 64, 64),
    'sky': (124, 220, 249),
    'nitronGrey': (25, 25, 30)
}


# ________________________________________________________________________________________________________________



# Translate Coords to fit new centered origin
def trCoords(screenCoords):
    return (screenCoords[0] + (screen_Width / 2), screenCoords[1] + (screen_Height / 2))

# magnitude function
def mag(Coords):
    return math.sqrt(Coords[0] ** 2 + Coords[1] ** 2 + Coords[2] ** 2)


# Rotation Functions
def rotateX(Coords, angle, axis=[0, 0, 0]):
    translatedCoords = [points - axis for points, axis in zip(Coords, axis)]
    Ax = [
        [1, 0, 0],
        [0, math.cos(angle), -1 * math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)]
    ]

    # newCoords = [[],[],[]]
    # newCoords[0] = Coords[0] * Ax[0][0] + Coords[1] * Ax[0][1] + Coords[2] * Ax[0][2]
    # newCoords[1] = Coords[0] * Ax[1][0] + Coords[1] * Ax[1][1] + Coords[2] * Ax[1][2]
    # newCoords[2] = Coords[0] * Ax[2][0] + Coords[1] * Ax[2][1] + Coords[2] * Ax[2][2]
    newCoords = np.dot(Ax, translatedCoords)
    newCoords1 = [new + axis for new, axis in zip(newCoords, axis)]
    return (newCoords1)


def rotateY(Coords, angle, axis=[0, 0, 0]):
    translatedCoords = [points - axis for points, axis in zip(Coords, axis)]
    Ay = [
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-1 * math.sin(angle), 0, math.cos(angle)]
    ]

    # newCoords = [[],[],[]]
    # newCoords[0] = Coords[0]*Ay[0][0] + Coords[1]*Ay[0][1] + Coords[2]*Ay[0][2]
    # newCoords[1] = Coords[0]*Ay[1][0] + Coords[1]*Ay[1][1] + Coords[2]*Ay[1][2]
    # newCoords[2] = Coords[0]*Ay[2][0] + Coords[1]*Ay[2][1] + Coords[2]*Ay[2][2]
    newCoords = np.dot(Ay, translatedCoords)
    newCoords1 = [new + axis for new, axis in zip(newCoords, axis)]
    return (newCoords1)


def rotateZ(Coords, angle, axis=[0, 0, 0]):
    translatedCoords = [points - axis for points, axis in zip(Coords, axis)]
    Az = [
        [math.cos(angle), -1 * math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ]

    # newCoords = [[],[],[]]
    # newCoords[0] = Coords[0]*Az[0][0] + Coords[1]*Az[0][1] + Coords[2]*Az[0][2]
    # newCoords[1] = Coords[0]*Az[1][0] + Coords[1]*Az[1][1] + Coords[2]*Az[1][2]
    # newCoords[2] = Coords[0]*Az[2][0] + Coords[1]*Az[2][1] + Coords[2]*Az[2][2]
    newCoords = np.dot(Az, translatedCoords)
    newCoords1 = [new + axis for new, axis in zip(newCoords, axis)]
    return (newCoords1)





Pz = 1


def Project(Coords):
    # R3 to R3
    coords_projected = [1 * (Coords[0] - camera_Position[0]), 1 * (Coords[1] - camera_Position[1]),
               1 * (Coords[2] - camera_Position[2])]
    for i in range(len(coords_projected)):
        if (camera_Position[2] - Coords[2]) != 0:
            coords_projected[i] *= ((camera_Position[2] - near_clipz) / (camera_Position[2] - Coords[2]))

    return coords_projected



def writeText2D(words, Coords, color, size=15):
    x, y = Coords[0], Coords[1]
    font = py.font.Font('freesansbold.ttf', size)

    # create a text surface object,
    # on which text is drawn on it.
    text = font.render(words, True, color)

    # create a rectangular object for the
    # text surface object
    textRect = text.get_rect()

    # set the center of the rectangular object.
    textRect.center = (trCoords([x, y]))
    screen.blit(text, textRect)


# ____________________________________________________________________________________________________


# Main Loop
#default set mouse to playable state:
#py.event.set_grab(True)
py.mouse.set_visible(True)

while True:
    # clear screen
    screen.fill(colors["nitronGrey"])

    # Words
    # writeText2D('wsg gang', (0,0), colors['green'])
    writeText2D(('Camera X: ' + str(camera_Position[0])), ((-screen_Width / 2 + 80), (-screen_Height / 2 + 25)),
                colors['green'], 15)
    writeText2D(('Camera Y: ' + str(camera_Position[1])), ((-screen_Width / 2 + 80), (-screen_Height / 2 + 45)),
                colors['green'], 15)
    writeText2D(('Camera Z: ' + str(camera_Position[2])), ((-screen_Width / 2 + 80), (-screen_Height / 2 + 65)),
                colors['green'], 15)
    writeText2D(('Sensitivity: ' + str(sensitivity)), ((-screen_Width / 2 + 80), (-screen_Height / 2 + 90)),
                colors['green'], 15)

    writeText2D(('Threshold: ' + str(node_threshold)), ((-screen_Width / 2 + 80), (-screen_Height / 2 + 120)),
                colors['green'], 15)


    # camera clip - this worked somehow
    near_clipz = camera_Position[2] - 500


    # DRAWING NODES __ __  __ __  __ __  __ __  __ __  __ __  __ __##@#@#!@#!@#!@#(!@&^$(!&@#%$(!&@

    for t in range(len(nodes)):
        

        projectedCoords = Project(nodes[t][0:3])[0:2]
        if nodes[t][2] > camera_Position[2] and nodes[t][3] > node_threshold:
            py.draw.circle(screen, (nodes[t][3],nodes[t][3],nodes[t][3]), trCoords(projectedCoords), node_size)

    # _ __  __ __  __ __  __ __  __ __  __ __  __  __ __  __ __##@#@#!@#!@#!@#(!@&^$(!&@#%$(!&@

    # AUTOMATIC MOVEMENT

    # KEYBOARD INPUT
    keys = py.key.get_pressed()

    #player movement
    
    #increase threshold
    if keys[py.K_p]:
        node_threshold += 1
        
    if keys[py.K_o]:   
        node_threshold -= 1
    
    if keys[py.K_1]:
        node_size = 1
    
    if keys[py.K_2]:
        node_size = 2
        
    if keys[py.K_3]:
        node_size = 3
    
    if keys[py.K_4]:
        node_size = 4
        
    if keys[py.K_5]:
        node_size = 5
    
    if keys[py.K_6]:
        node_size = 6
        
    if keys[py.K_7]:
        node_size = 7
        
    if keys[py.K_8]:
        node_size = 8

    if keys[py.K_s]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateX(nodes[t][0:3], .001 * sensitivity, axis)

    if keys[py.K_w]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateX(nodes[t][0:3], -.001 * sensitivity,axis)

    if keys[py.K_a]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateY(nodes[t][0:3], .001 * sensitivity,axis)

    if keys[py.K_d]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateY(nodes[t][0:3], -.001 * sensitivity,axis)

    if keys[py.K_e]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateZ(nodes[t][0:3], .001 * sensitivity,axis)

    if keys[py.K_q]:
        # rotate clockwise x-axis
        for t in range(len(nodes)):
            for i in range(len(nodes[t][0:3])):
                nodes[t][0:3] = rotateZ(nodes[t][0:3], -.001 * sensitivity,axis)


    if keys[py.K_LEFT]:
        # rotate clockwise x-axis
        camera_Position[0] += (speed)

    if keys[py.K_RIGHT]:
        # rotate clockwise x-axis
        camera_Position[0] -= (speed)

    if keys[py.K_DOWN]:
        # rotate clockwise x-axis
        camera_Position[2] -= (speed)

    if keys[py.K_UP]:
        # rotate clockwise x-axis
        camera_Position[2] += (speed)

    if keys[py.K_SPACE]:
        # rotate clockwise x-axis
        camera_Position[1] += (speed)

    if keys[py.K_LSHIFT]:
        # rotate clockwise x-axis
        camera_Position[1] -= (speed)

        # esc mouse cursor
    if keys[py.K_ESCAPE]:
        # release cursor
        exit()




    # UPDATE DISPLAY
    py.display.update()

    if py.event.get(py.QUIT):
        exit()
