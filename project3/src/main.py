from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import os
import numpy as np

g_distance = 5
g_azimuth = np.radians(45)
g_elevation = np.radians(45)
g_perspective = True
g_trans = glm.vec3(0.,0.,0.)
g_vertices = np.zeros(0, 'float32')
g_indices = glm.array.zeros(0, glm.float32)
hierarchical_mode = False
file_dropped = 0
solid_mode = False
rendering_mode = 0
g_joints = []
g_nodes = {}
g_rate = 0.0
g_frames = 0
animate_mode = False
cid = {}
cid['XPOSITION'] = 1
cid['YPOSITION'] = 2
cid['ZPOSITION'] = 3
cid['XROTATION'] = 4
cid['YROTATION'] = 5
cid['ZROTATION'] = 6
cid['3'] = 0
cid['6'] = 0

g_u =     glm.vec3(0.,0.,0.,)
g_v =     glm.vec3(0.,0.,0.,)
g_w =     glm.vec3(0.,0.,0.,)

lastX = 0.
lastY = 0.
dragging_left = False
dragging_right = False
YJ_SITE = 's i t e '

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''
g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;
uniform vec3 color;

out vec4 FragColor;

uniform vec3 view_pos;

struct Light {    
    vec3 position;
    vec3 color;
};

struct Material {
    vec3 color;
    float shininess;
};

vec3 calculateColor(Light light, Material material) {

    // light components
    vec3 light_ambient = 0.1 * light.color;
    vec3 light_diffuse = light.color;
    vec3 light_specular = light.color;

    // material components
    vec3 material_ambient = material.color;
    vec3 material_diffuse = material.color;
    vec3 material_specular = light.color;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light.position - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specular = spec * light_specular * material_specular;

    return (ambient + diffuse + specular);
}

void main()
{
    // light and material properties
    Material material;
    material.color = color;
    material.shininess = 32.0;

    Light light1;
    light1.position = vec3(-5,2,0);
    light1.color = vec3(1,.3,.3);

    Light light2;
    light2.position = vec3(3, 2, 4);
    light2.color = vec3(.3,1,.3);

    Light light3;
    light3.position = vec3(3, 2, -4);
    light3.color = vec3(.3,.3,1);

    Light light4;
    light4.position = vec3(0,10,0);
    light4.color = vec3(1,1,1);

    vec3 color1 = calculateColor(light1, material);
    vec3 color2 = calculateColor(light2, material);
    vec3 color3 = calculateColor(light3, material);
    vec3 color4 = calculateColor(light4, material);

    vec3 color_sum = vec3(min(color1.r + color2.r + color3.r + color4.r, 1), min(color1.g + color2.g + color3.g + color4.g, 1), min(color1.b + color2.b + color3.b + color4.b, 1));
    FragColor = vec4(color_sum, 1.);
}
'''

class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()
        self.draw_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
            self.draw_transform = self.parent.get_global_transform() * self.link_transform_from_parent
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform
            self.draw_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_draw_transform(self):
        return self.draw_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

class Joint:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.frame_motions = []
        self.channels = []
        self.offset = []
    def append_offset(self, offset):
        self.offset.append(offset)
    def append_channels(self, channels):
        if cid[channels] == 0:
            return
        self.channels.append(cid[channels])
    def append_frame_motions(self, fm):
        tmp = []
        for e in fm:
            tmp.append(e)
        self.frame_motions.append(tmp)

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def mouse_button_callback(window, button, action, mods):
    global dragging_left, dragging_right, lastX, lastY, startX, startY
    # Orbit
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            dragging_left = True
            startX, startY = glfwGetCursorPos(window)
            lastX, lastY = startX, startY
        elif action == GLFW_RELEASE:
            dragging_left = False
    # Pan
    if button == GLFW_MOUSE_BUTTON_RIGHT:
        if action == GLFW_PRESS:
            dragging_right = True
            startX, startY = glfwGetCursorPos(window)
            lastX, lastY = startX, startY
        elif action == GLFW_RELEASE:
            dragging_right = False

def mouse_callback(window, xpos, ypos):
    global lastX, lastY, g_azimuth, g_elevation, dragging_left, dragging_right, g_trans
    # Orbit
    if dragging_left:
        xoffset = xpos - lastX
        yoffset = lastY - ypos

        if np.cos(g_elevation) > 0:
            g_azimuth -= xoffset / 500
        else:
            g_azimuth += xoffset / 500
        g_elevation -= yoffset / 500

        lastX = xpos
        lastY = ypos
    # Pan
    if dragging_right:
        xoffset = xpos - lastX
        yoffset = lastY - ypos
        g_trans -= g_u * xoffset / 500
        g_trans -= g_v * yoffset / 500

        lastX = xpos
        lastY = ypos

def key_callback(window, key, scancode, action, mods):
    global g_perspective, hierarchical_mode, solid_mode, rendering_mode, animate_mode
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                g_perspective = not g_perspective
            elif key==GLFW_KEY_H:
                hierarchical_mode = True
            elif key==GLFW_KEY_Z:
                solid_mode = not solid_mode
            elif key==GLFW_KEY_1:
                rendering_mode = 1
            elif key==GLFW_KEY_2:
                rendering_mode = 2
            elif key==GLFW_KEY_SPACE:
                animate_mode = True

def scroll_callback(window, x_offset, y_offset):
    global g_distance
    g_distance *= np.power(2, y_offset/10)

def parse_file(path):
    global g_rate, g_frames
    filename = os.path.basename(path)
    f = open(path)
    lines = f.readlines()

    fields = []

    for line in lines:
        fields.extend(line.strip().split())
        
    hierarchy_section = False
    motion_section = False

    joint_section = False
    root_section = False
    end_section = False

    channels_section = False
    offset_section = False

    name_section = False

    frames_section = False
    time_section = False

    stack = []              # temporary stack
    joint_list = []          # all joints
    tmp_offset = []          # for offset list and channels list
    tmp_channels = []        # for offset list and channels list
    motion_index = -1
    site_count = 0

    for i in range(len(fields)):
        field = fields[i]
        if field == "HIERARCHY":
            hierarchy_section = True
            continue
        elif field == "MOTION":
            hierarchy_section = False
            motion_section = True
            continue
        elif field == "ROOT":
            root_section = True
            name_section = True
            end_section = False
            continue
        elif field == "JOINT":
            root_section = False
            joint_section = True
            name_section = True
            end_section = False
            continue
        elif field == "End":
            root_section = False
            joint_section = False
            offset_secrion = False
            channels_section = False
            name_section = True
            end_section = True
            continue
        elif field == "OFFSET":
            name_section = False
            channels_section = False
            offset_section = True
            tmp_offset = []
            continue
        elif field == "CHANNELS":
            offset_section = False
            channels_section = True
            tmp_channels = []
            continue
        elif field == "{":
            continue
        elif field == "}":
            if end_section:
                end_section = False
            stack.pop()
            continue
        elif field == "Frames:":
            frames_section = True
            continue
        elif field == "Frame":
            continue
        elif field == "Time:":
            frames_section = False
            time_section = True
            continue

        if hierarchy_section:
            if root_section:
                if name_section:
                    joint_list.append(Joint(None, field))
                    stack.append(joint_list[-1])
                elif offset_section:
                    tmp_offset.append(float(field))
                    stack[-1].append_offset(float(field))
                elif channels_section:
                    tmp_channels.append(field)
                    stack[-1].append_channels(field)
                else:
                    print("root- offset, channel error")
            elif joint_section:
                if name_section:
                    joint_list.append(Joint(stack[-1], field))
                    stack.append(joint_list[-1])
                elif offset_section:
                    tmp_offset.append(float(field))
                    stack[-1].append_offset(float(field))
                elif channels_section:
                    tmp_channels.append(field)
                    stack[-1].append_channels(field)
                else:
                    print("joint- offset, channel error")
            elif end_section:
                if name_section:
                    joint_list.append(Joint(stack[-1], YJ_SITE + str(site_count)))
                    site_count += 1
                    stack.append(joint_list[-1])
                elif offset_section:
                    tmp_offset.append(float(field))
                    stack[-1].append_offset(float(field))
                else:
                    print("end - name offset error")
        elif motion_section:
            if frames_section:
                g_frames = int(field)
            elif time_section:
                g_rate = float(field)
                motion_index = i
                break
        else:
            print("root, joint error")

    motion_index += 1
    cnt = 0
    for joint in joint_list:
        cnt += len(joint.channels)
    for frame_idx in range(g_frames):
        for joint in joint_list:
            channels = []
            for col in range(len(joint.channels)):
                channels.append(float(fields[motion_index]))
                motion_index += 1
            joint.append_frame_motions(channels)

    print('- File name:', filename)
    print('- Number of frames:', g_frames)
    print('- FPS:', g_rate)
    print('- Number of joints:', len(joint_list) - site_count)
    print('- List of all joint names:')

    for joint in joint_list:
        if joint.name[0:(len(YJ_SITE))] == YJ_SITE:
            continue
        print('\t' + joint.name)

    return joint_list

def channels_to_j(channels, rot):
    if (len(channels) == 0):
        return glm.mat4()
    rotation = glm.mat4()
    for i in range(len(channels)):
        if channels[i] == cid["XPOSITION"]:
            rotation = rotation * glm.translate(glm.vec3(rot[i],0,0))
        if channels[i] == cid["YPOSITION"]:
            rotation = rotation * glm.translate(glm.vec3(0,rot[i],0))
        if channels[i] == cid["ZPOSITION"]:
            rotation = rotation * glm.translate(glm.vec3(0,rot[i],1))
        if channels[i] == cid["XROTATION"]:
            rotation = rotation * glm.rotate(glm.radians(rot[i]), glm.vec3(1,0,0))
        if channels[i] == cid["YROTATION"]:
            rotation = rotation * glm.rotate(glm.radians(rot[i]), glm.vec3(0,1,0))
        if channels[i] == cid["ZROTATION"]:
            rotation = rotation * glm.rotate(glm.radians(rot[i]), glm.vec3(0,0,1))
    return rotation

def offset_to_distance(offset):
    sum = 0.0
    for i in range(3):
        sum += offset[i] * offset[i]
    sum = glm.sqrt(sum)
    return sum

def joints_to_nodes(joints):
    Nodes = {}
    for joint in joints:
        link = glm.translate(glm.vec3(joint.offset[0], joint.offset[1], joint.offset[2]))
        if joint.parent is None:
            Nodes[joint.name] = Node(None, link, glm.scale((.05,.05,.05)) * glm.translate(glm.vec3(0,1,0)), glm.vec3(1,1,1))
            continue

        parent_node = Nodes[joint.parent.name]
        dist = offset_to_distance(joint.offset)
        offset = glm.vec3(joint.offset[0], joint.offset[1], joint.offset[2])
        up = glm.vec3(0.,1.,0.)

        angle = glm.acos(glm.dot(offset, up) / glm.sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z))

        if offset.x == 0 and offset.z == 0:
            if offset.y >= 0:
                angle = 0
            else:
                angle = np.pi
        
        up_rotation = glm.rotate(angle, glm.normalize(glm.cross(up, offset)))
        if angle == 0:
            up_rotation = glm.mat4()
        elif angle == np.pi:
            up_rotation = glm.scale(glm.vec3(1.,-1.,1))
        Nodes[joint.name] = Node(parent_node, link,  glm.translate(-0.5 * offset) * up_rotation * glm.scale((.05, dist / 2, .05)), glm.vec3(1, 1, 1))
    return Nodes

def drop_callback(window, paths):
    global g_joints, g_nodes, file_dropped, animate_mode
    animate_mode = False
    for path in paths:
        g_joints = parse_file(path)
        g_nodes = joints_to_nodes(g_joints)
    file_dropped = True

def prepare_vao_grid():
    list = []
    s = -20
    e = 20
    for i in range(s, e):
        for j in range(s, e):
            list.extend([(i) * .2,      0., (j) * .2,       0., 1., 0.])
            list.extend([(i + 1) * .2,  0., (j) * .2,       0., 1., 0.])
            list.extend([(i + 1) * .2,  0., (j + 1) * .2,   0., 1., 0.])
            list.extend([(i) * .2,      0., (j + 1) * .2,   0., 1., 0.])
    tmp = np.array(list, dtype=np.float32)
    vertices = glm.array(tmp)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0, 0.,0.,1.,# x-axis start
         1.0, 0.0, 0.0, 0.,0.,1.,# x-axis end 
         0.0, 0.0, 0.0, 0.,1.,0.,# y-axis start
         0.0, 1.0, 0.0, 0.,1.,0.,# y-axis end 
         0.0, 0.0, 0.0, 1.,0.,0.,# z-axis start
         0.0, 0.0, 1.0, 1.,0.,0.,# z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position      normal
        -1 ,  1 ,  1 ,  0, 0, 1, # v0
         1 , -1 ,  1 ,  0, 0, 1, # v2
         1 ,  1 ,  1 ,  0, 0, 1, # v1

        -1 ,  1 ,  1 ,  0, 0, 1, # v0
        -1 , -1 ,  1 ,  0, 0, 1, # v3
         1 , -1 ,  1 ,  0, 0, 1, # v2

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  1 , -1 ,  0, 0,-1, # v5
         1 , -1 , -1 ,  0, 0,-1, # v6

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 , -1 , -1 ,  0, 0,-1, # v6
        -1 , -1 , -1 ,  0, 0,-1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 , -1 ,  1 ,  0,-1, 0, # v3
         1 , -1 , -1 ,  0,-1, 0, # v6
         1 , -1 ,  1 ,  0,-1, 0, # v2

        -1 , -1 ,  1 ,  0,-1, 0, # v3
        -1 , -1 , -1 ,  0,-1, 0, # v7
         1 , -1 , -1 ,  0,-1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 ,  1 ,  1, 0, 0, # v2
         1 , -1 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 , -1 , -1 , -1, 0, 0, # v7
        -1 , -1 ,  1 , -1, 0, 0, # v3

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  1 , -1 , -1, 0, 0, # v4
        -1 , -1 , -1 , -1, 0, 0, # v7
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_line():
    # prepare vertex data (in main memory)
    # 2 vertices for 1 line
    vertices = glm.array(glm.float32,
        # position      normal
         0 ,  1 ,  0 ,  0, 0, 1, # v0
         0 , -1 ,  0 ,  0, 0, 1, # v1
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_obj(vertices):

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, MVP_loc, M, M_loc, view_pos, view_pos_loc, color_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, 1., 1., 1.)
    glDrawArrays(GL_LINES, 0, 6)


def draw_grid(vao, MVP, MVP_loc, M, M_loc, view_pos, view_pos_loc, color_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, .5, .5, .5)
    for i in range(1600):
        glDrawArrays(GL_LINE_LOOP, i * 4, 4)

def draw_node(vao, node, VP, MVP_loc, M, M_loc, view_pos, view_pos_loc, color_loc):
    global rendering_mode
    if node.parent is None:
        return
    M = node.get_draw_transform() * node.get_shape_transform()
    MVP = VP * M
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, color.r, color.g, color.b)

    if rendering_mode == 1:
        glDrawArrays(GL_LINES, 0, 2)
    elif rendering_mode == 2:
        glDrawArrays(GL_TRIANGLES, 0, 36)

def draw_bvh(nodes, joints, vao_cube, vao_line, VP, MVP_loc, M_loc, view_pos, view_pos_loc, color_loc):
    global rendering_mode, animate_mode
    if rendering_mode == 1:
        vao = vao_line
    elif rendering_mode == 2:
        vao = vao_cube
    else:
        return

    I = glm.mat4()
    if animate_mode:
        t = glfwGetTime()
        frame = int(t // g_rate) % g_frames
        for joint in joints:
            nodes[joint.name].set_joint_transform(channels_to_j(joint.channels, joint.frame_motions[frame]))
    nodes[joints[0].name].update_tree_global_transform()

    for joint in joints:
        draw_node(vao, nodes[joint.name], VP, MVP_loc, I, M_loc, view_pos, view_pos_loc, color_loc)
    
def main():
    global g_u, g_v, g_w, g_perspective, file_dropped, hierarchical_mode, rendering_mode
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1000, 1000, '2020002960', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)

    # set the mouse callback function
    glfwSetCursorPosCallback(window, mouse_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    color_loc = glGetUniformLocation(shader_program, 'color')    

    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()
    vao_cube = prepare_vao_cube()
    vao_line = prepare_vao_line()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        if solid_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glUseProgram(shader_program)

        # Initialize camera position and target
        if np.cos(g_elevation) > 0:
            g_v = glm.vec3(0.,1.,0.)
        else:
            g_v = glm.vec3(0.,-1.,0)

        g_w = glm.normalize(glm.vec3(np.sin(g_azimuth) * np.cos(g_elevation), np.sin(g_elevation), np.cos(g_azimuth) * np.cos(g_elevation)))
        # right direction
        g_u = glm.normalize(glm.cross(g_v, g_w))
        # up direction
        g_v = glm.normalize(glm.cross(g_w, g_u))

        target_position = glm.vec3(0.0, 0.0, 0.0) + g_trans
        camera_position = target_position + g_w * g_distance

        # use orthogonal projection
        P = glm.ortho(-5,5, -5,5, -10000,10000)
        if g_perspective:
            P = glm.perspective(45, 1, .1, 10000)

        # view matrix
        # rotate camera position with g_azimuth / move camera up & down with g_elevation
        V = glm.lookAt(camera_position, target_position, g_v)
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(I))
        glUniform3f(view_pos_loc, camera_position.x, camera_position.y, camera_position.z)
        glUniform3f(color_loc, 1., 1., 1.)

        draw_frame(vao_frame, MVP, MVP_loc, I, M_loc, camera_position, view_pos_loc, color_loc)
        draw_grid(vao_grid, MVP, MVP_loc, I, M_loc, camera_position, view_pos_loc, color_loc)
        if file_dropped:
            draw_bvh(g_nodes, g_joints, vao_cube, vao_line, MVP, MVP_loc, M_loc, camera_position, view_pos_loc, color_loc)
        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
