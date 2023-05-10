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

g_u =     glm.vec3(0.,0.,0.,)
g_v =     glm.vec3(0.,0.,0.,)
g_w =     glm.vec3(0.,0.,0.,)

lastX = 0.
lastY = 0.
isDraggingLeft = False
isDraggingRight = False

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
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color
        self.num_vertex = 0

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color
    
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
    global isDraggingLeft, isDraggingRight, lastX, lastY, startX, startY
    # Orbit
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            isDraggingLeft = True
            startX, startY = glfwGetCursorPos(window)
            lastX, lastY = startX, startY
        elif action == GLFW_RELEASE:
            isDraggingLeft = False
    # Pan
    if button == GLFW_MOUSE_BUTTON_RIGHT:
        if action == GLFW_PRESS:
            isDraggingRight = True
            startX, startY = glfwGetCursorPos(window)
            lastX, lastY = startX, startY
        elif action == GLFW_RELEASE:
            isDraggingRight = False

def mouse_callback(window, xpos, ypos):
    global lastX, lastY, g_azimuth, g_elevation, isDraggingLeft, isDraggingRight, g_trans
    # Orbit
    if isDraggingLeft:
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
    if isDraggingRight:
        xoffset = xpos - lastX
        yoffset = lastY - ypos
        g_trans -= g_u * xoffset / 500
        g_trans -= g_v * yoffset / 500

        lastX = xpos
        lastY = ypos
        

def key_callback(window, key, scancode, action, mods):
    global g_perspective, hierarchical_mode, solid_mode
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


def scroll_callback(window, x_offset, y_offset):
    global g_distance
    g_distance *= np.power(2, y_offset/10)

def files_to_vertex_array(path, isHierarchy = True):
    filename = os.path.basename(path)
    f = open(path)
    
    lines = f.readlines()
    num_face_three = 0
    num_face_four = 0
    num_face_more = 0
    num_face_total = 0
    vertex_array = np.array([], 'float32')
    normal_array = np.array([], 'float32')
    face_vertex_array = np.array([], 'int32')
    face_normal_array = np.array([], 'int32')
    for line in lines:
        fields = line.split()
        if len(fields) == 0:
            continue
        if fields[0] == 'v':
            vertex = (fields[1], fields[2], fields[3])
            vertex_array = np.append(vertex_array, np.float32(vertex))
        elif fields[0] == 'vn':
            normal = (fields[1], fields[2], fields[3])
            normal_array = np.append(normal_array, np.float32(normal))
        elif fields[0] == 'f':
            if len(fields) == 4:
                num_face_three += 1
            elif len(fields) == 5:
                num_face_four += 1
            elif len(fields) >= 6:
                num_face_more += 1

            v0 = fields[1].split('/')
            # print(v0)
            for idx in range(2, len(fields) - 1):
                v1 = fields[idx].split('/')
                v2 = fields[idx + 1].split('/')
                if len(v0) < 3:
                    v0.append(-1)
                if len(v1) < 3:
                    v1.append(-1)
                if len(v2) < 3:
                    v2.append(-1)
                tmpv = (int(v0[0]) - 1, int(v1[0]) - 1, int(v2[0]) - 1)
                tmpvn = (int(v0[-1]) - 1, int(v1[-1]) - 1, int(v2[-1]) - 1)
                face_vertex_array = np.append(face_vertex_array, np.int32(tmpv))
                face_normal_array = np.append(face_normal_array, np.int32(tmpvn))
            num_face_total += 1
    if not isHierarchy:
        print('Obj file name: ' + filename)
        print('Total number of faces: ' + str(num_face_total))
        print('Number of faces with 3 vertices: ' + str(num_face_three))
        print('Number of faces with 4 vertices: ' + str(num_face_four))
        print('Number of faces with more than 4 vertices: ' + str(num_face_more))
    vertex_array = vertex_array.reshape(int(vertex_array.size / 3), 3)
    normal_array = normal_array.reshape(int(normal_array.size / 3), 3)
    face_vertex_array = face_vertex_array.reshape(int(face_vertex_array.size / 3), 3)
    face_normal_array = face_normal_array.reshape(int(face_normal_array.size / 3), 3)
    vertices = np.zeros(0, 'float32')
    if len(normal_array) == 0:
        normal_array = np.append(normal_array, [0,0,0])
    for idx in range(face_vertex_array.shape[0]):
        vertices = np.append(vertices, vertex_array[face_vertex_array[idx][0]])
        vertices = np.append(vertices, normal_array[face_normal_array[idx][0]])
        vertices = np.append(vertices, vertex_array[face_vertex_array[idx][1]])
        vertices = np.append(vertices, normal_array[face_normal_array[idx][1]])
        vertices = np.append(vertices, vertex_array[face_vertex_array[idx][2]])
        vertices = np.append(vertices, normal_array[face_normal_array[idx][2]])
    return glm.array(vertices)

def drop_callback(window, paths):
    global g_vertices, file_dropped, hierarchical_mode
    for path in paths:
        g_vertices = files_to_vertex_array(path, False)
        file_dropped = 1
        hierarchical_mode = False


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

def prepare_vao_hierarchy():
    VAOs = {}
    # create a hirarchical model - Node(parent, shape_transform, color)
    Nodes = {}
    Nodes['pipe'] = Node(None, glm.scale((.2,.2,.2,)),              glm.vec3(0., .8, 0.))
    Nodes['star'] = Node(Nodes['pipe'], glm.scale((.08,.08,.08)),   glm.vec3(1., 1., .1))
    Nodes['star1'] = Node(Nodes['star'], glm.scale((.04,.04,.04)),   glm.vec3(1., .5, .5))
    Nodes['star2'] = Node(Nodes['star'], glm.scale((.04,.04,.04)),   glm.vec3(.5, .5, 1))
    Nodes['star3'] = Node(Nodes['star'], glm.scale((.04,.04,.04)),   glm.vec3(1, 1, 1))
    Nodes['cube'] = Node(Nodes['pipe'], glm.scale((.02,.02,.02)),   glm.vec3(1., .8, 0.))
    Nodes['coin'] = Node(Nodes['cube'], glm.scale((.2,.2,.2)),      glm.vec3(1., 1., 0.))
    Nodes['mario'] = Node(Nodes['cube'], glm.scale((.01,.01,.01)),  glm.vec3(1., 0., 0.))

    for name in Nodes.keys():
        if name == 'star1' or name == 'star2' or name == 'star3':
            continue

        vertices = files_to_vertex_array(name + '.obj')
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
        VAOs[name] = VAO
        Nodes[name].num_vertex = int(len(vertices) / 6)
    
    Nodes['star1'].num_vertex = Nodes['star'].num_vertex
    Nodes['star2'].num_vertex = Nodes['star'].num_vertex
    Nodes['star3'].num_vertex = Nodes['star'].num_vertex
    VAOs['star1'] = VAOs['star']
    VAOs['star2'] = VAOs['star']
    VAOs['star3'] = VAOs['star']

    return VAOs, Nodes


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

def draw_obj(vao, MVP, MVP_loc, M, M_loc, view_pos, view_pos_loc, color_loc):
    global g_vertices
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, 1., 1., 1.)
    glDrawArrays(GL_TRIANGLES, 0, int(len(g_vertices) / 6))

def draw_hierarchy(vaos, nodes, VP, MVP_loc, M_loc, view_pos, view_pos_loc, color_loc):
    t = glfwGetTime()
    I = glm.mat4()
    nodes['pipe'].set_transform(glm.rotate(t / 10, glm.vec3(0, 1, 0)) * glm.translate(glm.vec3(.5,0.,0.)))
    nodes['star'].set_transform(glm.rotate(t, glm.vec3(0.,1.,0)) * glm.translate(glm.vec3(0.,3. + glm.cos(2 * t), 0.)))
    nodes['star1'].set_transform(glm.rotate(t * 2, glm.vec3(0.,1.,.5)) * glm.translate(glm.vec3(0.,1.,2.)))
    nodes['star2'].set_transform(glm.rotate(t * 2, glm.vec3(.5,1.,0)) * glm.translate(glm.vec3(2.,1.,0.)))
    nodes['star3'].set_transform(glm.rotate(t * 2, glm.vec3(.3,1.,3)) * glm.translate(glm.vec3(1.,1.,1.)))
    nodes['cube'].set_transform(glm.translate(glm.vec3(0.,2.,1.5)) * glm.scale(glm.vec3(1.,glm.max(.9, glm.abs(glm.sin(t * 2))), 1.)))
    nodes['coin'].set_transform(glm.translate(glm.vec3(0., .3 + 2 * glm.abs(glm.cos(t * 2)), 0.)) * glm.rotate(t * 2, glm.vec3(0,1,0)))
    nodes['mario'].set_transform(glm.translate(glm.vec3(0., -1 - glm.abs(glm.cos(t * 2)), 0.)))

    nodes['pipe'].update_tree_global_transform()

    for name in vaos.keys():
        draw_node(vaos[name], nodes[name], VP, MVP_loc, M_loc, view_pos, view_pos_loc, color_loc)

def draw_node(vao, node, VP, MVP_loc, M_loc, view_pos, view_pos_loc, color_loc):
    M = node.get_global_transform() * node.get_shape_transform()
    MVP = VP * M
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, node.num_vertex)
    
def main():
    global g_u, g_v, g_w, g_perspective, file_dropped, hierarchical_mode
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
    vao_hierarchy, node_hierarchy = prepare_vao_hierarchy()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        if solid_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if file_dropped == 1:
            vao_obj = prepare_vao_obj(g_vertices)
            file_dropped = 2

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
        if hierarchical_mode:
            draw_hierarchy(vao_hierarchy, node_hierarchy, MVP, MVP_loc, M_loc, camera_position, view_pos_loc, color_loc)
        elif file_dropped:
            draw_obj(vao_obj, MVP, MVP_loc, I, M_loc, camera_position, view_pos_loc, color_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
