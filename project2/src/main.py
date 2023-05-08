from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_distance = 5
g_azimuth = np.radians(45)
g_elevation = np.radians(45)
g_perspective = True
g_trans = glm.vec3(0.,0.,0.)

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
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

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
    global g_perspective
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                g_perspective = not g_perspective


def scroll_callback(window, x_offset, y_offset):
    global g_distance
    g_distance *= np.power(2, y_offset/10)

def drop_callback(window, count, paths):
    for path in paths:
        handle_dropped_files(path)

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position            color
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
 
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    list = []
    s = -10
    e = 10
    for i in range(s, e):
        for j in range(s, e):
            list.extend([(i) * .5,      0., (j) * .5,     0.5, 0.5, 0.5])
            list.extend([(i + 1) * .5,  0., (j) * .5,     0.5, 0.5, 0.5])
            list.extend([(i + 1) * .5,  0., (j + 1) * .5, 0.5, 0.5, 0.5])
            list.extend([(i) * .5,      0., (j + 1) * .5, 0.5, 0.5, 0.5])
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.5, 0.5, # x-axis start
         1.0, 0.0, 0.0,  1.0, 0.5, 0.5, # x-axis end 
         0.0, 0.0, 0.0,  0.5, 1.0, 0.5, # y-axis start
         0.0, 1.0, 0.0,  0.5, 1.0, 0.5, # y-axis end 
         0.0, 0.0, 0.0,  0.5, 0.5, 1.0, # z-axis start
         0.0, 0.0, 1.0,  0.5, 0.5, 1.0, # z-axis end 
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

# def draw_cube(vao, MVP, MVP_loc):
#     glBindVertexArray(vao)
#     glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
#     glDrawArrays(GL_TRIANGLES, 0, 36)

def draw_grid(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    for i in range(400):
        glDrawArrays(GL_LINE_LOOP, i * 4, 4)


def main():
    global g_u, g_v, g_w, g_perspective
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
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    # vao_cube = prepare_vao_cube()
    vao_grid = prepare_vao_grid()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
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
        P = glm.ortho(-5,5, -5,5, -10,10)
        if g_perspective:
            P = glm.perspective(45, 1, .1, 10)


        # view matrix
        # rotate camera position with g_azimuth / move camera up & down with g_elevation
        V = glm.lookAt(camera_position, target_position, g_v)
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw_cube(vao_cube, MVP, MVP_loc)
        draw_frame(vao_frame, MVP, MVP_loc)
        draw_grid(vao_grid, MVP, MVP_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
