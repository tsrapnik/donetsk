#version 450

layout(location = 0) in vec2 position;
layout(location = 2) in vec3 color; //location 2 in stead of 1, since 1 points to some padding.

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_color = color;
}