#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texture_coordinates;
layout(location = 0) out vec2 out_texture_coordinates;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_texture_coordinates = texture_coordinates;
}