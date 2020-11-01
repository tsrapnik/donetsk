#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 padding_0;
layout(location = 2) in vec3 color;
layout(location = 3) in float padding_1;

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_color = color;
}