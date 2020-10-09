#version 450

layout(location = 0) in vec2 texture_coordinates;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D font_atlas;

const float width = 0.5;
const float edge = 0.1;
const vec3 color = vec3(0.0, 0.0, 0.0);

void main() {
    float distance = 1.0 - texture(font_atlas, texture_coordinates).a;
    float alpha = 1.0 - smoothstep(width, width + edge, distance);
    out_color = vec4(color, alpha);
}