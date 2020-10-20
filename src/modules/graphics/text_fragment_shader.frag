#version 450

layout(location = 0) in vec2 glyph_position;
layout(location = 1) in vec3 color;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D glyph_atlas;

const float width = 0.5;
const float edge = 0.1;

void main() {
    float distance = 1.0 - texture(glyph_atlas, glyph_position).a;
    float alpha = 1.0 - smoothstep(width, width + edge, distance);
    out_color = vec4(color, alpha);
}