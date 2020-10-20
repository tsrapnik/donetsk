#version 450

layout(location = 0) in vec2 render_position;
layout(location = 1) in vec2 glyph_position;
layout(location = 2) in vec3 color;
layout(location = 0) out vec2 out_glyph_position;
layout(location = 1) out vec3 out_color;

void main() {
    gl_Position = vec4(render_position, 0.0, 1.0);
    out_glyph_position = glyph_position;
    out_color = color;
}