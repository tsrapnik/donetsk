use std::{env, fmt::Write, fs, path::Path};

fn main() {
    const ASCII_TABLE_LENGHT: usize = 128;
    const TEXTURE_SIDE: f32 = 512.0;
    let mut fnt_path = env::current_dir().unwrap();
    fnt_path.push(Path::new("src/font/deja_vu_sans_mono.fnt"));
    let fnt_file = fs::read_to_string(fnt_path).unwrap();

    let mut line_height = 0.0f32;

    #[derive(Default, Copy, Clone, Debug)]
    struct GlyphLayout {
        origin: [f32; 2],
        size: [f32; 2],
        offset: [f32; 2],
        advance: f32,
        padding: f32,
    }
    let mut glyph_layouts: [GlyphLayout; ASCII_TABLE_LENGHT] =
        [Default::default(); ASCII_TABLE_LENGHT];

    let lines = fnt_file.lines();
    for line in lines {
        let mut word_groups = line.split_whitespace();
        let first_word = word_groups.next();
        match first_word {
            Some("common") => {
                for key_value_pair in word_groups {
                    match to_key_and_value(key_value_pair) {
                        ("lineHeight", value) => line_height = (value as f32) / TEXTURE_SIDE,
                        _ => {}
                    }
                }
            }

            Some("char") => {
                let mut glyph_layout = GlyphLayout::default();
                let mut index = usize::default();
                for key_value_pair in word_groups {
                    match to_key_and_value(key_value_pair) {
                        ("x", value) => glyph_layout.origin[0] = (value as f32) / TEXTURE_SIDE,
                        ("y", value) => glyph_layout.origin[1] = (value as f32) / TEXTURE_SIDE,
                        ("width", value) => glyph_layout.size[0] = (value as f32) / TEXTURE_SIDE,
                        ("height", value) => glyph_layout.size[1] = (value as f32) / TEXTURE_SIDE,
                        ("xoffset", value) => glyph_layout.offset[0] = (value as f32) / TEXTURE_SIDE,
                        ("yoffset", value) => glyph_layout.offset[1] = (value as f32) / TEXTURE_SIDE,
                        ("xadvance", value) => glyph_layout.advance = (value as f32) / TEXTURE_SIDE,
                        ("id", value) => index = value as usize,
                        _ => {}
                    }
                }
                glyph_layouts[index] = glyph_layout;
            }

            _ => {}
        }
    }

    let mut output_string = String::from(
        "
#[derive(Default, Copy, Clone, Debug)]
pub struct GlyphLayout {
    pub origin: [f32; 2], //where in texture the glyph is located
    pub size: [f32; 2], //size of the glyph
    pub offset: [f32; 2], //offset from the cursor where the glyp should be rendered
    pub advance: f32, //offset the cursor should move horizontally for next glyph
    pub padding: f32, //padding to comply with std140 rules
}\n\n",
    );
    write!(
        &mut output_string,
        "pub const LINE_HEIGHT: f32 = {:.1};\n",
        line_height,
    )
    .unwrap();
    write!(
        &mut output_string,
        "pub const GLYPH_LAYOUTS: [GlyphLayout; {}] = [\n",
        ASCII_TABLE_LENGHT,
    )
    .unwrap();
    for glyph_layout in glyph_layouts.iter() {
        write!(&mut output_string, "    {:?},\n", glyph_layout).unwrap();
    }
    output_string += "];";

    let mut output_path = env::current_dir().unwrap();
    output_path.push(Path::new("src/font/mod.rs"));
    fs::write(output_path, output_string).unwrap();
}

fn to_key_and_value(key_value_pair: &str) -> (&str, i64) {
    let mut key_and_value = key_value_pair.split('=');
    (
        key_and_value.next().unwrap(),
        key_and_value.next().unwrap().parse::<i64>().unwrap(),
    )
}
