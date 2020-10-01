use std::{env, fs, path::Path, fmt::Write};

fn main() {
    let mut fnt_path = env::current_dir().unwrap();
    fnt_path.push(Path::new("src/font/deja_vu_sans_mono.fnt"));
    let fnt_file = fs::read_to_string(fnt_path).unwrap();

    let mut line_height = 0u32;

    #[derive(Default, Copy, Clone, Debug)]
    struct GlyphLayout {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        x_offset: i32,
        y_offset: i32,
        x_advance: u32,
    }
    let mut glyph_layouts: [GlyphLayout; 128] = [Default::default(); 128];

    let lines = fnt_file.lines();
    for line in lines {
        let mut word_groups = line.split_whitespace();
        let first_word = word_groups.next();
        match first_word {
            Some("common") => {
                for key_value_pair in word_groups {
                    match to_key_and_value(key_value_pair) {
                        ("lineHeight", value) => line_height = value as u32,
                        _ => {}
                    }
                }
            }

            Some("char") => {
                let mut glyph_layout = GlyphLayout {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                    x_offset: 0,
                    y_offset: 0,
                    x_advance: 0,
                };
                let mut index: usize = 0;
                for key_value_pair in word_groups {
                    match to_key_and_value(key_value_pair) {
                        ("x", value) => glyph_layout.x = value as u32,
                        ("y", value) => glyph_layout.y = value as u32,
                        ("width", value) => glyph_layout.width = value as u32,
                        ("height", value) => glyph_layout.height = value as u32,
                        ("xoffset", value) => glyph_layout.x_offset = value as i32,
                        ("yoffset", value) => glyph_layout.y_offset = value as i32,
                        ("xadvance", value) => glyph_layout.x_advance = value as u32,
                        ("id", value) => index = value as usize,
                        _ => {}
                    }
                }
                glyph_layouts[index] = glyph_layout;
            }

            _ => {}
        }
    }

    let mut output_string = String::from("//todo:");
    write!(&mut output_string, "let line_height = {}u32;\n", line_height).unwrap();
    output_string += "let glyph_layouts = [\n";
    for glyph_layout in glyph_layouts.iter(){
        write!(&mut output_string, "    {:?},\n", glyph_layout).unwrap();
    }
    output_string += "];";

    let mut output_path = env::current_dir().unwrap();
    output_path.push(Path::new("src/font/font.rs"));
    fs::write(output_path, output_string).unwrap();
}

fn to_key_and_value(key_value_pair: &str) -> (&str, i64) {
    let mut key_and_value = key_value_pair.split('=');
    (
        key_and_value.next().unwrap(),
        key_and_value.next().unwrap().parse::<i64>().unwrap(),
    )
}
