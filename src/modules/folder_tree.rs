use nalgebra::Vector2;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::Bfs,
};
use std::fs;

use crate::modules::graphics;

pub const FOLDER_SIDE: f32 = 0.1;
pub const FOLDER_MARGIN: f32 = 0.01;

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub position: Vector2<f32>,
    pub size: Vector2<f32>,
    pub color: [f32; 3],
}

pub fn browse_folder(folder_tree: &mut DiGraph<Node, ()>, parent_node: NodeIndex) {
    let path = &folder_tree[parent_node].name;

    if let Ok(subfolders) = fs::read_dir(path) {
        //collect subfolders in a vec, so we can count them while still looping through them
        //afterwards.
        let subfolders: Vec<fs::DirEntry> = subfolders.filter_map(Result::ok).collect();
        let subfolder_count = subfolders.len();

        //calculate how large the parent folder should be to fit subfolders.
        let subfolder_row_count = (subfolder_count as f64).sqrt().ceil() as f32;
        let parent_size = {
            let parent_side = subfolder_row_count * (FOLDER_SIDE + FOLDER_MARGIN) + FOLDER_MARGIN;
            Vector2::from([parent_side; 2])
        };

        //move existing folders to make place for parent folder.
        let parent_far_corner = folder_tree[parent_node].position + folder_tree[parent_node].size;
        for node in folder_tree.node_indices() {
            //only folders in the quadrant with x and y values greater than the parent x and y should be moved.
            if folder_tree[node].position > parent_far_corner {
                folder_tree[node].position += parent_size;
            }
        }

        //enlarge the parent folder to fit the subfolders.
        folder_tree[parent_node].size = parent_size;

        //decide color of the subfolders.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let color = [
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
        ];

        //place subfolders in parent folder.
        for (index, subfolder) in subfolders.into_iter().enumerate() {
            let offset = Vector2::new(
                (index as f32) % subfolder_row_count,
                ((index as f32) / subfolder_row_count).floor(),
            );
            let subfolder_position = folder_tree[parent_node].position
                + Vector2::from([FOLDER_MARGIN; 2])
                + offset.component_mul(&Vector2::from([FOLDER_SIDE + FOLDER_MARGIN; 2]));
            let new_node = folder_tree.add_node(Node {
                name: subfolder.path().into_os_string().into_string().unwrap(),
                position: subfolder_position,
                size: Vector2::from([FOLDER_SIDE; 2]),
                color: color,
            });
            folder_tree.add_edge(parent_node, new_node, ());
        }
    }
}

pub fn draw(
    folder_tree: &DiGraph<Node, ()>,
    root: NodeIndex,
    text_character_buffer: &mut Vec<graphics::TextCharacter>,
    rectangle_buffer: &mut Vec<graphics::Rectangle>,
) {
    let mut bfs = Bfs::new(&*folder_tree, root);
    while let Some(node) = bfs.next(&*folder_tree) {
        rectangle_buffer.push(graphics::Rectangle {
            position: folder_tree[node].position.into(),
            size: folder_tree[node].size.into(),
            color: folder_tree[node].color,
            padding: 0.0,
        });
        graphics::push_string(
            &folder_tree[node].name,
            0.05,
            folder_tree[node].position.into(),
            [0.0; 3],
            text_character_buffer,
        );
    }
}
