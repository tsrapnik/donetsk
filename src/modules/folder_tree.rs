use petgraph::{
    graph::{DiGraph, NodeIndex},
    Direction,
};
use std::fs;

use crate::modules::graphics;

pub const FOLDER_SIDE: f32 = 0.1;
pub const FOLDER_MARGIN: f32 = 0.01;

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
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
        let new_parent_side_length = subfolder_row_count * (FOLDER_SIDE + FOLDER_MARGIN) + FOLDER_MARGIN;
        let offset = [
            new_parent_side_length - folder_tree[parent_node].size[0],
            new_parent_side_length - folder_tree[parent_node].size[1],
        ];

        //move the siblings of the parent node out of the way and do the same for its uncles and
        //aunts, grand uncles and grand aunts, etc.
        let mut current_node = parent_node;
        loop {
            //we only need to move the siblings past bottom right corner of current node away.
            let current_node_far_corner = [
                folder_tree[current_node].position[0] + folder_tree[current_node].size[0],
                folder_tree[current_node].position[1] + folder_tree[current_node].size[1],
            ];

            //expand the node itself.
            folder_tree[current_node].size[0] += offset[0];
            folder_tree[current_node].size[1] += offset[1];

            if let Some(current_node_parent) = folder_tree.neighbors_directed(current_node, Direction::Incoming).next() {
                //move siblings out of the way.
                let mut sibling_edges = folder_tree.neighbors_directed(current_node_parent, Direction::Outgoing).detach();
                while let Some(sibling_edge) = sibling_edges.next_edge(&folder_tree) {
                    let (_, sibling) = folder_tree.edge_endpoints(sibling_edge).unwrap();
                    if folder_tree[sibling].position[0] > current_node_far_corner[0] {
                        folder_tree[sibling].position[0] += offset[0];
                    }
                    if folder_tree[sibling].position[1] > current_node_far_corner[1] {
                        folder_tree[sibling].position[1] += offset[1];
                    }
                }
                current_node = current_node_parent;
            } else
            {
                break;
            }
        }

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
            let offset = [
                (index as f32) % subfolder_row_count,
                ((index as f32) / subfolder_row_count).floor(),
            ];
            let subfolder_position = [
                folder_tree[parent_node].position[0]
                    + FOLDER_MARGIN
                    + offset[0] * (FOLDER_SIDE + FOLDER_MARGIN),
                folder_tree[parent_node].position[1]
                    + FOLDER_MARGIN
                    + offset[1] * (FOLDER_SIDE + FOLDER_MARGIN),
            ];
            let new_node = folder_tree.add_node(Node {
                name: subfolder.path().into_os_string().into_string().unwrap(),
                position: subfolder_position,
                size: [FOLDER_SIDE; 2],
                color: color,
            });
            folder_tree.add_edge(parent_node, new_node, ());
        }
    }
}

pub fn draw(
    folder_tree: &DiGraph<Node, ()>,
    text_character_buffer: &mut Vec<graphics::TextCharacter>,
    rectangle_buffer: &mut Vec<graphics::Rectangle>,
) {
    //at this stage rectangles are already ordered from back to front, so we can just loop through
    //them.
    for node in folder_tree.node_indices() {
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
