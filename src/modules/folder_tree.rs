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

        //decide color of the subfolders.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let color = [
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
        ];

        //place subfolders in parent folder.
        let subfolder_row_count = (subfolders.len() as f64).sqrt().ceil() as f32;
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

        //now increase the parent folder, so it encapsulates all children and push away the siblings
        //that would otherwise overlap. recursively do the same for all ancestors.
        let mut current_node = parent_node;
        let mut offset = [
            subfolder_row_count * (FOLDER_SIDE + FOLDER_MARGIN) + FOLDER_MARGIN
                - folder_tree[current_node].size[0],
            subfolder_row_count * (FOLDER_SIDE + FOLDER_MARGIN) + FOLDER_MARGIN
                - folder_tree[current_node].size[1],
        ];
        loop {
            //expand parent folder.
            folder_tree[current_node].size[0] += offset[0];
            folder_tree[current_node].size[1] += offset[1];

            let top_left_corner = [
                folder_tree[current_node].position[0] - FOLDER_MARGIN,
                folder_tree[current_node].position[1] - FOLDER_MARGIN,
            ];
            let bottom_right_corner = [
                folder_tree[current_node].position[0]
                    + folder_tree[current_node].size[0]
                    + FOLDER_MARGIN,
                folder_tree[current_node].position[1]
                    + folder_tree[current_node].size[1]
                    + FOLDER_MARGIN,
            ];

            if let Some(current_node_parent) = folder_tree
                .neighbors_directed(current_node, Direction::Incoming)
                .next()
            {
                //move siblings out of the way.
                //first check how far we need to move the siblings to avoid overlap.
                let mut sibling_edges = folder_tree
                    .neighbors_directed(current_node_parent, Direction::Outgoing)
                    .detach();
                offset = [0.0; 2];
                while let Some(sibling_edge) = sibling_edges.next_edge(&folder_tree) {
                    let (_, sibling) = folder_tree.edge_endpoints(sibling_edge).unwrap();
                    if (folder_tree[sibling].position[0] > top_left_corner[0])
                        && (folder_tree[sibling].position[0] < bottom_right_corner[0])
                        && (folder_tree[sibling].position[0] > top_left_corner[0])
                        && (folder_tree[sibling].position[1] < bottom_right_corner[1])
                    {
                        offset[0] = offset[0]
                            .max(bottom_right_corner[0] - folder_tree[sibling].position[0]);
                        offset[1] = offset[1]
                            .max(bottom_right_corner[1] - folder_tree[sibling].position[1]);
                    }
                }
                //then actually move the relevant siblings.
                let mut sibling_edges = folder_tree
                    .neighbors_directed(current_node_parent, Direction::Outgoing)
                    .detach();
                while let Some(sibling_edge) = sibling_edges.next_edge(&folder_tree) {
                    let (_, sibling) = folder_tree.edge_endpoints(sibling_edge).unwrap();
                    if (folder_tree[sibling].position[0] > top_left_corner[0])
                        && (folder_tree[sibling].position[0] < bottom_right_corner[0])
                        && (folder_tree[sibling].position[0] > top_left_corner[0])
                        && (folder_tree[sibling].position[1] < bottom_right_corner[1])
                    {
                        folder_tree[sibling].position[0] += offset[0];
                        folder_tree[sibling].position[1] += offset[1];
                    }
                }

                current_node = current_node_parent;
            } else {
                break;
            }
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
        let name = folder_tree[node]
            .name
            .split(&['\\', '/'][..])
            .last()
            .unwrap();
        graphics::push_string(
            name,
            0.05,
            folder_tree[node].position.into(),
            [0.0; 3],
            text_character_buffer,
        );
    }
}
