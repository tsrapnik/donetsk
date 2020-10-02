use nalgebra::Vector2;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::modules::graphics;

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub position: Vector2<f32>,
    pub ideal_distance: f32,
    pub color: [f32; 3],
}

pub fn browse_folder(folder_tree: &mut DiGraph<Node, ()>, parent_node: NodeIndex) {
    let path = &folder_tree[parent_node].name;
    match std::fs::read_dir(path) {
        Ok(subfolders) => {
            let subfolder_count = subfolders.count() as f32;
            //we cannot use the subfolders iterator again, so we create a new one (just unwrap, since we already checked for valid return).
            //note: in theory it is possible the folder gets deleted in between the two reads, but this is virtually impossible.
            let subfolders = std::fs::read_dir(path).unwrap();
            folder_tree[parent_node].ideal_distance =
                (subfolder_count * 40.0).max(300.0) + folder_tree[parent_node].ideal_distance;
            let subfolder_distance = (subfolder_count * 20.0).max(150.0);
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let color = [
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            ];
            for subfolder in subfolders {
                let subfolder_position = folder_tree[parent_node].position
                    + Vector2::new(rng.gen_range(-100.0, 100.0), rng.gen_range(-100.0, 100.0));
                let new_node = folder_tree.add_node(Node {
                    name: subfolder
                        .unwrap()
                        .path()
                        .into_os_string()
                        .into_string()
                        .unwrap(),
                    position: subfolder_position,
                    ideal_distance: subfolder_distance,
                    color: color,
                });
                folder_tree.add_edge(parent_node, new_node, ());
            }
        }
        Err(_) => {}
    }
}

pub fn move_folders(folder_tree: &mut DiGraph<Node, ()>, root: NodeIndex) {
    //todo: maybe remove creation of displacements and immediately change positions (should have nearly the same effect and requires less computation).
    let mut displacements = vec![Vector2::new(0.0, 0.0); folder_tree.node_count()];
    //todo: fix issue of chasing semicircles.
    //for each pair of nodes repel the nodes if they come too close.
    for node_0 in folder_tree.node_indices() {
        let fixed_position = folder_tree[node_0].position;
        for node_1 in folder_tree.node_indices() {
            if (node_1 != root) && (node_1 != node_0) {
                let repel_vector = folder_tree[node_1].position - fixed_position;
                let repel_vector_length = repel_vector.norm();
                let force_range = 300.0;
                let max_force = 15.0;
                if repel_vector_length == 0.0 {
                    //if two nodes are on the same position, we don't know at what direction to push the other node.
                    //just push it a little to a fixed direction to solve the issue.
                    displacements[node_1.index()] =
                        displacements[node_1.index()] + Vector2::new(1.0, 0.0);
                } else if repel_vector_length < force_range {
                    //linear relation between distance and force seems to work well.
                    let repel_vector = repel_vector.normalize()
                        * (max_force / force_range * (force_range - repel_vector_length));
                    displacements[node_1.index()] = displacements[node_1.index()] + repel_vector;
                }
            }
        }
    }
    //for each edge pull back the child node towards the parent node to a wanted distance (or push away if too close).
    for index in (*folder_tree).edge_indices() {
        let (node_0, node_1) = (*folder_tree).edge_endpoints(index).unwrap();
        let attract_vector = folder_tree[node_0].position - folder_tree[node_1].position;
        let attract_vector_length = attract_vector.norm();
        let speed = 1.0;
        let attract_vector = attract_vector.normalize()
            * (speed * (attract_vector_length - folder_tree[node_1].ideal_distance));
        displacements[node_1.index()] = displacements[node_1.index()] + attract_vector;
    }
    //apply the calculated displacements to all node positions.
    for node in folder_tree.node_indices() {
        folder_tree[node].position = folder_tree[node].position + displacements[node.index()];
    }
}

pub fn draw_folder_tree(
    folder_tree: &DiGraph<Node, ()>,
    vertex_buffer: &mut Vec<graphics::WindowVertex>,
    window_dimensions: Vector2<f32>,
) {
    for node in folder_tree.node_indices() {
        draw_folder(
            folder_tree[node].position,
            folder_tree[node].color,
            Vector2::new(100.0, 100.0),
            vertex_buffer,
            window_dimensions,
        );
    }
    for edge in folder_tree.edge_indices() {
        if let Some(node_pair) = folder_tree.edge_endpoints(edge) {
            draw_line(
                folder_tree[node_pair.0].position,
                folder_tree[node_pair.1].position,
                vertex_buffer,
                window_dimensions,
            );
        }
    }
}

fn draw_folder(
    position: Vector2<f32>,
    color: [f32; 3],
    size: Vector2<f32>,
    vertex_buffer: &mut Vec<graphics::WindowVertex>,
    window_dimensions: Vector2<f32>,
) {
    [
        Vector2::new(position[0], position[1]),
        Vector2::new(position[0] + size[0], position[1]),
        Vector2::new(position[0], position[1] - size[1]),
        Vector2::new(position[0], position[1] - size[1]),
        Vector2::new(position[0] + size[0], position[1]),
        Vector2::new(position[0] + size[0], position[1] - size[1]),
    ]
    .iter()
    .map(|x| graphics::pixel_to_screen_coordinates(*x, window_dimensions))
    .map(|p| {
        vertex_buffer.push(graphics::WindowVertex {
            position: [p.x, p.y],
            color: color,
        })
    })
    .count();
}

fn draw_line(
    start: Vector2<f32>,
    end: Vector2<f32>,
    vertex_buffer: &mut Vec<graphics::WindowVertex>,
    window_dimensions: Vector2<f32>,
) {
    let start = Vector2::new(start[0], start[1]);
    let end = Vector2::new(end[0], end[1]);

    let unit_line = (end - start).normalize();
    let orthogonal = Vector2::new(unit_line.y, -unit_line.x);

    let corner_0 = start + orthogonal;
    let corner_1 = start - orthogonal;
    let corner_2 = end + orthogonal;
    let corner_3 = end - orthogonal;

    [corner_0, corner_1, corner_2, corner_1, corner_2, corner_3]
        .iter()
        .map(|x| graphics::pixel_to_screen_coordinates(*x, window_dimensions))
        .map(|p| {
            vertex_buffer.push(graphics::WindowVertex {
                position: [p.x, p.y],
                color: [0.5, 0.2, 0.2],
            })
        })
        .count();
}
