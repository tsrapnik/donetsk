use std::time;

use petgraph::graph::{DiGraph, NodeIndex};

use nalgebra::Vector2;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

mod font;
mod modules;
use modules::folder_tree;
use modules::graphics;

fn main() {
    let event_loop = EventLoop::new();
    let mut renderer = graphics::Renderer::new(&event_loop);

    let initial_dimensions = [1000, 1000];
    let dimensions = initial_dimensions;

    let mut folder_tree = DiGraph::new();
    let root = folder_tree.add_node(folder_tree::Node {
        name: "C:/Users/tsrapnik".to_string(),
        position: Vector2::new(
            0.5 * (initial_dimensions[0] as f32),
            0.5 * (initial_dimensions[1] as f32),
        ),
        ideal_distance: 0.0,
        color: [0.5, 0.2, 0.2],
    });

    folder_tree::browse_folder(&mut folder_tree, root);
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(1));
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(2));
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(3));
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(4));
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(5));

    let mut window_resized = false;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
        }
        Event::RedrawEventsCleared => {
            let now = time::Instant::now();

            folder_tree[root].position.x = 0.5 * (dimensions[0] as f32);
            folder_tree[root].position.y = 0.5 * (dimensions[1] as f32);

            let mut vertices = Vec::new();

            folder_tree::move_folders(&mut folder_tree, root);

            folder_tree::draw_folder_tree(
                &folder_tree,
                &mut vertices,
                Vector2::new(dimensions[0] as f32, dimensions[1] as f32),
            );

            let mut character_buffer = Vec::with_capacity(11);
            graphics::push_string(
                "hello\nworld.",
                0.3,
                [0.0, 0.0],
                [0.0, 0.0, 0.0],
                &mut character_buffer,
            );

            renderer.render(character_buffer, vertices, window_resized);
            window_resized = false;

            println!("{:?}", now.elapsed());
        }
        _ => (),
    })
}
