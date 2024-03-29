use petgraph::graph::{DiGraph, NodeIndex};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

mod font;
mod modules;
use modules::folder_tree;
use modules::graphics;

//when set to true we measure different parts of the rendering loop and print it to the console.
const DEBUG_MODE: bool = false;

fn main() {
    let event_loop = EventLoop::new();
    let mut renderer = graphics::Renderer::new(&event_loop);

    let mut folder_tree = DiGraph::new();
    let root = folder_tree.add_node(folder_tree::Node {
        name: "/home/tsrapnik".to_string(),
        position: [-0.9, -0.9],
        size: [folder_tree::FOLDER_SIDE; 2],
        color: [0.5, 0.2, 0.2],
    });

    folder_tree::browse_folder(&mut folder_tree, root);
    folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(1));
    // folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(2));
    // folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(3));
    // folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(4));
    // folder_tree::browse_folder(&mut folder_tree, NodeIndex::new(5));

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
            let frame_start = std::time::Instant::now();

            let mut rectangle_buffer = Vec::new();
            let mut character_buffer = Vec::new();
            folder_tree::draw(&folder_tree, &mut character_buffer, &mut rectangle_buffer);

            if DEBUG_MODE {
                println!("time lost by cpu scheduling is ignored.");
                println!("cpu processing time: {:?}", frame_start.elapsed());
            }
            renderer.render(character_buffer, rectangle_buffer, window_resized);
            window_resized = false;

            if DEBUG_MODE {
                println!("frame time: {:?}\n", frame_start.elapsed());
            }
        }
        _ => (),
    })
}
