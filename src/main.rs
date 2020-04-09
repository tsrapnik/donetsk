// Any difference in code from vulkano triangle.rs example is noted with ALL CAPS COMMENT BLOCKS
// The formatting is different though.

// IMPORT
use vulkano_text::{DrawText, DrawTextTrait};
// IMPORT END

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use std::sync::Arc;

use petgraph::graph::{DiGraph, NodeIndex};

use nalgebra::Vector2;

#[derive(Debug)]
struct Node {
    name: String,
    position: Vector2<f32>,
    ideal_distance: f32,
    color: [f32; 3],
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, color);

fn browse_folder(a_graph: &mut DiGraph<Node, ()>, parent_node: NodeIndex) {
    let path = &a_graph[parent_node].name;
    match std::fs::read_dir(path) {
        Ok(subfolders) => {
            let subfolder_count = subfolders.count() as f32;
            //we cannot use the subfolders iterator again, so we create a new one (just unwrap, since we already checked for valid return).
            //note: in theory it is possible the folder gets deleted in between the two reads, but this is virtually impossible.
            let subfolders = std::fs::read_dir(path).unwrap();
            a_graph[parent_node].ideal_distance =
                (subfolder_count * 40.0).max(300.0) + a_graph[parent_node].ideal_distance;
            let subfolder_distance = (subfolder_count * 20.0).max(150.0);
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let color = [
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            ];
            for subfolder in subfolders {
                let subfolder_position = a_graph[parent_node].position
                    + Vector2::new(rng.gen_range(-100.0, 100.0), rng.gen_range(-100.0, 100.0));

                let new_node = a_graph.add_node(Node {
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
                a_graph.add_edge(parent_node, new_node, ());
            }
        }
        Err(_) => {}
    }
}

fn move_graph_nodes(a_graph: &mut DiGraph<Node, ()>, root: NodeIndex) {
    //todo: maybe remove creation of displacements and immediately change positions (should have nearly the same effect and requires less computation).
    let mut displacements = vec![Vector2::new(0.0, 0.0); a_graph.node_count()];

    //todo: fix issue of chasing semicircles.

    //for each pair of nodes repel the nodes if they come too close.
    for node_0 in a_graph.node_indices() {
        let fixed_position = a_graph[node_0].position;

        for node_1 in a_graph.node_indices() {
            if (node_1 != root) && (node_1 != node_0) {
                let repel_vector = a_graph[node_1].position - fixed_position;
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
    for index in (*a_graph).edge_indices() {
        let (node_0, node_1) = (*a_graph).edge_endpoints(index).unwrap();

        let attract_vector = a_graph[node_0].position - a_graph[node_1].position;
        let attract_vector_length = attract_vector.norm();
        let speed = 1.0;
        let attract_vector = attract_vector.normalize()
            * (speed * (attract_vector_length - a_graph[node_1].ideal_distance));
        displacements[node_1.index()] = displacements[node_1.index()] + attract_vector;
    }

    //apply the calculated displacements to all node positions.
    for node in a_graph.node_indices() {
        a_graph[node].position = a_graph[node].position + displacements[node.index()];
    }
}

fn draw_graph(
    a_graph: & DiGraph<Node, ()>,
    text_buffer: &mut DrawText,
    vertex_buffer: &mut Vec<Vertex>,
    window_dimensions: Vector2<f32>,
) {
    for node in a_graph.node_indices() {
        let folder_or_file_index = a_graph[node].name.rfind("/").unwrap() + 1;
        draw_folder(
            &a_graph[node].name[folder_or_file_index..],
            a_graph[node].position,
            a_graph[node].color,
            Vector2::new(100.0, 100.0),
            text_buffer,
            vertex_buffer,
            window_dimensions,
        );
    }

    for edge in a_graph.edge_indices() {
        if let Some(node_pair) = a_graph.edge_endpoints(edge) {
            draw_line(
                a_graph[node_pair.0].position,
                a_graph[node_pair.1].position,
                vertex_buffer,
                window_dimensions,
            );
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_color = color;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec3 color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(color, 1.0);
}"
    }
}

fn pixel_to_screen_coordinates(
    position: Vector2<f32>,
    window_dimensions: Vector2<f32>,
) -> Vector2<f32> {
    position.zip_map(&window_dimensions, |p, w| 2.0 / w * p - 1.0)
}

fn screen_to_pixel_coordinates(
    position: Vector2<f32>,
    window_dimensions: Vector2<f32>,
) -> Vector2<f32> {
    position.zip_map(&window_dimensions, |p, w| w / 2.0 * (p + 1.0))
}

fn draw_folder(
    title: &str,
    position: Vector2<f32>,
    color: [f32; 3],
    size: Vector2<f32>,
    text_buffer: &mut DrawText,
    vertex_buffer: &mut Vec<Vertex>,
    window_dimensions: Vector2<f32>,
) {
    text_buffer.queue_text(
        position[0],
        position[1],
        size[1] * 0.3,
        [0.6, 0.6, 0.6, 1.0],
        title,
    );

    [
        Vector2::new(position[0], position[1]),
        Vector2::new(position[0] + size[0], position[1]),
        Vector2::new(position[0], position[1] - size[1]),
        Vector2::new(position[0], position[1] - size[1]),
        Vector2::new(position[0] + size[0], position[1]),
        Vector2::new(position[0] + size[0], position[1] - size[1]),
    ]
    .iter()
    .map(|x| pixel_to_screen_coordinates(*x, window_dimensions))
    .map(|p| {
        vertex_buffer.push(Vertex {
            position: [p.x, p.y],
            color: color,
        })
    })
    .count();
}

fn draw_line(
    start: Vector2<f32>,
    end: Vector2<f32>,
    vertex_buffer: &mut Vec<Vertex>,
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
        .map(|x| pixel_to_screen_coordinates(*x, window_dimensions))
        .map(|p| {
            vertex_buffer.push(Vertex {
                position: [p.x, p.y],
                color: [0.5, 0.2, 0.2],
            })
        })
        .count();
}

fn main() {
    let mut a_graph = DiGraph::new();
    let root = a_graph.add_node(Node {
        name: "/home/tsrapnik/stack/projects".to_string(),
        position: Vector2::new(2000.0, 1000.0),
        ideal_distance: 0.0,
        color: [0.5, 0.2, 0.2],
    });

    browse_folder(&mut a_graph, root);
    browse_folder(&mut a_graph, NodeIndex::new(1));
    browse_folder(&mut a_graph, NodeIndex::new(2));
    browse_folder(&mut a_graph, NodeIndex::new(3));
    browse_folder(&mut a_graph, NodeIndex::new(4));
    browse_folder(&mut a_graph, NodeIndex::new(5));

    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        return;
    };

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            initial_dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None,
        )
        .unwrap()
    };

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut draw_text = DrawText::new(device.clone(), queue.clone(), swapchain.clone(), &images);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;
    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
    };
    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    loop {
        previous_frame_end.cleanup_finished();

        let dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        if recreate_swapchain {
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);
            // RECREATE DRAWTEXT ON RESIZE
            draw_text = DrawText::new(
                device.clone(),
                queue.clone(),
                swapchain.clone(),
                &new_images,
            );
            // RECREATE DRAWTEXT ON RESIZE END

            recreate_swapchain = false;
        }

        let mut vertices = Vec::new();

        move_graph_nodes(&mut a_graph, root);

        draw_graph(
            & a_graph,
            &mut draw_text,
            &mut vertices,
            Vector2::new(dimensions[0] as f32, dimensions[1] as f32),
        );

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()]; // CHANGED TO BLACK BACKGROUND
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                vertices.iter().cloned(),
            )
            .unwrap()
        };

        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                // DRAW THE TEXT
                .draw_text(&mut draw_text, image_num)
                // DRAW THE TEXT END
                .build()
                .unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => done = true,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => { /*todo: use for click events.*/ }
            _ => (),
        });
        if done {
            return;
        }
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
