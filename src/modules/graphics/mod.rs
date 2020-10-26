use crate::font;
use nalgebra::Vector2;
use png;
use std::{io::Cursor, iter, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, DeviceLocalBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, DrawIndirectCommand, DynamicState, CommandBuffer},
    descriptor::{
        descriptor_set::{DescriptorSet, PersistentDescriptorSet},
        pipeline_layout::PipelineLayout,
        PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{Dimensions, ImageUsage, ImmutableImage, SwapchainImage},
    instance::{Instance, PhysicalDevice},
    pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline, GraphicsPipelineAbstract},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain,
    swapchain::{
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
        SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

const MAX_RECTANGLE_COUNT: usize = 10; //how many windows can be rendered at once
const MAX_GLYPH_COUNT: usize = MAX_RECTANGLE_COUNT * 10; //how many letters we can render

#[derive(Default, Debug, Clone, Copy)]
pub struct Rectangle {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 3],
    pub padding: f32, //padding to comply with std140 rules
}

#[derive(Default, Debug, Clone, Copy)]
pub struct TextCharacter {
    pub character: u32,
    pub scale: f32,
    pub position: [f32; 2],
    pub color: [f32; 3],
    padding: f32, //padding to comply with std140 rules
}

#[derive(Default, Debug, Clone, Copy)]
pub struct PolygonVertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
    padding: [f32; 3], //padding to comply with std140 rules TODO: check if this is necessary for device local vertexbuffers.
}
vulkano::impl_vertex!(PolygonVertex, position, color);

#[derive(Default, Debug, Clone, Copy)]
struct TextVertex {
    pub render_position: [f32; 2],
    pub glyph_position: [f32; 2],
    pub color: [f32; 3],
    padding: f32, //padding to comply with std140 rules
}
vulkano::impl_vertex!(TextVertex, render_position, glyph_position, color);

pub mod rectangle_compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/modules/graphics/rectangle_compute_shader.comp"
    }
}

pub mod polygon_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/modules/graphics/polygon_vertex_shader.vert"
    }
}

pub mod polygon_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/modules/graphics/polygon_fragment_shader.frag"
    }
}

pub mod text_compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/modules/graphics/text_compute_shader.comp"
    }
}

pub mod text_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/modules/graphics/text_vertex_shader.vert"
    }
}

pub mod text_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/modules/graphics/text_fragment_shader.frag"
    }
}

pub fn pixel_to_screen_coordinates(
    position: Vector2<f32>,
    window_dimensions: Vector2<f32>,
) -> Vector2<f32> {
    position.zip_map(&window_dimensions, |p, w| 2.0 / w * p - 1.0)
}

pub fn push_string(
    string: &str,
    height: f32,
    mut position: [f32; 2],
    color: [f32; 3],
    buffer: &mut Vec<TextCharacter>,
) {
    //TODO: only supports ascii.
    let original_x_position = position[0];
    let scale = height / font::LINE_HEIGHT;
    for character in string.chars() {
        match character {
            '\n' => {
                //newline
                position[0] = original_x_position;
                position[1] += height;
            }
            ' '..='~' => {
                //regular ascii character
                buffer.push(TextCharacter {
                    character: character as u32,
                    scale: scale,
                    position: position,
                    color: color,
                    padding: 0.0,
                });
                position[0] += font::GLYPH_LAYOUTS[character as usize].advance * scale;
            }
            _ => {
                //not renderable characters
            }
        }
    }
}

pub struct Renderer {
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    swapchain: Arc<Swapchain<Window>>,

    dynamic_state: DynamicState,

    rectangle_compute_pipeline:
        Arc<ComputePipeline<PipelineLayout<rectangle_compute_shader::Layout>>>,
    rectangle_pool: CpuBufferPool<Rectangle>,
    rectangle_indirect_args_pool: CpuBufferPool<DrawIndirectCommand>,
    polygon_vertex_buffer: Arc<DeviceLocalBuffer<[PolygonVertex]>>,

    polygon_render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    polygon_render_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    polygon_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    text_compute_pipeline: Arc<ComputePipeline<PipelineLayout<text_compute_shader::Layout>>>,
    text_character_pool: CpuBufferPool<TextCharacter>,
    text_glyph_buffer: Arc<ImmutableBuffer<[font::GlyphLayout; font::GLYPH_LAYOUTS.len()]>>,
    text_indirect_args_pool: CpuBufferPool<DrawIndirectCommand>,
    text_vertex_buffer: Arc<DeviceLocalBuffer<[TextVertex]>>,

    text_render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    text_render_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    text_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    text_set: Arc<dyn DescriptorSet + Send + Sync>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
}

impl Renderer {
    pub fn new(event_loop: &EventLoop<()>) -> Renderer {
        //objects used by all renderpasses
        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &extensions, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();
        let window = surface.window();
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
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
        let initial_dimensions: [u32; 2] = window.inner_size().into();
        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                initial_dimensions,
                1,
                ImageUsage::color_attachment(),
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap()
        };

        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };

        //objects uses by polygon compute pass
        let rectangle_pool: CpuBufferPool<Rectangle> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let rectangle_indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let polygon_vertex_buffer: Arc<DeviceLocalBuffer<[PolygonVertex]>> =
            DeviceLocalBuffer::array(
                device.clone(),
                MAX_RECTANGLE_COUNT * 6, //TODO: change buffer size at runtime, when more than MAX_RECTANGLE_COUNT.
                BufferUsage::all(),
                vec![queue.family()],
            )
            .unwrap();
        let rectangle_compute_shader =
            rectangle_compute_shader::Shader::load(device.clone()).unwrap();
        let rectangle_compute_pipeline = Arc::new(
            ComputePipeline::new(
                device.clone(),
                &rectangle_compute_shader.main_entry_point(),
                &(),
            )
            .unwrap(),
        );

        //objects used by polygon render pass
        let polygon_render_pass = Arc::new(
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
        let polygon_vertex_shader = polygon_vertex_shader::Shader::load(device.clone()).unwrap();
        let polygon_fragment_shader =
            polygon_fragment_shader::Shader::load(device.clone()).unwrap();
        let polygon_render_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<PolygonVertex>()
                .vertex_shader(polygon_vertex_shader.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(polygon_fragment_shader.main_entry_point(), ())
                .render_pass(Subpass::from(polygon_render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );
        let polygon_framebuffers = Self::window_size_dependent_setup(
            &images,
            polygon_render_pass.clone(),
            &mut dynamic_state,
        );

        //objects used by text compute pass
        let text_character_pool: CpuBufferPool<TextCharacter> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let (text_glyph_buffer, glyph_layout_future) =
            ImmutableBuffer::from_data(font::GLYPH_LAYOUTS, BufferUsage::all(), queue.clone()).unwrap();
        let text_indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let text_vertex_buffer: Arc<DeviceLocalBuffer<[TextVertex]>> = DeviceLocalBuffer::array(
            device.clone(),
            MAX_GLYPH_COUNT * 6, //TODO: change buffer size at runtime, when more than MAX_GLYPH_COUNT.
            BufferUsage::all(),
            vec![queue.family()],
        )
        .unwrap();
        let text_compute_shader = text_compute_shader::Shader::load(device.clone()).unwrap();
        let text_compute_pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &text_compute_shader.main_entry_point(), &())
                .unwrap(),
        );

        //objects used by text render pass
        let text_render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: Load,
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
        let text_vertex_shader = text_vertex_shader::Shader::load(device.clone()).unwrap();
        let text_fragment_shader = text_fragment_shader::Shader::load(device.clone()).unwrap();
        let text_render_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<TextVertex>()
                .vertex_shader(text_vertex_shader.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(text_fragment_shader.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(Subpass::from(text_render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );
        let text_framebuffers = Self::window_size_dependent_setup(
            &images,
            text_render_pass.clone(),
            &mut dynamic_state,
        );

        let (text_set, glyph_atlas_future) = {
            let (glyph_atlas, glyph_atlas_future) = {
                let png_bytes = include_bytes!("../../font/deja_vu_sans_mono.png").to_vec();
                let cursor = Cursor::new(png_bytes);
                let decoder = png::Decoder::new(cursor);
                let (info, mut reader) = decoder.read_info().unwrap();
                let dimensions = Dimensions::Dim2d {
                    width: info.width,
                    height: info.height,
                };
                let mut image_data = Vec::new();
                image_data.resize((info.width * info.height * 4) as usize, 0);
                reader.next_frame(&mut image_data).unwrap();
                ImmutableImage::from_iter(
                    image_data.iter().cloned(),
                    dimensions,
                    Format::R8G8B8A8Srgb, //TODO: change to r8 file format.
                    queue.clone(),
                )
                .unwrap()
            };
            let sampler = Sampler::new(
                device.clone(),
                Filter::Linear,
                Filter::Linear,
                MipmapMode::Nearest,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .unwrap();
            let layout = text_render_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap();
            let set = Arc::new(
                PersistentDescriptorSet::start(layout.clone())
                    .add_sampled_image(glyph_atlas.clone(), sampler.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            (set, glyph_atlas_future)
        };

        //join all futures for loading buffers to the gpu. flush them and wait till they're done.
        glyph_layout_future.join(glyph_atlas_future).then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        //move all the stuff we need to keep for rendering in the renderer struct.
        Renderer {
            device: device.clone(),
            queue: queue,
            swapchain: swapchain,

            dynamic_state: dynamic_state,
            rectangle_compute_pipeline: rectangle_compute_pipeline,
            rectangle_pool: rectangle_pool,
            rectangle_indirect_args_pool: rectangle_indirect_args_pool,
            polygon_vertex_buffer: polygon_vertex_buffer,

            polygon_render_pass: polygon_render_pass,
            polygon_render_pipeline: polygon_render_pipeline,
            polygon_framebuffers: polygon_framebuffers,

            text_compute_pipeline: text_compute_pipeline,
            text_character_pool: text_character_pool,
            text_glyph_buffer: text_glyph_buffer,
            text_indirect_args_pool: text_indirect_args_pool,
            text_vertex_buffer: text_vertex_buffer,
            text_render_pass: text_render_pass,
            text_render_pipeline: text_render_pipeline,
            text_framebuffers: text_framebuffers,
            text_set: text_set,

            previous_frame_end: Some(sync::now(device.clone()).boxed()),
            recreate_swapchain: false,
        }
    }

    pub fn render(
        &mut self,
        text_character_buffer: Vec<TextCharacter>,
        rectangle_buffer: Vec<Rectangle>,
        window_resized: bool,
    ) {
        self.recreate_swapchain |= window_resized;

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let dimensions: [u32; 2] = self.swapchain.surface().window().inner_size().into();

        if self.recreate_swapchain {
            let (new_swapchain, new_images) =
                match self.swapchain.recreate_with_dimensions(dimensions) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => return, //TODO: return replaces continue?
                    Err(err) => panic!("{:?}", err),
                };

            self.swapchain = new_swapchain;
            self.polygon_framebuffers = Self::window_size_dependent_setup(
                &new_images,
                self.polygon_render_pass.clone(),
                &mut self.dynamic_state,
            );
            self.text_framebuffers = Self::window_size_dependent_setup(
                &new_images,
                self.text_render_pass.clone(),
                &mut self.dynamic_state,
            );

            self.recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return; //TODO: return replaces continue?
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let clear_values = vec![[0.1, 0.1, 0.1, 1.0].into()];

        //rectangle compute pass stuff
        let rectangle_indirect_args = self
            .rectangle_indirect_args_pool
            .chunk(iter::once(DrawIndirectCommand {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }))
            .unwrap();
        let rectangle_compute_descriptor_set = {
            let rectangle_buffer = self.rectangle_pool.chunk(rectangle_buffer).unwrap();
            let layout = self
                .rectangle_compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap();
            Arc::new(
                PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(rectangle_buffer.clone())
                    .unwrap()
                    .add_buffer(self.polygon_vertex_buffer.clone())
                    .unwrap()
                    .add_buffer(rectangle_indirect_args.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        };

        //text compute pass stuff
        let text_indirect_args = self
            .text_indirect_args_pool
            .chunk(iter::once(DrawIndirectCommand {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }))
            .unwrap();
        let text_compute_descriptor_set = {
            let text_character_buffer = self
                .text_character_pool
                .chunk(text_character_buffer)
                .unwrap();
            let layout = self
                .text_compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap();
            Arc::new(
                PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(text_character_buffer.clone())
                    .unwrap()
                    .add_buffer(self.text_glyph_buffer.clone())
                    .unwrap()
                    .add_buffer(self.text_vertex_buffer.clone())
                    .unwrap()
                    .add_buffer(text_indirect_args.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        };

        let rectangle_compute_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.device.clone(),
                self.queue.family(),
            )
            .unwrap();
            builder
                .dispatch(
                    [1, 1, 1],
                    self.rectangle_compute_pipeline.clone(),
                    rectangle_compute_descriptor_set,
                    (),
                )
                .unwrap();
            builder.build().unwrap()
        };

        let polygon_render_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.device.clone(),
                self.queue.family(),
            )
            .unwrap();
            builder
                .begin_render_pass(
                    self.polygon_framebuffers[image_num].clone(),
                    false,
                    clear_values,
                )
                .unwrap()
                .draw(
                    self.polygon_render_pipeline.clone(),
                    &self.dynamic_state,
                    vec![self.polygon_vertex_buffer.clone()],
                    (),
                    (),

                )
                // .draw_indirect(
                //     self.polygon_render_pipeline.clone(),
                //     &self.dynamic_state,
                //     vec![self.polygon_vertex_buffer.clone()],
                //     rectangle_indirect_args.clone(),
                //     (),
                //     (),
                // )
                .unwrap()
                .end_render_pass()
                .unwrap();
            builder.build().unwrap()
        };

        let text_compute_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.device.clone(),
                self.queue.family(),
            )
            .unwrap();
            builder
                .dispatch(
                    [1, 1, 1],
                    self.text_compute_pipeline.clone(),
                    text_compute_descriptor_set.clone(),
                    (),
                )
                .unwrap();
            builder.build().unwrap()
        };

        let text_render_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.device.clone(),
                self.queue.family(),
            )
            .unwrap();
            builder
                .begin_render_pass(
                    self.text_framebuffers[image_num].clone(),
                    false,
                    vec![ClearValue::None],
                )
                .unwrap()
                .draw(
                    self.text_render_pipeline.clone(),
                    &self.dynamic_state,
                    vec![self.text_vertex_buffer.clone()],
                    self.text_set.clone(),
                    (),

                )
                // .draw_indirect(
                //     self.text_render_pipeline.clone(),
                //     &self.dynamic_state,
                //     vec![self.text_vertex_buffer.clone()],
                //     text_indirect_args.clone(),
                //     self.text_set.clone(),
                //     (),
                // )
                .unwrap()
                .end_render_pass()
                .unwrap();
            builder.build().unwrap()
        };

        //by creating the compute futures seperately and joining them at appropriate places with the render futures,
        //the compute passes are run simultaniously.
        let rectangle_compute_future = rectangle_compute_command_buffer.execute(self.queue.clone()).unwrap();
        let text_compute_future = text_compute_command_buffer.execute(self.queue.clone()).unwrap();
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .join(rectangle_compute_future)
            .then_execute(self.queue.clone(), polygon_render_command_buffer)
            .unwrap()
            .join(text_compute_future)
            .then_execute(self.queue.clone(), text_render_command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    //this method is called once during initialization, then again whenever the window is resized.
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
}
