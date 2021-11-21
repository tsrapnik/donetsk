use crate::font;
use png;
use std::{io::Cursor, iter, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, DeviceLocalBuffer, ImmutableBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, DrawIndirectCommand, PrimaryCommandBuffer,
        SubpassContents,
    },
    descriptor_set::{DescriptorSet, PersistentDescriptorSet},
    device::{physical::PhysicalDevice, Device, DeviceExtensions},
    format::{ClearValue, Format},
    image::{
        view::ImageView, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount, SwapchainImage,
    },
    instance::Instance,
    pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline, PipelineBindPoint},
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain,
    swapchain::{AcquireError, Swapchain, SwapchainCreationError},
    sync,
    sync::{FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

//when set to true we measure different parts of the rendering loop and print it to the console. for
//this we have to let the cpu wait for the gpu to complete its tasks, so DEBUG_MODE has a
//performance penalty.
const DEBUG_MODE: bool = true;

const MAX_RECTANGLE_COUNT: u64 = 100; //how many windows can be rendered at once
const VERTICES_PER_RECTANGLE: u64 = 6;
const MAX_GLYPH_COUNT: u64 = MAX_RECTANGLE_COUNT * 100; //how many letters we can render
const VERTICES_PER_GLYPH: u64 = 6;

#[derive(Default, Debug, Clone, Copy)]
pub struct Rectangle {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 3],
    pub padding: f32, //padding to comply with vulkan alignment rules TODO: checkout library to cope with padding.
}

#[derive(Default, Debug, Clone, Copy)]
pub struct TextCharacter {
    pub character: u32,
    pub scale: f32,
    pub position: [f32; 2],
    pub color: [f32; 3],
    _padding: f32, //padding to comply with vulkan alignment rules
}

#[derive(Default, Debug, Clone, Copy)]
struct PolygonVertex {
    pub position: [f32; 2],
    pub padding_0: [f32; 2], //padding to comply with vulkan alignment rules
    pub color: [f32; 3],
    pub padding_1: f32, //padding to comply with vulkan alignment rules
}
vulkano::impl_vertex!(PolygonVertex, position, padding_0, color, padding_1);

#[derive(Default, Debug, Clone, Copy)]
struct TextVertex {
    pub render_position: [f32; 2],
    pub glyph_position: [f32; 2],
    pub color: [f32; 3],
    _padding: f32, //padding to comply with vulkan alignment rules
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
                    _padding: 0.0,
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
    viewport: Viewport,

    rectangle_compute_pipeline: Arc<ComputePipeline>,
    rectangle_pool: CpuBufferPool<Rectangle>,
    rectangle_indirect_args_pool: CpuBufferPool<DrawIndirectCommand>,
    polygon_vertex_buffer: Arc<DeviceLocalBuffer<[PolygonVertex]>>,

    polygon_render_pass: Arc<RenderPass>,
    polygon_render_pipeline: Arc<GraphicsPipeline>,
    polygon_framebuffers: Vec<Arc<dyn FramebufferAbstract>>,

    text_compute_pipeline: Arc<ComputePipeline>,
    text_character_pool: CpuBufferPool<TextCharacter>,
    text_glyph_buffer: Arc<ImmutableBuffer<[font::GlyphLayout; font::GLYPH_LAYOUTS.len()]>>,
    text_indirect_args_pool: CpuBufferPool<DrawIndirectCommand>,
    text_vertex_buffer: Arc<DeviceLocalBuffer<[TextVertex]>>,

    text_render_pass: Arc<RenderPass>,
    text_render_pipeline: Arc<GraphicsPipeline>,
    text_framebuffers: Vec<Arc<dyn FramebufferAbstract>>,
    text_set: Arc<dyn DescriptorSet>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
}

impl Renderer {
    pub fn new(event_loop: &EventLoop<()>) -> Renderer {
        //objects used by all renderpasses
        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, Version::V1_1, &extensions, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();
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
        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            let dimensions: [u32; 2] = surface.window().inner_size().into();
            Swapchain::start(device.clone(), surface.clone())
                .num_images(caps.min_image_count)
                .format(format)
                .dimensions(dimensions)
                .usage(ImageUsage::color_attachment())
                .sharing_mode(&queue)
                .composite_alpha(composite_alpha)
                .build()
                .unwrap()
        };

        //objects uses by polygon compute pass
        let rectangle_pool: CpuBufferPool<Rectangle> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let rectangle_indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let polygon_vertex_buffer: Arc<DeviceLocalBuffer<[PolygonVertex]>> =
            DeviceLocalBuffer::array(
                device.clone(),
                MAX_RECTANGLE_COUNT * VERTICES_PER_RECTANGLE, //TODO: change buffer size at runtime, when more than MAX_RECTANGLE_COUNT.
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
                None,
                |_| {},
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
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };
        let polygon_framebuffers =
            Self::window_size_dependent_setup(&images, polygon_render_pass.clone(), &mut viewport);

        //objects used by text compute pass
        let text_character_pool: CpuBufferPool<TextCharacter> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let (text_glyph_buffer, glyph_layout_future) =
            ImmutableBuffer::from_data(font::GLYPH_LAYOUTS, BufferUsage::all(), queue.clone())
                .unwrap();
        let text_indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
            CpuBufferPool::new(device.clone(), BufferUsage::all());
        let text_vertex_buffer: Arc<DeviceLocalBuffer<[TextVertex]>> = DeviceLocalBuffer::array(
            device.clone(),
            MAX_GLYPH_COUNT * VERTICES_PER_GLYPH, //TODO: change buffer size at runtime, when more than MAX_GLYPH_COUNT.
            BufferUsage::all(),
            vec![queue.family()],
        )
        .unwrap();
        let text_compute_shader = text_compute_shader::Shader::load(device.clone()).unwrap();
        let text_compute_pipeline = Arc::new(
            ComputePipeline::new(
                device.clone(),
                &text_compute_shader.main_entry_point(),
                &(),
                None,
                |_| {},
            )
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
        let text_framebuffers =
            Self::window_size_dependent_setup(&images, text_render_pass.clone(), &mut viewport);

        let (text_set, glyph_atlas_future) = {
            let (glyph_atlas, glyph_atlas_future) = {
                let png_bytes = include_bytes!("../../font/deja_vu_sans_mono.png").to_vec();
                let cursor = Cursor::new(png_bytes);
                let decoder = png::Decoder::new(cursor);
                let mut reader = decoder.read_info().unwrap();
                let info = reader.info();
                let dimensions = ImageDimensions::Dim2d {
                    width: info.width,
                    height: info.height,
                    array_layers: 1,
                };
                let mut image_data = Vec::new();
                image_data.resize((info.width * info.height * 4) as usize, 0);
                reader.next_frame(&mut image_data).unwrap();
                let (image, future) = ImmutableImage::from_iter(
                    image_data.iter().cloned(),
                    dimensions,
                    MipmapsCount::One,
                    Format::R8G8B8A8_SRGB, //TODO: change to r8 file format.
                    queue.clone(),
                )
                .unwrap();
                (ImageView::new(image).unwrap(), future)
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
                .descriptor_set_layouts()
                .get(0)
                .unwrap();
            let mut set_builder = PersistentDescriptorSet::start(layout.clone());
            set_builder
                .add_sampled_image(glyph_atlas.clone(), sampler.clone())
                .unwrap();
            let set = Arc::new(set_builder.build().unwrap());
            (set, glyph_atlas_future)
        };

        //join all futures for loading buffers to the gpu. flush them and wait till they're done.
        glyph_layout_future
            .join(glyph_atlas_future)
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        //move all the stuff we need to keep for rendering in the renderer struct.
        Renderer {
            device: device.clone(),
            queue: queue,
            swapchain: swapchain,
            viewport: viewport,

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

            previous_frame_end: Some(sync::now(device).boxed()),
            recreate_swapchain: false,
        }
    }

    pub fn render(
        &mut self,
        text_character_buffer: Vec<TextCharacter>,
        rectangle_buffer: Vec<Rectangle>,
        window_resized: bool,
    ) {
        let start_rendering = std::time::Instant::now();

        self.recreate_swapchain |= window_resized;

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            let dimensions: [u32; 2] = self.swapchain.surface().window().inner_size().into();
            let (new_swapchain, new_images) =
                match self.swapchain.recreate().dimensions(dimensions).build() {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => return, //TODO: return replaces continue?
                    Err(err) => panic!("{:?}", err),
                };

            self.swapchain = new_swapchain;
            self.polygon_framebuffers = Self::window_size_dependent_setup(
                &new_images,
                self.polygon_render_pass.clone(),
                &mut self.viewport,
            );
            self.text_framebuffers = Self::window_size_dependent_setup(
                &new_images,
                self.text_render_pass.clone(),
                &mut self.viewport,
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
        let rectangle_buffer_length = rectangle_buffer.len() as u32;
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
                .descriptor_set_layouts()
                .get(0)
                .unwrap();
            let mut set_builder = PersistentDescriptorSet::start(layout.clone());
            set_builder
                .add_buffer(Arc::new(rectangle_buffer))
                .unwrap()
                .add_buffer(self.polygon_vertex_buffer.clone())
                .unwrap()
                .add_buffer(Arc::new(rectangle_indirect_args.clone()))
                .unwrap();
            Arc::new(set_builder.build().unwrap())
        };

        //text compute pass stuff
        let text_character_buffer_length = text_character_buffer.len() as u32;
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
                .descriptor_set_layouts()
                .get(0)
                .unwrap();
            let mut set_builder = PersistentDescriptorSet::start(layout.clone());
            set_builder
                .add_buffer(Arc::new(text_character_buffer))
                .unwrap()
                .add_buffer(self.text_glyph_buffer.clone())
                .unwrap()
                .add_buffer(self.text_vertex_buffer.clone())
                .unwrap()
                .add_buffer(Arc::new(text_indirect_args.clone()))
                .unwrap();
            Arc::new(set_builder.build().unwrap())
        };

        let rectangle_compute_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .bind_pipeline_compute(self.rectangle_compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.rectangle_compute_pipeline.layout().clone(),
                    0,
                    rectangle_compute_descriptor_set,
                )
                .dispatch([rectangle_buffer_length, 1, 1])
                .unwrap();
            builder.build().unwrap()
        };

        let polygon_render_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    self.polygon_framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .set_viewport(0, [self.viewport.clone()])
                .bind_pipeline_graphics(self.polygon_render_pipeline.clone())
                .bind_vertex_buffers(0, self.polygon_vertex_buffer.clone())
                .draw_indirect(rectangle_indirect_args)
                .unwrap()
                .end_render_pass()
                .unwrap();
            builder.build().unwrap()
        };

        let text_compute_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .bind_pipeline_compute(self.text_compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.text_compute_pipeline.layout().clone(),
                    0,
                    text_compute_descriptor_set,
                )
                .dispatch([text_character_buffer_length, 1, 1])
                .unwrap();
            builder.build().unwrap()
        };

        let text_render_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .begin_render_pass(
                    self.text_framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    vec![ClearValue::None],
                )
                .unwrap()
                .set_viewport(0, [self.viewport.clone()])
                .bind_pipeline_graphics(self.text_render_pipeline.clone())
                .bind_vertex_buffers(0, self.text_vertex_buffer.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.text_render_pipeline.layout().clone(),
                    0,
                    self.text_set.clone(),
                )
                .draw_indirect(text_indirect_args)
                .unwrap()
                .end_render_pass()
                .unwrap();
            builder.build().unwrap()
        };

        //by creating the compute futures seperately and joining them at appropriate places with the render futures,
        //the compute passes are run simultaniously.
        let rectangle_compute_future = rectangle_compute_command_buffer
            .execute(self.queue.clone())
            .unwrap();
        let text_compute_future = text_compute_command_buffer
            .execute(self.queue.clone())
            .unwrap();
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .join(rectangle_compute_future)
            .join(text_compute_future) //TODO: why does joining just before executing text_render_command_buffer not work?
            .then_execute(self.queue.clone(), polygon_render_command_buffer)
            .unwrap()
            .then_execute(self.queue.clone(), text_render_command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        if DEBUG_MODE {
            let cpu_render_time = start_rendering.elapsed();
            println!("cpu render time: {:?}", cpu_render_time);

            //wait till gpu rendering finished, so we can measure its duration.
            future.unwrap().wait(None).unwrap();

            let gpu_render_time = start_rendering.elapsed() - cpu_render_time;
            println!("gpu render timer: {:?}", gpu_render_time);

            //provide dummy future, since rendering in fact already finished.
            self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        } else {
            //in non debug mode actually pass the future to the next frame, so we let the gpu run in
            //parallel.
            match future {
                Ok(future) => {
                    self.previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
                Err(e) => {
                    println!("{:?}", e);
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
            }
        }
    }

    //this method is called once during initialization, then again whenever the window is resized.
    fn window_size_dependent_setup(
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<dyn FramebufferAbstract>> {
        let dimensions = images[0].dimensions();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        images
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(view)
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<dyn FramebufferAbstract>
            })
            .collect::<Vec<_>>()
    }
}
