use nalgebra::Vector2;
use png;
use std::{io::Cursor, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    descriptor::descriptor_set::{DescriptorSet, PersistentDescriptorSet},
    device::{Device, DeviceExtensions},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{Dimensions, ImmutableImage, SwapchainImage, ImageUsage},
    instance::{Instance, PhysicalDevice},
    pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
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

#[derive(Default, Debug, Clone, Copy)]
pub struct WindowVertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(WindowVertex, position, color);

#[derive(Default, Debug, Clone)]
struct TextVertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(TextVertex, position);

pub mod window_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/modules/graphics/window_vertex_shader.vert"
    }
}

pub mod window_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/modules/graphics/window_fragment_shader.frag"
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

const MAX_VERTEX_COUNT: usize = 1000;

pub struct Renderer {
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    swapchain: Arc<Swapchain<Window>>,

    dynamic_state: DynamicState,

    window_render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    window_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    window_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    window_vertex_buffer: CpuBufferPool<[WindowVertex; MAX_VERTEX_COUNT]>,

    text_render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    text_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    text_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    text_vertex_buffer: Arc<ImmutableBuffer<[TextVertex]>>,
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

        //objects used by window render pass
        let window_render_pass = Arc::new(
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
        let window_vertex_shader = window_vertex_shader::Shader::load(device.clone()).unwrap();
        let window_fragment_shader = window_fragment_shader::Shader::load(device.clone()).unwrap();
        let window_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<WindowVertex>()
                .vertex_shader(window_vertex_shader.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(window_fragment_shader.main_entry_point(), ())
                .render_pass(Subpass::from(window_render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );
        let window_framebuffers = Self::window_size_dependent_setup(
            &images,
            window_render_pass.clone(),
            &mut dynamic_state,
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
        let text_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<TextVertex>()
                .vertex_shader(text_vertex_shader.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(text_fragment_shader.main_entry_point(), ())
                .render_pass(Subpass::from(text_render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );
        let text_framebuffers = Self::window_size_dependent_setup(
            &images,
            text_render_pass.clone(),
            &mut dynamic_state,
        );

        let (text_vertex_buffer, _) = ImmutableBuffer::from_iter(
            [
                TextVertex {
                    position: [-0.5, -0.5],
                },
                TextVertex {
                    position: [-0.5, 0.5],
                },
                TextVertex {
                    position: [0.5, -0.5],
                },
                TextVertex {
                    position: [0.5, 0.5],
                },
            ]
            .iter()
            .cloned(),
            BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();

        let (text_set, tex_future) = {
            let (texture, tex_future) = {
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
                    Format::R8G8B8A8Srgb,
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
            let layout = text_pipeline.layout().descriptor_set_layout(0).unwrap();
            let set = Arc::new(
                PersistentDescriptorSet::start(layout.clone())
                    .add_sampled_image(texture.clone(), sampler.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            (set, tex_future)
        };

        let previous_frame_end = Some(Box::new(tex_future) as Box<dyn GpuFuture>);

        //move all the stuff we need to keep for rendering in the renderer struct.
        Renderer {
            device: device.clone(),
            queue: queue,
            swapchain: swapchain,

            dynamic_state: dynamic_state,

            window_render_pass: window_render_pass,
            window_pipeline: window_pipeline,
            window_framebuffers: window_framebuffers,
            window_vertex_buffer: CpuBufferPool::vertex_buffer(device.clone()),

            text_render_pass: text_render_pass,
            text_pipeline: text_pipeline,
            text_framebuffers: text_framebuffers,
            text_vertex_buffer: text_vertex_buffer,
            text_set: text_set,

            previous_frame_end: previous_frame_end,
            recreate_swapchain: false,
        }
    }

    pub fn render(&mut self, vertices: &mut Vec<WindowVertex>, window_resized: bool) {
        self.recreate_swapchain |= window_resized;

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let dimensions: [u32; 2] = self.swapchain.surface().window().inner_size().into();

        if self.recreate_swapchain {
            let (new_swapchain, new_images) =
                match self.swapchain.recreate_with_dimensions(dimensions) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => return, //todo: return replaces continue?
                    Err(err) => panic!("{:?}", err),
                };

            self.swapchain = new_swapchain;
            self.window_framebuffers = Self::window_size_dependent_setup(
                &new_images,
                self.window_render_pass.clone(),
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
                    return; //todo: return replaces continue?
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let vertex_buffer = {
            let mut vertex_array: [WindowVertex; MAX_VERTEX_COUNT] =
                [Default::default(); MAX_VERTEX_COUNT]; //todo: is there a better way to copy vertices to vertex_buffer?
            for (index, vertex) in vertices.iter().enumerate() {
                vertex_array[index] = *vertex;
            }
            Arc::new(self.window_vertex_buffer.next(vertex_array).unwrap())
        };
        let clear_values = vec![[0.0, 0.0, 0.5, 1.0].into()];

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
        .begin_render_pass(
            self.window_framebuffers[image_num].clone(),
            false,
            clear_values,
        )
        .unwrap()
        .draw(
            self.window_pipeline.clone(),
            &self.dynamic_state,
            vec![vertex_buffer],
            (),
            (),
        )
        .unwrap()
        .end_render_pass()
        .unwrap()
        .begin_render_pass(
            self.text_framebuffers[image_num].clone(),
            false,
            vec![ClearValue::None],
        )
        .unwrap()
        .draw(
            self.text_pipeline.clone(),
            &self.dynamic_state,
            vec![self.text_vertex_buffer.clone()],
            self.text_set.clone(),
            (),
        )
        .unwrap()
        .end_render_pass()
        .unwrap();
        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
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
