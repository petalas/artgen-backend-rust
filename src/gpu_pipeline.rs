use std::{borrow::Cow, mem};

use wgpu::{
    vertex_attr_array, BlendState, ComputePipeline, Device, Gles3MinorVersion, InstanceFlags,
    SubmissionIndex,
};

use crate::{
    buffer_dimensions::BufferDimensions,
    models::drawing::Drawing,
    settings::{MAX_ERROR_PER_PIXEL, PER_POINT_MULTIPLIER},
    texture_wrapper::TextureWrapper,
};



#[derive(Debug)]
pub struct GpuPipeline {
    
}

impl GpuPipeline {
   
}
