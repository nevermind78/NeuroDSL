
module Backend
    const CUDA_AVAILABLE = try
        using CUDA; CUDA.functional()
    catch; false; end

    struct CPUDevice  end
    struct CUDADevice end
    active_device() = CUDA_AVAILABLE ? CUDADevice() : CPUDevice()

    zeros32(::CPUDevice,  dims...) = zeros(Float32, dims...)
    zeros32(::CUDADevice, dims...) = CUDA.zeros(Float32, dims...)
    ones32(::CPUDevice,   dims...) = ones(Float32, dims...)
    ones32(::CUDADevice,  dims...) = CUDA.ones(Float32, dims...)
    rand32(::CPUDevice,   dims...) = rand(Float32, dims...)
    rand32(::CUDADevice,  dims...) = CUDA.rand(Float32, dims...)
    randn32(::CPUDevice,  dims...) = randn(Float32, dims...)
    randn32(::CUDADevice, dims...) = CUDA.randn(Float32, dims...)

    to_device(::CPUDevice, x) = x isa AbstractArray ? Array(x) : x
    function to_device(::CUDADevice, x)
        CUDA_AVAILABLE || return (x isa AbstractArray ? Array(x) : x)
        x isa CUDA.CuArray  && return x
        x isa AbstractArray && return CUDA.cu(x)
        return x
    end

    to_cpu(x) = (x isa AbstractArray && !(x isa Array)) ? Array(x) : x
    device_of(x::Array) = CPUDevice()
    device_of(x)        = (CUDA_AVAILABLE && x isa CUDA.CuArray) ? CUDADevice() : CPUDevice()

    export CPUDevice, CUDADevice, active_device,
           zeros32, ones32, rand32, randn32,
           to_device, to_cpu, device_of
end
