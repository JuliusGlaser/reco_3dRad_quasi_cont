import cupy as cp
import cupyx.scipy.ndimage as cpx_ndi

print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
x = cp.ones((2000, 2000))
y = cp.dot(x, x.T)
cp.cuda.Stream.null.synchronize()
print("OK, result on device:", y.device)
