"""
A stand-in for nnInteractiveInferenceSession that enforces the real API's shapes.

The weights live on the cluster, so the real session cannot run here. What can
be checked without them is the part that was actually broken: the shapes handed
to set_image and set_target_buffer, and whether the mask is written back in the
input's orientation.

The contract below is taken from nni_dsc_foldall.py, which is the script that
produced correct DSC on the cluster:

    image      = data['imgs']                 # bare 3-D (H, W, D)
    target_buf = torch.zeros(image.shape)     # 3-D, no channel axis
    session.set_image(image[None])            # 4-D, channel axis added

So: image 4-D, buffer 3-D, and the two must agree on the spatial dims. The
server previously passed a 5-D image and a 4-D buffer, which is why this tool
had never once run to completion.
"""
import numpy as np


class nnInteractiveInferenceSession:
    def __init__(self, device=None, use_torch_compile=False, verbose=False,
                 torch_n_threads=None, do_autozoom=True, use_pinned_memory=True):
        self.device = device
        self._image = None
        self._buffer = None
        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        self._folder = None
        self._fold = None

    def initialize_from_trained_model_folder(self, folder, use_fold=None):
        # fold='all' is not cosmetic: fold=0 scores about 0.33 DSC on this task.
        if use_fold != "all":
            raise AssertionError(f"expected use_fold='all', got {use_fold!r}")
        self._folder, self._fold = folder, use_fold

    def set_image(self, img):
        if img.ndim != 4:
            raise AssertionError(
                f"set_image expects (C, H, W, D); got {img.ndim}-D array {img.shape}")
        self._image = img

    def set_target_buffer(self, buf):
        if buf.ndim != 3:
            raise AssertionError(
                f"set_target_buffer expects (H, W, D); got {buf.ndim}-D tensor {tuple(buf.shape)}")
        self._buffer = buf

    def reset_interactions(self):
        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []

    def add_bbox_interaction(self, bbox, include_interaction=True, run_prediction=False):
        if self._image is None or self._buffer is None:
            raise AssertionError("add_bbox_interaction before set_image/set_target_buffer")
        spatial = self._image.shape[-3:]
        if tuple(self._buffer.shape) != tuple(spatial):
            raise AssertionError(
                f"buffer {tuple(self._buffer.shape)} does not match image spatial dims {tuple(spatial)}")
        for i, (lo, hi) in enumerate(bbox):
            if not (0 <= lo < hi <= spatial[i]):
                raise AssertionError(f"bbox axis {i} [{lo},{hi}) outside extent {spatial[i]}")
        self._bbox = bbox
        self.new_interaction_centers.append(
            [int((lo + hi) // 2) for lo, hi in bbox])
        self.new_interaction_zoom_out_factors.append(1.0)

    def _predict(self):
        # Fill the box so the caller sees a non-zero, positionally correct mask.
        (z0, z1), (y0, y1), (x0, x1) = self._bbox
        arr = self._buffer.numpy()
        arr[z0:z1, y0:y1, x0:x1] = 1
        return arr
