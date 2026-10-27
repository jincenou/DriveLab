# PyTorch3D compatibility subset

The two retained source files are copied byte-for-byte from [`facebookresearch/pytorch3d@f34104c`](https://github.com/facebookresearch/pytorch3d/tree/f34104cf6ebefacd7b7e07955ee7aaa823e616ac) (release `v0.7.6`):

- `common/datatypes.py`
- `transforms/rotation_conversions.py`

PyTorch3D is BSD-licensed; its license is preserved in `LICENSE`. The smaller `__init__.py` files and native-PyTorch KNN fallback are 4DAnyone adapter code. They expose only the rotation/geometry surface imported by classic GVHMR inference and intentionally omit PyTorch3D's renderer and compiled operators.
