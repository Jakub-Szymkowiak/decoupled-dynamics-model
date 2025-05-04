#### Input Structure
The input directory should be structured as follows:

```
input-dir/
├── pred_traj.txt
├── pred_intrinsics.txt
├── static/
│   ├── images/
│   │   └── frame_<id:04d>.png
│   ├── confs/
│   │   └── conf_<id>.npy
│   └── depths/
│       └── frame_<id:04d>.npy
├── dynamic/
│   ├── images/
│   │   └── frame_<id:04d>.png
│   ├── confs/
│   │   └── conf_<id>.npy
│   ├── depths/
│   │   └── frame_<id:04d>.npy
│   └── masks/
│       └── dynamic_mask_<id>.png
```

whre `id` ranges from `0` to `<num_frames> - 1`:
- `input-dir/static` should contain MonST3R output from the inpainted video run
- `input_dir/dynamic` should contain MonST3R output from the original video run
- for `input-dir/pred_traj.txt` and `input-dir/pred_intrinsics.txt`, it should not matter which MonST3R run they come from

#### TODOs
1. Check whether there is difference between trajectory and intrinsics predictions from the original and inpainted video MonST3R runs.
