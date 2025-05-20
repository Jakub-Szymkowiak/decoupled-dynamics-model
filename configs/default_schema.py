DEFAULT_SCHEMA = {
    "static": {
        "dirname": "static", 
        "subdirs": {
            "images": "images",
            "depths": "depths",
        }
    },
    "dynamic": {
        "dirname": "dynamic",
        "subdirs": {
            "images": "images",
            "depths": "depths",
            "masks": "masks",
            "confs": "confs"
        }
    },
    "extra": {
        "dirname": "extra",
        "subdirs": {
            "flow": "optical_flow"
        }
    },
    "cameras": {
        "dirname": "cameras",
        "filenames": {
            "intrinsics": "pred_intrinsics.txt",
            "trajectory": "pred_traj.txt"
        }
    }
}