from pathlib import Path


SCHEMA_TEMPLATE = {
    "static": {
        "dirname": None,
        "subdirs": {
            "images": None,
            "depths": None,
        }
    },
    "dynamic": {
        "dirname": None,
        "subdirs": {
            "images": None,
            "depths": None,
            "masks": None,
            "confs": None
        }
    },
    "extra": {
        "dirname": None,
        "subdirs": {}
    },
    "cameras": {
        "dirname": None,
        "filenames": {
            "intrinsics": None,
            "trajectory": None
        }
    }
}


def assert_schema_structure_match(schema: dict, template: dict):
    if not isinstance(schema, dict):
        raise TypeError("Schema must be a dict")
    
    for key, template_value in template.items():
        if key not in schema:
            raise ValueError(f"Missing key: {key}")

        value = schema[key]
        if isinstance(template_value, dict):
            assert_schema_structure_match(value, template_value)

def validate_schema(schema: dict):
    assert_schema_structure_match(schema, SCHEMA_TEMPLATE)

def parse_schema(root: Path, schema: dict):
    validate_schema(schema)

    paths = {}
    for role in ("static", "dynamic", "extra"):
        role_spec = schema.get(role)

        dirname = role_spec.get("dirname")
        subdirs = role_spec.get("subdirs")

        paths[role] = {}
        for modality, subdir in subdirs.items():
            paths[role][modality] = root / dirname / subdir

    cameras = schema.get("cameras")
    cameras_dirname = root / cameras.get("dirname")
    cameras_filenames = cameras.get("filenames")

    paths["cameras"] = {
        name: cameras_dirname / file_name for name, file_name in cameras_filenames.items()
    }

    return paths