import copy
import os
from os import listdir
from os.path import isfile, join

bandu_model_path = "./models/bandu_models"
bandu_filenames = [
    f for f in listdir(bandu_model_path) if isfile(join(bandu_model_path, f))
]
bandu_stls = [f for f in bandu_filenames if f.endswith("stl")]
with open(os.path.join(bandu_model_path, "template.urdf"), "r") as file:
    template = file.read()
for bandu_stl in bandu_stls:
    copy_template = copy.copy(template)

    write_string = copy_template.replace(
        "<template_name>", bandu_stl.replace(".stl", "")
    )
    nurdf = bandu_stl.replace(".stl", "") + ".urdf"
    with open(os.path.join(bandu_model_path, nurdf), "w") as text_file:
        text_file.write(write_string)
