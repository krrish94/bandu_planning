import os
from xml.etree import ElementTree

from tqdm import tqdm


def modify_urdf(input_path, output_path):
    # Parse the XML file
    tree = ElementTree.parse(input_path)
    root = tree.getroot()

    # Modify the filename for the <visual> tag's <mesh>
    for visual in root.findall(".//visual/geometry/mesh"):
        filename = visual.get("filename")
        if filename and not filename.endswith("_decimated.obj"):
            base, ext = os.path.splitext(filename)
            visual.set("filename", f"{base}_decimated{ext}")

    # Modify the filename for the <collision> tag's <mesh>
    for collision in root.findall(".//collision/geometry/mesh"):
        filename = collision.get("filename")
        if filename and not filename.endswith("_decimated.obj"):
            base, ext = os.path.splitext(filename)
            collision.set("filename", f"{base}_decimated{ext}")

    # Save the modified XML content to the output file
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        tree.write(f, encoding="utf-8")


def duplicate_urdf(input_path, output_path):
    # Parse the XML file
    tree = ElementTree.parse(input_path)

    # Save the XML content to the output file
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        tree.write(f, encoding="utf-8")


def main():
    input_directory = (
        "/home/krishna/code/bandu_planning/bandu_stacking/models/bandu_models"
    )
    output_directory = (
        "/home/krishna/code/bandu_planning/bandu_stacking/models/bandu_simplified"
    )

    urdf_files = [
        "Barrell.urdf",
        "block.urdf",
        "Bowl.urdf",
        "Bridge.urdf",
        "cross.urdf",
        "cylinder_long.urdf",
        "cylinder_parallogram.urdf",
        "Diamond_Hex.urdf",
        "doughnut.urdf",
        "foundation.urdf",
        "J_Block.urdf",
        "Nut.urdf",
        "Vase_W_Hole.urdf",
        "wedge_short_stout.urdf",
    ]

    for urdf_file in tqdm(urdf_files, total=len(urdf_files)):
        input_path = os.path.join(input_directory, urdf_file)
        output_path = os.path.join(output_directory, urdf_file)

        modify_urdf(input_path, output_path)
        print(f"Modified and saved: {output_path}")

        # # Write the original urdf file to the output directory
        # # add "_original" to the filename (e.g., "Barrell.urdf" -> "Barrell_original.urdf")
        # output_path = os.path.join(
        #     output_directory, urdf_file.replace(".urdf", "_original.urdf")
        # )
        # duplicate_urdf(input_path, output_path)


if __name__ == "__main__":
    main()
