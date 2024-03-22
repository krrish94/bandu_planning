import bpy


def clear_all():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def load_mesh(filepath, obj_name):
    bpy.ops.import_scene.obj(filepath=filepath)
    imported_objs = [obj for obj in bpy.context.selected_objects]
    mesh_obj = None
    for obj in imported_objs:
        if obj.type == "MESH":
            mesh_obj = obj
            break
    mesh_obj.name = obj_name
    return mesh_obj


def decimate_mesh(obj_name, reduction_factor, target_faces):
    """Decimate the mesh object iteratively by the reduction_factor until the
    number of faces is less than target_faces.

    Parameters:
        obj_name (str): Name of the mesh object to be decimated.
        reduction_factor (float): Fraction of faces to remove in each iteration (e.g., 0.1 to remove 10%).
        target_faces (int): Target number of faces.
    """

    # Set the object as active
    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]
    obj = bpy.data.objects[obj_name]
    obj.select_set(True)

    # Ensure we're in object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Apply decimation while the face count is above target_faces
    while len(obj.data.polygons) > target_faces:
        bpy.ops.object.modifier_add(type="DECIMATE")
        decimate_mod = obj.modifiers[-1]
        decimate_mod.ratio = 1.0 - reduction_factor
        bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    obj.select_set(False)


clear_all()  # Clear the scene
load_mesh(
    "/home/krishna/code/bandu_planning/bandu_stacking/models/random_models/teddy.obj",
    "my_mesh",
)
decimate_mesh("my_mesh", 0.1, 50)
bpy.ops.export_scene.obj(
    filepath="/home/krishna/code/bandu_planning/bandu_stacking/models/random_models/teddy_decimated.obj"
)
