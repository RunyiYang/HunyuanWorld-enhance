#!/usr/bin/env python3
"""
Mesh Rendering Script for HunyuanWorld Output (Headless)
Loads and renders 3D meshes from different viewpoints using Open3D OffscreenRenderer
"""

import os
import numpy as np
import open3d as o3d
from PIL import Image


def main():
    # Set up the paths
    mesh_dir = "/home/runyi_yang/Gen3D/HunyuanWorld-enhance/examples/indoor_kitchen/with_mid_prompt"
    mesh_files = ["mesh_layer0.ply", "mesh_layer1.ply"]

    print(f"Working with meshes from: {mesh_dir}")
    print(f"Available mesh files: {mesh_files}")

    # Check if files exist
    for mesh_file in mesh_files:
        path = os.path.join(mesh_dir, mesh_file)
        if os.path.exists(path):
            print(f"✓ {mesh_file} exists")
        else:
            print(f"✗ {mesh_file} not found")

    # Load the mesh files
    meshes = []
    mesh_info = []

    for i, mesh_file in enumerate(mesh_files):
        mesh_path = os.path.join(mesh_dir, mesh_file)

        if os.path.exists(mesh_path):
            print(f"\nLoading {mesh_file}...")
            mesh = o3d.io.read_triangle_mesh(mesh_path)

            # Get mesh statistics
            num_vertices = len(mesh.vertices)
            num_triangles = len(mesh.triangles)
            has_colors = mesh.has_vertex_colors()

            print(f"  Vertices: {num_vertices}")
            print(f"  Triangles: {num_triangles}")
            print(f"  Has colors: {has_colors}")

            # Calculate bounding box
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            bbox_center = bbox.get_center()

            print(f"  Bounding box size: {bbox_size}")
            print(f"  Bounding box center: {bbox_center}")

            meshes.append(mesh)
            mesh_info.append({
                'name': mesh_file,
                'vertices': num_vertices,
                'triangles': num_triangles,
                'has_colors': has_colors,
                'bbox_size': bbox_size,
                'bbox_center': bbox_center
            })

            # Compute normals if not present
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
        else:
            print(f"Warning: {mesh_file} not found!")

    print(f"\nSuccessfully loaded {len(meshes)} meshes")

    if len(meshes) == 0:
        print("No meshes loaded, exiting.")
        return

    # Calculate overall bounding box for all meshes
    all_vertices = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        all_vertices.append(vertices)

    if all_vertices:
        combined_vertices = np.vstack(all_vertices)
        overall_center = np.mean(combined_vertices, axis=0)
        min_coords = np.min(combined_vertices, axis=0)
        max_coords = np.max(combined_vertices, axis=0)
        overall_size = max_coords - min_coords
    else:
        overall_center = np.array([0, 0, 0])
        overall_size = np.array([1, 1, 1])

    print(f"Overall scene center: {overall_center}")
    print(f"Overall scene size: {overall_size}")
    
    # Debug: Check if meshes are valid
    for i, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        print(f"Mesh {i} bounds: min={np.min(vertices, axis=0)}, max={np.max(vertices, axis=0)}")
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            print(f"Mesh {i} colors: min={np.min(colors)}, max={np.max(colors)}")

    # Set up output directory
    output_dir = os.path.join(mesh_dir, "rendered_views")
    os.makedirs(output_dir, exist_ok=True)

    # Generate cube map from inner mesh center
    print("Generating cube map from mesh center...")
    cube_map_images = render_cube_map(meshes, overall_center, resolution=1024)

    # Save cube map faces
    cube_faces = ['positive_x', 'negative_x', 'positive_y', 'negative_y', 'positive_z', 'negative_z']
    for i, face_name in enumerate(cube_faces):
        if i < len(cube_map_images):
            face_image = cube_map_images[i]
            face_pil = Image.fromarray((face_image * 255).astype(np.uint8))
            face_output_path = os.path.join(output_dir, f"cubemap_{face_name}.png")
            face_pil.save(face_output_path)
            print(f"  Saved cube map face: {face_output_path}")

    print(f"\nSuccessfully rendered cube map with {len(cube_map_images)} faces")

    # Create cube map layout
    if len(cube_map_images) == 6:
        layout_output_path = os.path.join(output_dir, "cubemap_layout.png")
        create_cube_map_layout(cube_map_images, layout_output_path)

    print(f"All cube map images saved to: {output_dir}")

    # Render some additional perspective views for comparison
    print("\nRendering additional perspective views (offscreen)...")

    camera_positions = setup_camera_positions(overall_center, overall_size)

    for i, camera_config in enumerate(camera_positions[:4]):  # Just render first 4 views
        print(f"Rendering view {i+1}/{4}: {camera_config['name']}")
        try:
            image = render_view(meshes, camera_config, width=1024, height=768)
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            output_path = os.path.join(output_dir, f"view_{camera_config['name']}.png")
            image_pil.save(output_path)
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  Error rendering {camera_config['name']}: {str(e)}")

    # Optional: Load and display the original panorama info
    panorama_path = os.path.join(mesh_dir, "panorama.png")

    if os.path.exists(panorama_path):
        print("Original panorama found for comparison")
        panorama = Image.open(panorama_path)
        print(f"Original panorama size: {panorama.size}")
    else:
        print("Original panorama not found")

    print("\n🎉 Rendering complete! Check the output directory for all rendered views.")
    print(f"Output directory: {output_dir}")


# ---------- Offscreen rendering helpers ----------

def _make_offscreen_renderer(width, height, bg_color):
    try:
        r = o3d.visualization.rendering.OffscreenRenderer(width, height)
        print(f"    Created offscreen renderer: {width}x{height}")
        
        # BG color RGBA - use darker background to see geometry better
        r.scene.set_background(0.1, 0.1, 0.1, 1.0)  # Dark gray background
        
        # Add lighting
        try:
            # Add directional light
            r.scene.scene.enable_sun_light(True)
            r.scene.scene.set_sun_light_direction([0.577, -0.577, -0.577])
            print("    Added sun light")
            
            # Add ambient lighting
            r.scene.scene.set_indirect_light_intensity(30000)
            print("    Set ambient lighting")
            
            # Add multiple point lights for better illumination
            r.scene.scene.add_point_light("light1", [1.0, 1.0, 1.0], [10, 10, 10], 50000, 1000.0)
            r.scene.scene.add_point_light("light2", [1.0, 1.0, 1.0], [-10, -10, 10], 50000, 1000.0)
            r.scene.scene.add_point_light("light3", [1.0, 1.0, 1.0], [0, 0, -10], 50000, 1000.0)
            print("    Added point lights")
            
        except Exception as e:
            print(f"    Warning: Lighting setup failed: {e}")
        
        return r
        
    except Exception as e:
        print(f"    Failed to create offscreen renderer: {e}")
        raise


def _add_meshes_to_scene(renderer, meshes):
    print(f"    Adding {len(meshes)} meshes to scene...")
    
    for i, m in enumerate(meshes):
        # Create a copy of the mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = m.vertices
        mesh.triangles = m.triangles
        
        print(f"      Mesh {i}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Copy vertex colors if they exist
        if m.has_vertex_colors():
            mesh.vertex_colors = m.vertex_colors
        else:
            # Apply default colors per layer
            if i == 0:
                mesh.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue
            else:
                mesh.paint_uniform_color([0.9, 0.7, 0.7])  # Light red
        
        # Compute normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Create material with bright colors for testing
        mat = o3d.visualization.rendering.MaterialRecord()
        
        if mesh.has_vertex_colors():
            # Use vertex colors but make them brighter
            mat.shader = "defaultLit"  # Try defaultLit instead of transparency
            mat.base_color = [1.0, 1.0, 1.0, 1.0]  # White to show vertex colors
            print(f"      Using vertex colors for mesh {i}")
        else:
            # Use bright base colors for visibility
            mat.shader = "defaultLit"
            if i == 0:
                mat.base_color = [1.0, 0.5, 0.5, 1.0]  # Bright red
            else:
                mat.base_color = [0.5, 1.0, 0.5, 1.0]  # Bright green
            print(f"      Using base color {mat.base_color} for mesh {i}")
        
        # Bright material properties
        mat.metallic = 0.0
        mat.roughness = 0.3  # Less rough = more reflective
        mat.reflectance = 0.5  # More reflective
        
        name = f"mesh_{i}"
        renderer.scene.add_geometry(name, mesh, mat)
        print(f"      Added mesh {i} to scene as '{name}'")


def setup_camera_positions(bbox_center, bbox_size):
    """
    Generate camera positions around the scene for rendering different views
    """
    max_extent = float(np.max(bbox_size))
    camera_distance = max(0.1, max_extent * 0.8)  # Much closer to the mesh
    print(f"    Camera distance: {camera_distance} (max_extent: {max_extent})")

    camera_configs = [
        # Front views
        {'name': 'front', 'theta': 0, 'phi': 0},
        {'name': 'front_high', 'theta': 0, 'phi': 30},
        {'name': 'front_low', 'theta': 0, 'phi': -30},

        # Side views
        {'name': 'right', 'theta': 90, 'phi': 0},
        {'name': 'left', 'theta': -90, 'phi': 0},
        {'name': 'back', 'theta': 180, 'phi': 0},

        # Top-down and angled views
        {'name': 'top', 'theta': 0, 'phi': 80},
        {'name': 'angled_1', 'theta': 45, 'phi': 30},
        {'name': 'angled_2', 'theta': 135, 'phi': 30},
        {'name': 'angled_3', 'theta': 225, 'phi': 30},
        {'name': 'angled_4', 'theta': 315, 'phi': 30},
    ]

    camera_positions = []
    for config in camera_configs:
        theta_rad = np.radians(config['theta'])
        phi_rad = np.radians(config['phi'])

        # Convert spherical to cartesian coordinates
        x = camera_distance * np.cos(phi_rad) * np.cos(theta_rad)
        y = camera_distance * np.cos(phi_rad) * np.sin(theta_rad)
        z = camera_distance * np.sin(phi_rad)

        # Position relative to scene center
        position = bbox_center + np.array([x, y, z])

        camera_positions.append({
            'name': config['name'],
            'position': position,
            'target': bbox_center,
            'up': [0, 0, 1]  # Z-up coordinate system
        })

    return camera_positions


def render_view(meshes, camera_config, width=1024, height=768, background_color=[1, 1, 1]):
    """
    Render a view of the meshes from a specific camera position (headless)
    """
    try:
        r = _make_offscreen_renderer(width, height, background_color)
        _add_meshes_to_scene(r, meshes)

        eye = np.array(camera_config['position'], dtype=float)
        target = np.array(camera_config['target'], dtype=float)
        up = np.array(camera_config['up'], dtype=float)

        # 60° vertical FOV default
        fov_deg = 60.0
        
        # Calculate proper near/far based on distance to mesh
        dist_to_target = np.linalg.norm(eye - target)
        znear = max(0.01, dist_to_target * 0.01)  # 1% of distance
        zfar = max(100.0, dist_to_target * 10.0)   # 10x distance
        
        print(f"    Camera setup: eye={eye}, target={target}, dist={dist_to_target:.2f}, near={znear:.2f}, far={zfar:.2f}")

        r.setup_camera(fov_deg, eye, target, up)
        r.scene.camera.set_near(znear)
        r.scene.camera.set_far(zfar)

        img = r.render_to_image()
        result = np.asarray(img)[:, :, :3] / 255.0
        
        # Check if image is all black (potential rendering failure)
        if np.max(result) < 0.01:
            print("    Warning: Rendered image appears to be black, trying fallback...")
            raise Exception("Black image detected")
            
        return result
        
    except Exception as e:
        print(f"    Offscreen rendering failed: {e}")
        print("    Falling back to visualizer method...")
        return render_view_fallback(meshes, camera_config, width, height, background_color)


def render_view_fallback(meshes, camera_config, width=1024, height=768, background_color=[1, 1, 1]):
    """
    Fallback rendering using the old visualizer method (may have display issues but sometimes works)
    """
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        
        # Try to create window
        success = vis.create_window(width=width, height=height, visible=False)
        if not success:
            # Try again with default settings
            vis.create_window(visible=False)
        
        # Add meshes
        for i, mesh in enumerate(meshes):
            mesh_copy = o3d.geometry.TriangleMesh()
            mesh_copy.vertices = mesh.vertices
            mesh_copy.triangles = mesh.triangles
            
            if mesh.has_vertex_colors():
                mesh_copy.vertex_colors = mesh.vertex_colors
            else:
                if i == 0:
                    mesh_copy.paint_uniform_color([0.7, 0.7, 0.9])
                else:
                    mesh_copy.paint_uniform_color([0.9, 0.7, 0.7])
            
            if not mesh_copy.has_vertex_normals():
                mesh_copy.compute_vertex_normals()
            
            vis.add_geometry(mesh_copy)
        
        # Set camera
        ctr = vis.get_view_control()
        ctr.set_lookat(camera_config['target'])
        ctr.set_front(np.array(camera_config['target']) - np.array(camera_config['position']))
        ctr.set_up(camera_config['up'])
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Capture
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        return np.asarray(image)
        
    except Exception as e:
        print(f"    Fallback rendering also failed: {e}")
        # Return a colored image so we know it failed
        fallback_img = np.ones((height, width, 3)) * 0.3  # Dark gray
        return fallback_img


def render_cube_map(meshes, center_position, resolution=1024, background_color=[1, 1, 1]):
    """
    Render a cube map from the center position looking outward in 6 directions
    Returns list of 6 HxWx3 float images in [0,1]
    """
    cube_faces = [
        {'name': 'positive_x', 'target_offset': [1, 0, 0], 'up': [0, 0, 1]},
        {'name': 'negative_x', 'target_offset': [-1, 0, 0], 'up': [0, 0, 1]},
        {'name': 'positive_y', 'target_offset': [0, 1, 0], 'up': [0, 0, 1]},
        {'name': 'negative_y', 'target_offset': [0, -1, 0], 'up': [0, 0, 1]},
        {'name': 'positive_z', 'target_offset': [0, 0, 1], 'up': [0, 1, 0]},
        {'name': 'negative_z', 'target_offset': [0, 0, -1], 'up': [0, 1, 0]}
    ]

    cube_images = []

    for face in cube_faces:
        print(f"  Rendering cube face: {face['name']}")
        target_position = np.array(center_position) + np.array(face['target_offset'])

        camera_config = {
            'position': center_position,
            'target': target_position,
            'up': face['up']
        }

        try:
            image = render_cube_face(meshes, camera_config, resolution, background_color)
            cube_images.append(image)
        except Exception as e:
            print(f"    Error rendering {face['name']}: {str(e)}")
            fallback_image = np.zeros((resolution, resolution, 3), dtype=float)
            cube_images.append(fallback_image)

    return cube_images


def render_cube_face(meshes, camera_config, resolution=1024, background_color=[1, 1, 1]):
    """
    Render a single cube map face with 90-degree FOV (headless)
    """
    r = _make_offscreen_renderer(resolution, resolution, background_color)
    _add_meshes_to_scene(r, meshes)

    eye = np.array(camera_config['position'], dtype=float)
    target = np.array(camera_config['target'], dtype=float)
    up = np.array(camera_config['up'], dtype=float)

    # 90° FOV for cubemaps
    fov_deg = 90.0
    
    # Calculate proper near/far for cube map
    dist_to_target = np.linalg.norm(eye - target)
    znear = max(0.01, dist_to_target * 0.01)
    zfar = max(100.0, dist_to_target * 10.0)

    r.setup_camera(fov_deg, eye, target, up)
    r.scene.camera.set_near(znear)
    r.scene.camera.set_far(zfar)

    img = r.render_to_image()
    return np.asarray(img)[:, :, :3] / 255.0


def render_custom_view(meshes, eye_position, target_position, up_vector=[0, 0, 1],
                       width=1024, height=768, background_color=[1, 1, 1]):
    """
    Render a custom view with manually specified camera parameters (headless)
    """
    camera_config = {
        'position': eye_position,
        'target': target_position,
        'up': up_vector
    }
    return render_view(meshes, camera_config, width, height, background_color)


def create_cube_map_layout(cube_images, output_path):
    """
    Create a cube map layout image showing all 6 faces in a cross pattern

    Standard cube map layout:
        [ ]  [+Y] [ ]  [ ]
        [-X] [+Z] [+X] [-Z]
        [ ]  [-Y] [ ]  [ ]
    """
    if len(cube_images) != 6:
        print(f"Warning: Expected 6 cube faces, got {len(cube_images)}")
        return

    # Get resolution from first image
    resolution = cube_images[0].shape[0]

    # Create layout: 4x3 grid
    layout_width = resolution * 4
    layout_height = resolution * 3
    layout_image = np.ones((layout_height, layout_width, 3))  # White background

    # Face indices: [+X, -X, +Y, -Y, +Z, -Z]
    pos_x, neg_x, pos_y, neg_y, pos_z, neg_z = cube_images

    # Place faces in cross pattern
    # Top row: [empty, +Y, empty, empty]
    layout_image[0:resolution, resolution:2*resolution] = pos_y

    # Middle row: [-X, +Z, +X, -Z]
    layout_image[resolution:2*resolution, 0:resolution] = neg_x
    layout_image[resolution:2*resolution, resolution:2*resolution] = pos_z
    layout_image[resolution:2*resolution, 2*resolution:3*resolution] = pos_x
    layout_image[resolution:2*resolution, 3*resolution:4*resolution] = neg_z

    # Bottom row: [empty, -Y, empty, empty]
    layout_image[2*resolution:3*resolution, resolution:2*resolution] = neg_y

    # Save the layout
    layout_pil = Image.fromarray((layout_image * 255).astype(np.uint8))
    layout_pil.save(output_path)
    print(f"Cube map layout saved to: {output_path}")


def display_rendered_views_summary(output_dir):
    """
    Display a summary of rendered views (for command-line usage)
    """
    if not os.path.exists(output_dir):
        print("Output directory not found")
        return

    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

    print(f"\nRendered {len(png_files)} views:")
    for filename in sorted(png_files):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()

    # Optional: Show summary of rendered files
    output_dir = "/home/runyi_yang/Gen3D/HunyuanWorld-enhance/examples/indoor_kitchen/with_mid_prompt/rendered_views"
    display_rendered_views_summary(output_dir)
