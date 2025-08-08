import cv2
import numpy as np
import cv2.aruco as aruco
import time


def load_obj_file(obj_path):
    """
    Loads a simple OBJ file.
    It extracts vertices and faces.
    """
    vertices = []
    faces = []
    try:
        with open(obj_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                        faces.append(face_indices)
        if vertices:
            vertices = np.array(vertices, dtype=np.float32)
            # Normalize and scale the model
            vertices -= np.mean(vertices, axis=0)
            max_dim = np.max(np.abs(vertices))
            if max_dim > 0:
                vertices *= (1.0 / max_dim)  # Scale to fit in a 1x1x1 cube
            return {'vertices': vertices, 'faces': faces}
    except FileNotFoundError:
        print(f"Error: OBJ file not found at {obj_path}")
        return None
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return None


def render_obj_model(frame, obj_data, rvec, tvec, camera_matrix, dist_coeffs, rotation_angle):
    """
    Renders the OBJ model on top of the marker with a slow clockwise rotation.
    """
    if obj_data is None or rvec is None or tvec is None:
        return

    # 1. Get the rotation matrix from the ArUco marker's rotation vector
    rmat_marker, _ = cv2.Rodrigues(rvec)

    # 2. Create a rotation matrix for the slow clockwise spin around the Y-axis
    angle_rad = np.deg2rad(rotation_angle)
    rmat_spin = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ], dtype=np.float32)

    # 3. Combine the rotations. This new order ensures a stable "turntable" spin.
    # The model is first oriented by the marker, then it spins around the global Y-axis.
    combined_rmat = rmat_spin @ rmat_marker

    # Convert the combined rotation matrix back to a rotation vector for projectPoints
    combined_rvec, _ = cv2.Rodrigues(combined_rmat)

    # --- Adjust Position to be on TOP of the marker ---
    # Define how high above the marker the object should be (in meters)
    height_offset = 0.025
    # Create the offset vector in the marker's coordinate system (along its Y axis)
    t_offset_marker_coords = np.array([[0], [height_offset], [0]], dtype=np.float32)
    # Transform the offset into the camera's coordinate system using the marker's rotation
    t_offset_camera_coords = rmat_marker @ t_offset_marker_coords
    # Add the calculated offset to the marker's translation vector
    final_tvec = tvec + t_offset_camera_coords

    # Scale the model vertices based on the marker size
    marker_length = 0.25
    scaled_vertices = obj_data['vertices'] * marker_length

    # Project the rotated, scaled, and lifted 3D points to the 2D image plane
    vertices_2d, _ = cv2.projectPoints(scaled_vertices, combined_rvec, final_tvec, camera_matrix, dist_coeffs)

    if vertices_2d is None:
        return

    vertices_2d = np.squeeze(vertices_2d).astype(int)

    # Draw the faces with a simple color
    for face in obj_data['faces']:
        points = np.array([vertices_2d[i] for i in face if i < len(vertices_2d)], dtype=np.int32)
        if len(points) > 2:
            # Draw filled polygon with a light grey color
            cv2.fillPoly(frame, [points], (200, 200, 200))
            # Draw the outline in black
            cv2.polylines(frame, [points], True, (0, 0, 0), 1)


def main():
    """
    Main function to run ArUco-based 3D model rendering.
    """
    # --- Camera and ArUco Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ArUco dictionary and detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # --- Camera Calibration Parameters ---
    # NOTE: These are example values. For better results, calibrate your camera.
    camera_matrix = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # --- Load 3D Model ---
    obj_model = load_obj_file('Katana/Katana.obj')
    if obj_model is None:
        print("Fallback: Creating a simple cube model.")
        cube_verts = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        cube_faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4],
            [1, 2, 6, 5], [0, 1, 5, 4], [3, 2, 6, 7]
        ]
        obj_model = {'vertices': cube_verts, 'faces': cube_faces}

    print("AR tracking started. Show an ArUco marker to the camera. Press 'q' to quit.")

    # Define the 3D coordinates of the ArUco marker corners
    marker_length = 0.05  # IMPORTANT: This should be the actual size of your printed marker in meters
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    rotation_angle = 30
    rotation_speed = 30  # degrees per second

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        # Increment rotation angle for slow clockwise spin
        rotation_angle += rotation_speed * delta_time
        if rotation_angle > 360:
            rotation_angle -= 360

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, _ in enumerate(ids):
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
                if success:
                    # Render the 3D model with the current rotation angle
                    render_obj_model(frame, obj_model, rvec, tvec, camera_matrix, dist_coeffs, rotation_angle)

        cv2.imshow('Simple ArUco AR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
