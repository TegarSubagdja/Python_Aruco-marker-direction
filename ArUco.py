import cv2
import numpy as np

def detect_marker_3d():
    """
    Mendeteksi posisi 3D marker ArUco menggunakan webcam dengan tampilan di layar
    """
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)
    
    # Set resolusi webcam (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Parameter kamera
    focal_length = 1000.0
    center = (640, 360)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype=np.float32)
    
    # Asumsi tidak ada distorsi lensa
    dist_coeffs = np.zeros((4, 1))
    
    # Ukuran marker dalam meter
    marker_size = 0.1  # 10 cm
    
    # Inisialisasi ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    try:
        while True:
            # Baca frame dari webcam
            ret, frame = cap.read()
            if not ret:
                print("Error membaca frame dari webcam")
                break
            
            # Buat overlay untuk informasi
            overlay = frame.copy()
            
            # Konversi ke grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi marker
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            # Jika marker terdeteksi
            if ids is not None:
                # Gambar marker yang terdeteksi
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Estimasi pose untuk setiap marker
                for i in range(len(ids)):
                    # Hitung pusat marker
                    corner = corners[i][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))
                    
                    # Estimasi pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], marker_size, camera_matrix, dist_coeffs
                    )
                    
                    # Gambar axis koordinat 3D
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)
                    
                    # Dapatkan posisi dalam meter
                    x, y, z = tvecs[0][0]
                    
                    # Konversi rotasi vector ke sudut Euler (dalam derajat)
                    rmat, _ = cv2.Rodrigues(rvecs[0])
                    euler_angles = cv2.RQDecomp3x3(rmat)[0]
                    
                    # Informasi yang akan ditampilkan
                    info_lines = [
                        f"Marker ID: {ids[i][0]}",
                        f"Position (m):",
                        f"X: {x:.3f}",
                        f"Y: {y:.3f}",
                        f"Z: {z:.3f}",
                        f"Rotation (deg):",
                        f"Roll: {euler_angles[0]:.1f}",
                        f"Pitch: {euler_angles[1]:.1f}",
                        f"Yaw: {euler_angles[2]:.1f}"
                    ]
                    
                    # Tampilkan informasi di samping marker
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    line_spacing = 25
                    bg_color = (0, 0, 0)
                    text_color = (0, 255, 0)
                    
                    # Posisi teks (di samping marker)
                    text_x = center_x + 100
                    text_y = center_y - len(info_lines) * line_spacing // 2
                    
                    # Gambar background semi-transparan untuk teks
                    for j, line in enumerate(info_lines):
                        y = text_y + j * line_spacing
                        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                        cv2.rectangle(overlay, 
                                    (text_x - 5, y - 20),
                                    (text_x + text_size[0] + 5, y + 5),
                                    bg_color, -1)
                    
                    # Aplikasikan transparansi
                    alpha = 0.6
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Tulis teks
                    for j, line in enumerate(info_lines):
                        y = text_y + j * line_spacing
                        cv2.putText(frame, line,
                                  (text_x, y),
                                  font, font_scale, text_color, font_thickness)
            
            # Tampilkan frame
            cv2.imshow('ArUco 3D Pose Detection', frame)
            
            # Keluar dengan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Bersihkan
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_marker_3d()