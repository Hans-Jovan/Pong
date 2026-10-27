
import cv2
import numpy as np
import random
import math

# =====================================
# 1. PENGATURAN GAME
# =====================================
W, H = 800, 600

TABLE_LEFT = 35
TABLE_RIGHT = W - 35
TABLE_TOP = 35
TABLE_BOTTOM = H - 35
TABLE_MID = H // 2

PADDLE_R = 25
PUCK_R = 13
WIN_SCORE = 5

# =====================================
# 2. PENGATURAN KAMERA
# =====================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak terdeteksi!")
    exit()

kernel = np.ones((5, 5), np.uint8)

# Rentang warna kulit YCrCb
lower = np.array([0, 140, 85], dtype=np.uint8)
upper = np.array([255, 175, 135], dtype=np.uint8)

# Fokus ke bagian tengah-bawah kamera
ROI_X1, ROI_X2 = 0.01, 0.99
ROI_Y1, ROI_Y2 = 0.01, 0.99

MIN_AREA = 2500
MIN_PALM_RADIUS = 12

previous_center = None

# =====================================
# 3. POSISI AWAL GAME
# =====================================
player_x = W // 2
player_y = H - 115

smooth_x = player_x
smooth_y = player_y

ai_x = W // 2
ai_y = 115

puck_x = W // 2
puck_y = H // 2

puck_vx = random.choice([-3, 3])
puck_vy = 5

player_score = 0
ai_score = 0

game_over = False
winner = ""

# =====================================
# 4. FUNGSI BANTU
# =====================================
def limit(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def reset_puck(direction):
    global puck_x, puck_y, puck_vx, puck_vy

    puck_x = W // 2
    puck_y = H // 2
    puck_vx = random.choice([-3, -2, 2, 3])
    puck_vy = 5 * direction


# =====================================
# 5. LOOP UTAMA
# =====================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca webcam!")
        break

    frame = cv2.flip(frame, 1)
    output = frame.copy()

    fh, fw = frame.shape[:2]

    # ---------------------------------
    # A. AMBIL AREA DETEKSI TANGAN
    # ---------------------------------
    x1 = int(fw * ROI_X1)
    x2 = int(fw * ROI_X2)
    y1 = int(fh * ROI_Y1)
    y2 = int(fh * ROI_Y2)

    roi = frame[y1:y2, x1:x2]

    # ---------------------------------
    # B. SEGMENTASI WARNA KULIT
    # ---------------------------------
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    mask = cv2.inRange(ycrcb, lower, upper)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # ---------------------------------
    # C. CARI KONTUR TANGAN
    # ---------------------------------
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    hand_found = False
    palm_x = None
    palm_y = None
    best_score = float("inf")

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        # Buat mask khusus untuk satu kontur
        hand_mask = np.zeros_like(mask)
        cv2.drawContours(
            hand_mask, [contour], -1, 255, thickness=-1
        )

        # Pusat lingkaran terbesar di dalam kontur
        distance = cv2.distanceTransform(
            hand_mask, cv2.DIST_L2, 5
        )

        _, radius, _, center = cv2.minMaxLoc(distance)
        cx, cy = center

        if radius < MIN_PALM_RADIUS:
            continue

        # Konversi koordinat ROI ke koordinat kamera
        candidate_x = x1 + cx
        candidate_y = y1 + cy

        # Bentuk kontur: singkirkan bentuk terlalu memanjang
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / max(bh, 1)

        if aspect_ratio < 0.35 or aspect_ratio > 2.5:
            continue

        # Pilih kontur terdekat dengan posisi tangan sebelumnya
        if previous_center is not None:
            distance_previous = math.hypot(
                candidate_x - previous_center[0],
                candidate_y - previous_center[1]
            )
            score = distance_previous - radius * 0.5
        else:
            # Saat pertama kali, utamakan kontur besar
            score = -area - radius * 100

        if score < best_score:
            best_score = score
            palm_x = candidate_x
            palm_y = candidate_y
            best_contour = contour
            best_radius = radius
            best_center = center
            hand_found = True

    # ---------------------------------
    # D. TRACKING TELAPAK
    # ---------------------------------
    if hand_found:
        previous_center = (palm_x, palm_y)

        # Koordinat tangan ke koordinat meja game
        target_x = int(palm_x * W / fw)
        target_y = int(palm_y * H / fh)

        target_x = max(
            TABLE_LEFT + PADDLE_R,
            min(TABLE_RIGHT - PADDLE_R, target_x)
        )
        

        target_y = max(
            TABLE_MID + PADDLE_R,
            min(TABLE_BOTTOM - PADDLE_R, target_y)
        )

        # Responsif tetapi tetap halus
        smooth_x = int(0.25 * smooth_x + 0.75 * target_x)
        smooth_y = int(0.25 * smooth_y + 0.75 * target_y)

        player_x = smooth_x
        player_y = smooth_y

        # Gambar kontur tangan dan perkiraan telapak
        cv2.drawContours(
            output,
            [best_contour + np.array([[[x1, y1]]])],
            -1, (0, 255, 0), 2
        )

        cv2.circle(
            output, (palm_x, palm_y),
            int(best_radius), (0, 255, 255), 2
        )

        cv2.circle(
            output, (palm_x, palm_y),
            6, (0, 0, 255), -1
        )

        cv2.putText(
            output, "PALM DETECTED",
            (palm_x + 10, palm_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    else:
        cv2.putText(
            output, "Show your palm in the lower area",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2
        )

    # Kotak ROI
    cv2.rectangle(
        output, (x1, y1), (x2, y2), (255, 255, 0), 2
    )

    # =====================================
    # 6. GAMBAR MEJA
    # =====================================
    game = np.zeros((H, W, 3), dtype=np.uint8)
    game[:] = (25, 30, 35)

    cv2.rectangle(
        game,
        (TABLE_LEFT, TABLE_TOP),
        (TABLE_RIGHT, TABLE_BOTTOM),
        (220, 220, 220), 3
    )

    cv2.line(
        game,
        (TABLE_LEFT, TABLE_MID),
        (TABLE_RIGHT, TABLE_MID),
        (100, 100, 100), 2
    )

    cv2.circle(
        game, (W // 2, H // 2), 65,
        (100, 100, 100), 2
    )

    goal_left = W // 2 - 90
    goal_right = W // 2 + 90

    cv2.line(
        game, (goal_left, TABLE_TOP),
        (goal_right, TABLE_TOP), (0, 255, 255), 6
    )

    cv2.line(
        game, (goal_left, TABLE_BOTTOM),
        (goal_right, TABLE_BOTTOM), (0, 255, 255), 6
    )

    # =====================================
    # 7. LOGIKA GAME
    # =====================================
    if not game_over:

        # AI mengikuti puck
        ai_x += int(limit((puck_x - ai_x) * 0.08, -5, 5))
        ai_x = limit(
            ai_x,
            TABLE_LEFT + PADDLE_R + 5,
            TABLE_RIGHT - PADDLE_R - 5
        )

        # Gerak puck
        puck_x += puck_vx
        puck_y += puck_vy

        # Dinding kiri dan kanan
        if puck_x - PUCK_R <= TABLE_LEFT:
            puck_x = TABLE_LEFT + PUCK_R
            puck_vx = abs(puck_vx)

        if puck_x + PUCK_R >= TABLE_RIGHT:
            puck_x = TABLE_RIGHT - PUCK_R
            puck_vx = -abs(puck_vx)

        # Gawang atas: pemain mencetak poin
        if puck_y - PUCK_R <= TABLE_TOP:
            if goal_left <= puck_x <= goal_right:
                player_score += 1
                reset_puck(1)
            else:
                puck_y = TABLE_TOP + PUCK_R
                puck_vy = abs(puck_vy)

        # Gawang bawah: AI mencetak poin
        if puck_y + PUCK_R >= TABLE_BOTTOM:
            if goal_left <= puck_x <= goal_right:
                ai_score += 1
                reset_puck(-1)
            else:
                puck_y = TABLE_BOTTOM - PUCK_R
                puck_vy = -abs(puck_vy)

        # Tabrakan dengan paddle - mencegah puck tersangkut
        paddles = [
            (ai_x, ai_y, 1),          # AI memantulkan puck ke bawah
            (player_x, player_y, -1)  # Player memantulkan puck ke atas
        ]

        for px, py, direction in paddles:
            dx = puck_x - px
            dy = puck_y - py
            distance = math.hypot(dx, dy)
            min_distance = PADDLE_R + PUCK_R

            # Hanya pantulkan jika puck sedang bergerak menuju paddle
            moving_toward = (
                (direction == 1 and puck_vy < 0) or
                (direction == -1 and puck_vy > 0)
            )

            if 0 < distance < min_distance and moving_toward:
                nx = dx / distance
                ny = dy / distance

                # Pisahkan puck dari paddle agar tidak bertabrakan terus-menerus
                puck_x = px + nx * (min_distance + 2)
                puck_y = py + ny * (min_distance + 2)

                # Pantulan dengan kecepatan yang konsisten
                puck_vx = nx * 6
                puck_vy = 6 * direction

                # Pastikan puck tetap berada di dalam batas meja
                puck_x = limit(
                    puck_x,
                    TABLE_LEFT + PUCK_R,
                    TABLE_RIGHT - PUCK_R
                )
                puck_y = limit(
                    puck_y,
                    TABLE_TOP + PUCK_R,
                    TABLE_BOTTOM - PUCK_R
                )

        # Cek pemenang
        if player_score >= WIN_SCORE:
            game_over = True
            winner = "YOU WIN!"

        elif ai_score >= WIN_SCORE:
            game_over = True
            winner = "AI WINS!"

    # =====================================
    # 8. GAMBAR PADDLE, PUCK, DAN SKOR
    # =====================================
    cv2.circle(
        game, (int(ai_x), int(ai_y)),
        PADDLE_R, (0, 100, 255), -1
    )

    cv2.circle(
        game, (int(player_x), int(player_y)),
        PADDLE_R, (255, 100, 0), -1
    )

    cv2.circle(
        game, (int(puck_x), int(puck_y)),
        PUCK_R, (255, 255, 255), -1
    )

    cv2.putText(
        game, f"AI: {ai_score}",
        (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 150, 255), 2
    )

    cv2.putText(
        game, f"YOU: {player_score}",
        (50, H - 55), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 150, 50), 2
    )

    if game_over:
        cv2.putText(
            game, winner, (W // 2 - 120, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 255, 0), 3
        )

        cv2.putText(
            game, "Press R to restart",
            (W // 2 - 135, H // 2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

    # =====================================
    # 9. TAMPILKAN WINDOW
    # =====================================
    cv2.imshow(
        "Camera - Palm Tracking",
        cv2.resize(output, (480, 360))
    )

    cv2.imshow(
        "AIR HOCKEY",
        cv2.resize(game, (640, 480))
    )

    cv2.imshow("Skin Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord("r") and game_over:
        player_score = 0
        ai_score = 0
        game_over = False
        winner = ""

        player_x = W // 2
        player_y = H - 115
        smooth_x = player_x
        smooth_y = player_y
        previous_center = None

        reset_puck(1)

cap.release()
cv2.destroyAllWindows()
