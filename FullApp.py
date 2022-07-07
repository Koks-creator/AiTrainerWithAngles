import cv2
import mediapipe as mp
import math
import numpy as np
from time import sleep, time
import json


def get_rowing_data(points: list) -> list:
    global counter
    global stage

    arm_angle, arm_perc, arm_bar = 0, 0, 600
    if lm_list[24][3] > 0.7 or lm_list[23][3] > 0.7:
        # czy w pochylony
        # ktora strona jest blizej
        if lm_list[12][2] > lm_list[11][2]:
            arm_points = [lm_list[11], lm_list[13]]
            back_angle = angle2pt(lm_list[11], lm_list[23])

            arm_angle = angle2pt(arm_points[0], arm_points[1])
            arm_angle = 360 - arm_angle if arm_angle > 180 else arm_angle  # 170, 50

            ankle_shoulder_ydiff = lm_list[25][1] - lm_list[11][1]
        else:
            arm_points = [lm_list[14], lm_list[12]]
            back_angle = angle2pt(lm_list[24], lm_list[12]) * -1

            arm_angle = angle2pt(arm_points[0], arm_points[1]) * -1
            arm_angle = 360 - arm_angle if arm_angle > 180 else arm_angle  # 170, 50

            ankle_shoulder_ydiff = lm_list[26][1] - lm_list[12][1]

        cv2.putText(img, f"Back angle: {int(back_angle)} degr.", (10, 280),  cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)

        if 0 < back_angle < 50:

            # odleglosc Y miedzy barkiem a kolanem zeby chlop se pompek nie mogl robic, bo z reguly jak pompki robisz
            # to Y barku i kolan sa podobne jako tako
            if ankle_shoulder_ydiff > 180:
                arm_perc = np.interp(arm_angle, (0, 80), (100, 0))
                arm_bar = np.interp(arm_angle, (0, 80), (300, 600))

                if arm_perc <= 10:
                    stage = "Down"
                # czy lokiec wyzej od barku
                if arm_perc >= 95 and stage == "Down" and arm_angle < -2:
                    stage = "Up"
                    if counter < reps:
                        counter += 1

                for arm_point in arm_points:
                    cv2.circle(img, arm_point[:2], 8, points_color, -1)
                    cv2.circle(img, arm_point[:2], 12, points_color, 2)
                # cv2.putText(img, f"{ankle_shoulder_ydiff}", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255),4)
            else:
                cv2.putText(img, f"Stop trying to make pushups", (200, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
        else:
            cv2.putText(img, f"Bend over to back", (320, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            cv2.putText(img, f"angle less than 50", (320, 480), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
    else:
        cv2.putText(img, f"Whole body must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    return [arm_perc, arm_bar, arm_angle]


def get_pullup_data(points: list) -> list:
    global counter
    global stage
    global counter2
    global stage2

    right_arm_angle, right_arm_perc, right_arm_bar = 0, 0, 600

    points_right = [lm_list[12], lm_list[14], lm_list[16]]
    points_left = [lm_list[11], lm_list[13], lm_list[15]]

    right_arm_angle = angle3pt(points_right[0], points_right[1], points_right[2])
    right_arm_angle = 360 - right_arm_angle if right_arm_angle > 180 else right_arm_angle

    if lm_list[24][3] > 0.7:
        for i in range(len(points_left)):
            cv2.circle(img, points_right[i][:2], 8, points_color, -1)
            cv2.circle(img, points_right[i][:2], 12, points_color, 2)
            cv2.circle(img, points_left[i][:2], 8, points_color, -1)
            cv2.circle(img, points_left[i][:2], 12, points_color, 2)

        if lm_list[20][1] < lm_list[12][1]:
            right_arm_perc = np.interp(right_arm_angle, (60, 140), (100, 0))
            right_arm_bar = np.interp(right_arm_angle, (60, 140), (300, 600))
            if right_arm_perc <= 5:
                stage = "Down"
            if right_arm_perc >= 95 and stage == "Down" and lm_list[10][1] < lm_list[20][1]:
                stage = "Up"
                if counter < reps:
                    counter += 1
    else:
        cv2.putText(img, f"Hips must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    return [right_arm_perc, right_arm_bar, right_arm_angle]


def get_reverse_pushup_data(arm_points: list) -> list:
    global counter
    global stage

    arm_angle, bar, perc = 0, 600, 0

    allow = False
    if lm_list[26][3] > 0.7 or lm_list[25][3] > 0.7:
        if lm_list[11][2] > lm_list[12][2]:
            diff = lm_list[24][0] - lm_list[12][0]
            # cv2.putText(img, f"{diff}", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

            if diff > 60:
                allow = True
                arm_points = [lm_list[12], lm_list[14], lm_list[16]]
                arm_angle = angle3pt(arm_points[0], arm_points[1], arm_points[2])
                arm_angle = 360 - arm_angle if arm_angle > 180 else arm_angle

        else:
            if lm_list[13][0] > lm_list[11][0]:
                diff = lm_list[11][0] - lm_list[23][0]

                if diff > 60:
                    allow = True
                    arm_points = [lm_list[11], lm_list[13], lm_list[15]]
                    arm_angle = angle3pt(arm_points[0], arm_points[1], arm_points[2])
                    arm_angle = 360 - arm_angle if arm_angle > 180 else arm_angle
            else:
                arm_angle, bar, perc = 0, 600, 0

        for arm_point in arm_points:
            cv2.circle(img, arm_point[:2], 8, points_color, -1)
            cv2.circle(img, arm_point[:2], 12, points_color, 2)

        perc = np.interp(arm_angle, (90, 155), (100, 0))
        bar = np.interp(arm_angle, (90, 155), (300, 600))

        if perc <= 5:
            stage = "Up"
        if perc >= 95 and stage == "Up" and allow:
            stage = "Down"
            counter += 1
    else:
        cv2.putText(img, f"Ankles must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    if allow is False:
        arm_angle, bar, perc = 0, 600, 0

    return [perc, bar, arm_angle]


def get_dumbbell_lateral_raise_data() -> list:
    global counter
    global stage
    global counter2
    global stage2

    right_angle, right_arm_perc, right_arm_bar = 0, 0, 600
    left_angle, left_arm_perc, left_arm_bar = 0, 0, 600
    if lm_list[24][3] > 0.4:
        z1 = round(lm_list[12][2], 3)
        z2 = round(lm_list[11][2], 3)
        z_diff = round(z1 - z2, 3)
        z_diff = z_diff * -1 if z_diff < 0 else z_diff

        if z_diff < 0.3:
            left_arm_points = [lm_list[11], lm_list[13], lm_list[15]]
            arm_angle_left = angle3pt(left_arm_points[0], left_arm_points[1], left_arm_points[2])
            arm_angle_left = 360 - arm_angle_left if arm_angle_left > 180 else arm_angle_left

            for point in left_arm_points:
                cv2.circle(img, point[:2], 8, (255, 0, 125), -1)
                cv2.circle(img, point[:2], 12, (255, 0, 125), 2)

            if 60 < arm_angle_left < 190:
                left_angle = angle3pt(lm_list[23], lm_list[11], lm_list[13])
                left_angle = 360 - left_angle

                left_arm_perc = np.interp(left_angle, (30, 90), (0, 100))
                left_arm_bar = np.interp(left_angle, (30, 90), (600, 300))

                if left_angle < 120:
                    if left_arm_perc <= 5:
                        stage2 = "Down"
                    if left_arm_perc >= 95 and stage2 == "Down" and left_angle < 110:
                        stage2 = "Up"
                        if counter2 < reps:
                            counter2 += 1
                else:
                    cv2.putText(img, f"Your arms are too high", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

            right_arm_points = [lm_list[12], lm_list[14], lm_list[16]]
            arm_angle_right = angle3pt(right_arm_points[0], right_arm_points[1], right_arm_points[2])
            arm_angle_right = 360 - arm_angle_right if arm_angle_right > 180 else arm_angle_right

            for point in right_arm_points:
                cv2.circle(img, point[:2], 8, points_color, -1)
                cv2.circle(img, point[:2], 12, points_color, 2)

            if 60 < arm_angle_right < 190:
                right_angle = angle3pt(lm_list[24], lm_list[12], lm_list[14])

                right_arm_perc = np.interp(right_angle, (30, 100), (0, 100))
                right_arm_bar = np.interp(right_angle, (30, 100), (600, 300))

                if right_angle < 120:
                    if right_arm_perc <= 5:
                        stage = "Down"
                    if right_arm_perc >= 95 and stage == "Down" and right_angle < 110:
                        stage = "Up"
                        if counter < reps:
                            counter += 1
                else:
                    cv2.putText(img, f"Your arms are too high", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
        else:
            cv2.putText(img, f"You have to face forward", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
    else:
        cv2.putText(img, f"Hips must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
    return [right_angle, right_arm_perc, right_arm_bar, left_angle, left_arm_perc, left_arm_bar]


def get_overhead_dumbbell_data() -> list:
    global counter
    global stage
    global counter2
    global stage2

    right_arm_angle, right_arm_perc, right_arm_bar = 0, 0, 600
    left_arm_angle, left_arm_perc, left_arm_bar = 0, 0, 600
    if lm_list[24][3] > 0.4:
        z1 = round(lm_list[12][2], 3)
        z2 = round(lm_list[11][2], 3)
        z_diff = round(z1 - z2, 3)
        z_diff = z_diff * -1 if z_diff < 0 else z_diff

        if z_diff < 0.3:
            if lm_list[16][1] < lm_list[12][1] and lm_list[14][0] < lm_list[16][0]:
                right_arm_points = [lm_list[12], lm_list[14], lm_list[16]]
                for point in right_arm_points:
                    cv2.circle(img, point[:2], 8, points_color, -1)
                    cv2.circle(img, point[:2], 12, points_color, 2)

                right_arm_angle = angle3pt(lm_list[12], lm_list[14], lm_list[16])
                right_arm_angle = 360 - right_arm_angle

                right_arm_perc = np.interp(right_arm_angle, (65, 110), (0, 100))
                right_arm_bar = np.interp(right_arm_angle, (65, 100), (600, 300))

                if right_arm_perc <= 10:
                    stage = "Down"
                if right_arm_perc >= 95 and stage == "Down":
                    stage = "Up"
                    if counter < reps:
                        counter += 1

            if lm_list[15][1] < lm_list[11][1] and lm_list[13][0] > lm_list[15][0]:
                left_arm_points = [lm_list[11], lm_list[13], lm_list[15]]
                for point in left_arm_points:
                    cv2.circle(img, point[:2], 8, points_color, -1)
                    cv2.circle(img, point[:2], 12, points_color, 2)

                left_arm_angle = angle3pt(lm_list[11], lm_list[13], lm_list[15])
                # angle = 360 - angle

                left_arm_perc = np.interp(left_arm_angle, (65, 110), (0, 100))
                left_arm_bar = np.interp(left_arm_angle, (65, 110), (600, 300))

                if left_arm_perc <= 10:
                    stage2 = "Down"
                if left_arm_perc >= 95 and stage2 == "Down":
                    stage2 = "Up"
                    if counter2 < reps:
                        counter2 += 1
        else:
            cv2.putText(img, f"You have to face forward", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
    else:
        cv2.putText(img, f"Hips must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    return [right_arm_angle, right_arm_perc, right_arm_bar, left_arm_angle, left_arm_perc, left_arm_bar]


def get_crunch_data(points_lm: list, min_angle: int, max_angle: int) -> list:
    global counter
    global stage

    crunch_perc, crunch_bar, crunch_angle = 0, 600, 0

    if lm_list[24][2] < lm_list[23][2]:
        leg_points = points_lm
        leg_angle = angle3pt(leg_points[0], leg_points[1], leg_points[2])

        crunch_point = lm_list[12]
        crunch_angle = angle3pt(crunch_point, lm_list[24], lm_list[24])
    else:
        leg_points = [lm_list[23], lm_list[25], lm_list[27]]
        leg_angle = angle3pt(leg_points[0], leg_points[1], leg_points[2])

        crunch_point = lm_list[11]
        crunch_angle = angle3pt(crunch_point, lm_list[23], lm_list[23])
        crunch_angle = 180 - crunch_angle

    crunch_angle = crunch_angle * -1 if crunch_angle < 0 else crunch_angle

    for leg_point in leg_points + [crunch_point]:
        cv2.circle(img, leg_point[:2], 8, points_color, -1)
        cv2.circle(img, leg_point[:2], 12, points_color, 2)

    cv2.putText(img, f"Leg angle: {int(leg_angle)} degr.", (10, 280), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    if 50 < leg_angle < 100:
        crunch_perc = np.interp(crunch_angle, (min_angle, max_angle), (100, 0))
        crunch_bar = np.interp(crunch_angle, (min_angle, max_angle), (300, 600))
        # print(crunch_bar)
        cv2.rectangle(img, (10, 300), (85, 600), (255, 0, 255), 3)
        cv2.rectangle(img, (10, int(crunch_bar)), (85, 600), (255, 0, 255), cv2.FILLED)

        if crunch_perc >= 95 and stage == "Down":
            stage = "Up"
            counter += 1
        if crunch_perc <= 5:
            stage = "Down"

    return [crunch_perc, crunch_bar, crunch_angle]


def get_plank_data(arm_landmarks, leg_landmarks) -> list:
    global counter
    global stage

    diff = lm_list[23][0] - lm_list[27][0]
    if diff < -1:
        diff = diff * -1

    angle_leg = angle3pt(leg_landmarks[0], leg_landmarks[1], leg_landmarks[2])
    angle_arm = angle3pt(arm_landmarks[0], arm_landmarks[1], arm_landmarks[2])

    if diff > 100:
        z1 = round(lm_list[12][2], 3)
        z2 = round(lm_list[11][2], 3)
        z_diff = round(z1 - z2, 3)

        if z_diff < 0:
            z_diff = z_diff * -1

        if 0 < z_diff < 0.1:
            angle_arm = 360 - angle_arm
            angle_leg = 360 - angle_leg
        else:
            if lm_list[12][2] > lm_list[11][2]:
                for point in [lm_list[11], lm_list[23], lm_list[27]]:
                    cv2.circle(img, point[:2], 8, points_color, -1)
                    cv2.circle(img, point[:2], 12, points_color, -2)

                angle_arm = 360 - angle_arm
                angle_leg = 360 - angle_leg
            else:
                for point in [lm_list[12], lm_list[24], lm_list[26]]:
                    cv2.circle(img, point[:2], 8, points_color, -1)
                    cv2.circle(img, point[:2], 12, points_color, -2)

        if 70 < angle_arm < 110 and 160 < angle_leg < 200:
            stage = "Plank"
        else:
            stage = "-"

    return [angle_arm, angle_leg]


def get_data_for_bic_curl(points_lm: list, min_angle: int, max_angle: int, second=False) -> list:
    """
    Zeby troche skrocic kod, bo curle uzywaja tego samego

    :param points_lm:
    :param min_angle:
    :param max_angle:
    :return:
    """
    # print(points_lm)
    global counter
    global stage
    global counter2
    global stage2

    arm_perc, arm_bar, arm_angle = 0, 600, 0

    z1 = round(lm_list[12][2], 3)
    z2 = round(lm_list[11][2], 3)
    z_diff = round(z1 - z2, 3)

    if lm_list[24][3] > 0.7 or lm_list[23][3] > 0.7:
        if lm_list[12][2] > lm_list[11][2]:
            back_angle = angle2pt(lm_list[11], lm_list[23])
        else:
            back_angle = angle2pt(lm_list[24], lm_list[12]) * -1

        if 70 < back_angle < 110:
            for point_lm in points_lm:
                cv2.circle(img, point_lm[:2], 8, points_color, -1)
                cv2.circle(img, point_lm[:2], 12, points_color, 2)

            if z_diff < 0:
                z_diff = z_diff * -1

            if z_diff > 0 or z_diff < 0.08:
                min_angle = min_angle + 20

            arm_angle = angle3pt(points_lm[0], points_lm[1], points_lm[2])
            arm_angle = 360 - arm_angle if arm_angle > 180 else arm_angle

            arm_perc = np.interp(arm_angle, (min_angle, max_angle), (100, 0))
            arm_bar = np.interp(arm_angle, (min_angle, max_angle), (300, 600))
            if second is False:
                if (angle2pt(lm_list[14], lm_list[12]) * -1) > 55:
                    if arm_angle < 200:
                        if arm_perc <= 5:
                            stage = "Down"
                        if arm_perc >= 95 and stage == 'Down':
                            stage = "Up"
                            if counter < reps:
                                counter += 1
                else:
                    cv2.putText(img, f"Keep your right elbow closer to hips", (300, 620),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 2)
            else:
                if angle2pt(lm_list[11], lm_list[13]) > 55:
                    if arm_angle < 200:
                        if arm_perc <= 5:
                            stage2 = "Down"
                        if arm_perc >= 95 and stage2 == 'Down':
                            stage2 = "Up"
                            if counter2 < reps:
                                counter2 += 1
                else:
                    cv2.putText(img, f"Keep your left elbow closer to hips", (300, 500),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), )
        else:
            cv2.putText(img, "Stand fairly upright", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
    else:
        cv2.putText(img, f"Hips must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    return [arm_perc, arm_bar, arm_angle]


def get_pushups_data(points_lm: list, min_angle: int, max_angle: int) -> list:
    global counter
    global stage
    global pushup_exc

    arm_perc, arm_bar, arm_angle = 0, 600, 0
    allow = False

    arm_angle = angle3pt(points_lm[0], points_lm[1], points_lm[2])

    # sprawdzanie ktore ramie jest blizej kamery (to istotne bo potem wala sie katy)
    z1 = round(lm_list[12][2], 3)
    z2 = round(lm_list[11][2], 3)
    z_diff = round(z1 - z2, 3)
    z_diff = z_diff * -1 if z_diff < 0 else z_diff
    cv2.putText(img, f"{lm_list[15][1] > lm_list[13][1]}", (300, 500), cv2.FONT_HERSHEY_PLAIN, 4,(0, 0, 200), 4)
    cv2.putText(img, f"{z_diff}", (300, 700), cv2.FONT_HERSHEY_PLAIN, 4,(0, 0, 200), 4)
    if z_diff < 0.08:


        # jak ziomus ustawiony przodem
        z_shoulder = round(lm_list[12][2], 3)
        z_hip = round(lm_list[24][2], 3)

        # Sprawdza czy przedramie jest nizej niz lokiec i roznice Z miedzy barkiem a biodrem (czy bark jest blizej)
        z_hip_shoulder = round(z_shoulder - z_hip, 3)
        if lm_list[15][1] > lm_list[13][1]:
            arm_angle = 360 - arm_angle
            allow = True
    else:
        # tu jak bokiem
        pushup_exc = False
        if lm_list[12][2] > lm_list[11][2]:
            # Zeby reverse pushupow nie mozna bylo robic ogolnie
            diff = lm_list[23][0] - lm_list[11][0]
            points = [lm_list[11], lm_list[13], lm_list[15]]
            if diff > 170:
                arm_angle = 360 - arm_angle
                allow = True
        else:
            diff = (lm_list[24][0] - lm_list[12][0]) * -1
            points = [lm_list[12], lm_list[14], lm_list[16]]
            if diff > 170:
                allow = True

        for point in points:
            cv2.circle(img, point[:2], 8, points_color, -1)
            cv2.circle(img, point[:2], 12, points_color, 2)

    if allow:
        arm_perc = np.interp(arm_angle, (min_angle, max_angle), (100, 0))
        arm_bar = np.interp(arm_angle, (min_angle, max_angle), (300, 600))

        if arm_perc <= 10:
            stage = "Up"
        if arm_perc >= 90 and stage == "Up":
            stage = "Down"
            counter += 1
    else:
        cv2.putText(img, f"Hips must be visible and get into push up pos", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    return [arm_perc, arm_bar, arm_angle]


def get_squat_data(points_lm: list) -> list:
    global counter
    global stage

    perc, bar, angle = 0, 600, 0
    if lm_list[27][3] > 0.6:
        if lm_list[25][1] - lm_list[23][1] > -10: # czy kolano nie jest za wysoko
            angle = angle3pt(points_lm[2], points_lm[1], points_lm[0])

            z1 = round(lm_list[26][2], 3)
            z2 = round(lm_list[25][2], 3)
            z_diff = round(z1 - z2, 3)

            if z_diff < 0:
                z_diff = z_diff * -1

            if z_diff < 0.08:
                perc = np.interp(angle, (110, 160), (100, 0))
                bar = np.interp(angle, (110, 160), (300, 600))
            else:
                if lm_list[25][2] > lm_list[26][2]:
                    points_lm = [lm_list[24], lm_list[26], lm_list[28]]
                    angle = 360 - angle if angle > 180 else angle
                else:
                    angle = 360 - angle if angle > 180 else angle

                perc = np.interp(angle, (70, 160), (100, 0))
                bar = np.interp(angle, (70, 160), (300, 600))

            for point_lm in points_lm:
                cv2.circle(img, point_lm[:2], 8, points_color, -1)
                cv2.circle(img, point_lm[:2], 12, points_color, 2)

            if perc <= 5:
                stage = "Down"
            if perc >= 95 and stage == "Down":
                stage = "Up"
                counter += 1
        else:
            cv2.putText(img, f"Lower your knee", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)
    else:
        cv2.putText(img, f"Ankles must be visible", (300, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 200), 4)

    return [perc, bar, angle]


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def angle2pt(a, b):
    change_inx = b[0] - a[0]
    change_iny = b[1] - a[1]
    ang = math.degrees(math.atan2(change_iny, change_inx))
    return ang


def get_landmarks(img: np.array, draw=False):
    h, w, _ = img.shape
    landmark_list = []

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result_class = pose.process(hsv_img)
    result = result_class.pose_landmarks

    if result:
        if draw:
            mp_draw.draw_landmarks(img, result, mp_pose.POSE_CONNECTIONS)
        for id, landmark in enumerate(result.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            v = landmark.visibility
            landmark_list.append((x, y, z, v))

    return landmark_list


def display_info(reps_pos: tuple, angle_pos: tuple, perc_pos: tuple, bar_pos: list, barfill_pos: list, stage_pos: tuple,
                 angle, perc, counter, c_set, reps, sets, stage, img):

    cv2.putText(img, f"Sets: {c_set}/{sets}", (reps_pos[0], reps_pos[1] - 40), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    cv2.putText(img, f"Reps: {counter}/{reps}", reps_pos, cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    cv2.putText(img, f"Angle: {int(angle)} degr.", angle_pos, cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    if 5 < perc < 95:
        cv2.putText(img, "Stage: -", stage_pos, cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    else:
        cv2.putText(img, "Stage: " + stage, stage_pos, cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)

    color = (255, 0, 255)
    if perc == 100:
        color = (0, 255, 0)
    if perc == 0:
        color = (0, 0, 255)

    cv2.putText(img, f"{int(perc)}%", perc_pos, cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
    cv2.rectangle(img, bar_pos[0], bar_pos[1], color, 3)
    cv2.rectangle(img, barfill_pos[0], barfill_pos[1], color, cv2.FILLED)


def get_text_center(text, x1, y1, x2, y2, size, t) -> list:
    # get boundary of this text
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, size, t)[0]

    w = x2 - x1
    h = y2 - y1

    textX = x1 + w // 2
    textY = y1 + h // 2

    return [textX, textY, textsize]


modes = {
    0: {
        "mode": "Right arm biceps curl",
        "points": [11, 13, 15],
        "angles": [40, 160],
        "func": get_data_for_bic_curl
        },
    1: {
        "mode": "Left arm biceps curl",
        "points": [12, 14, 16],
        "angles": [40, 160],
        "func": get_data_for_bic_curl
        },

    2: {
        "mode": "Both arms biceps curl"
        },
    3: {
        "mode": "Pushups",
        "points": [11, 13, 15],
        "angles": [70, 155],
        "func": get_pushups_data
        },

    4: {
        "mode": "Squats",
        "points": [23, 25, 27],
        "angles": [],
        "func": get_squat_data
        },

    5: {
        "mode": "Plank"
        },
    6: {
        "mode": "Crunches",
        "points": [26, 24, 28],
        "angles": [170, 180],
        "func": get_crunch_data
         },
    7: {
        "mode": "Over head Dumbbell Press",
        "func": get_overhead_dumbbell_data
        },
    8: {
        "mode": "Dum bbell lateral raise",
        "func": get_dumbbell_lateral_raise_data
        },
    9: {
        "mode": "Reverse pushup from chair",
        "points": [11, 13, 15],
        "angles": [],
        "func": get_reverse_pushup_data
        },
    10: {
        "mode": "Pullups",
        "points": [0, 0, 0],
        "angles": [],
        "func": get_pullup_data
    },
    11: {
        "mode": "Dumbbell row",
        "points": [0, 0, 0],
        "angles": [],
        "func": get_rowing_data
    },
}


with open("config.json", 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)


# cap = cv2.VideoCapture(r"Video/biceps_curls.mp4")
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=.6)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Screen
screen = "menu"

# Workout data
mode = 2
# reps = 10
# sets = 5
# break_seconds = 5
# c_second = break_seconds


exceptions_list = [2, 5, 7, 8]

# Counter, stage, current set
counter = 0
counter2 = 0
stage = "-"
stage2 = "-"
current_set = 0

set_timer = False  # timer przerw miedzy powtorzeniami
stop = False  # zeby nic sie wykrywalo kiedy jest przerwa

# cursor
click = False
cursor = 0, 0
cursor_color = (255, 0, 255)
nwm = []
cleared = False

# colors of interface
info_color = (0, 0, 255)
points_color = (255, 0, 125)

# tam jest ladowanie konfigow z config.json w petli a klikniecie w jakis button od tych repsow itp, to blokuje, zeby
# sie co chwile nie resetowalo
switcher = False


# Flagi zwiazane z edycja konfigu
config_reset = False  # do tego, by aktualizowac konfig po zmianach ( w workout menu sie zmienia konfig i laduje,
# wiec trzeba bylo zmienic screen na inny i wrocic)
config_save = False

config_save_timer_msg = False  # timer od tego ze wiadomosc stoi przez pare 5
config_reset_timer_msg = False
while True:
    x1, y1 = 0, 0
    x2, y2 = 0, 0

    success, img = cap.read()
    toproc = img.copy()
    img = cv2.resize(img, (1280, 720))
    if success is False:
        break
    # img = cv2.imread("Video/brzuszki2.jpg")

    # img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_landmarks = []
    left_h = False
    right_h = False
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)
            for lm in hand_lm.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)

                hand_landmarks.append((x, y))

    if len(hand_landmarks) != 0:
        cursor = hand_landmarks[8]

        x1, y1 = hand_landmarks[8]
        x2, y2 = hand_landmarks[12]

        finger_dist = math.hypot(x2 - x1, y2 - y1)

        if finger_dist < 40:
            if cleared:
                click = False
            nwm.append(1)

            cleared = False
            if len(nwm) == 8:
                cursor_color = (0, 200, 0)
                click = True
                nwm.clear()
                cleared = True
        else:
            cursor_color = (255, 0, 255)
            click = False

    if screen == "menu":
        if config_reset:
            screen = "workout menu"

        if config_save:
            screen = "workout menu"

        switcher = False
        # navbar
        cv2.rectangle(img, (0, 0), (1280, 80), (255, 123, 50), -1)
        cv2.rectangle(img, (0, 0), (455, 80), (91, 64, 61), -1)
        cv2.putText(img, f"Konon Trener AI", (20, 52), cv2.FONT_HERSHEY_PLAIN, 3, (255, 123, 50), 4)
        cv2.rectangle(img, (1180, 0), (1280, 80), (0, 0, 200), -1)
        cv2.putText(img, f"X", (1210, 66), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        if 1050 < cursor[0] < 1200 and 20 < cursor[1] < 120 and click:
            break
        rect_padding = 50
        padding = 5
        rect_x1, rect_y1 = 50, 110
        rect_x2, rect_y2 = 310, 290

        for index, i in enumerate(modes.keys()):
            if index % 3 == 0 and index != 0:
                rect_x1, rect_y1 = 310 + rect_padding, 110
                rect_x2, rect_y2 = 560 + rect_padding, 290

            if index % 6 == 0 and index != 0:
                rect_x1, rect_y1 = 560 + (2 * rect_padding), 110
                rect_x2, rect_y2 = 820 + (2 * rect_padding), 290

            if index % 9 == 0 and index != 0:
                rect_x1, rect_y1 = 820 + (3 * rect_padding), 110
                rect_x2, rect_y2 = 1080 + (3 * rect_padding), 290

            if len(modes[i]["mode"]) < 10:
                cv2.rectangle(img, (rect_x1 - padding, rect_y1 - padding), (rect_x2 + padding, rect_y2 + padding),
                              (255, 255, 255), -1)
                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 100), -1)

                text = modes[i]['mode']
                text_x, text_y, text_size = get_text_center(text, rect_x1, rect_y1, rect_x2, rect_y2, 3, 5)
                cv2.putText(img, f"{text}", (text_x - text_size[0] // 2, text_y + text_size[1] // 2),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 5)

                if rect_x1 < cursor[0] < rect_x2 and rect_y1 < cursor[1] < rect_y2 and click:
                    mode = i
                    screen = "workout menu"
            else:
                raw = modes[i]['mode'].split(" ")

                cv2.rectangle(img, (rect_x1 - padding, rect_y1 - padding), (rect_x2 + padding, rect_y2 + padding),
                              (255, 255, 255), -1)
                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 100), -1)
                if len(raw) == 2:
                    adj = 85
                else:
                    adj = 45

                for text in raw:
                    text_x, text_y, text_size = get_text_center(text, rect_x1, rect_y1, rect_x2, rect_y2, 3, 5)
                    cv2.putText(img, f"{text}", (text_x - text_size[0] // 2, rect_y1 + adj), cv2.FONT_HERSHEY_PLAIN, 3,
                                (255, 0, 255), 5)
                    adj += 40

                if rect_x1 < cursor[0] < rect_x2 and rect_y1 < cursor[1] < rect_y2 and click:
                    mode = i
                    screen = "workout menu"

            rect_y1 += 200
            rect_y2 += 200
        # screen = "workout menu"
    if screen == "workout menu":
        if switcher is False:
            reps = json_object[modes[mode]['mode']]['reps']
            sets = json_object[modes[mode]['mode']]['sets']
            break_seconds = json_object[modes[mode]['mode']]['break']
            c_second = break_seconds

        # Navbar
        cv2.rectangle(img, (0, 0), (1280, 80), (255, 123, 50), -1)
        cv2.putText(img, f"Mode: {modes[mode]['mode']}", (20, 52), cv2.FONT_HERSHEY_PLAIN, 3, (91, 64, 61), 4)

        # Reps
        cv2.rectangle(img, (10, 120), (320, 280), (200, 100, 0), -1)
        text = f"Reps: {reps}"
        text_x, text_y, text_size = get_text_center(text, 10, 120, 320, 280, 3.5, 8)
        cv2.putText(img, text, (text_x - text_size[0] // 2, text_y + text_size[1] // 2), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 255), 8)

        cv2.rectangle(img, (320, 120), (470, 200), (255, 0, 100), -1)
        cv2.line(img, (340, 180), (395, 140), (255, 0, 255), 8)
        cv2.line(img, (395, 140), (450, 180), (255, 0, 255), 8)

        cv2.rectangle(img, (320, 200), (470, 280), (125, 0, 0), -1)
        cv2.line(img, (340, 220), (395, 260), (255, 0, 255), 8)
        cv2.line(img, (395, 260), (450, 220), (255, 0, 255), 8)

        if 320 < cursor[0] < 470 and 120 < cursor[1] < 200 and click:
            reps += 1
            switcher = True
        if 320 < cursor[0] < 470 and 200 < cursor[1] < 280 and click:
            if reps != 0:
                reps -= 1
                switcher = True

        # Sets
        cv2.rectangle(img, (10, 300), (320, 460), (200, 100, 0), -1)
        text = f"Sets: {sets}"
        text_x, text_y, text_size = get_text_center(text, 10, 300, 320, 460, 3.5, 8)
        cv2.putText(img, text, (text_x - text_size[0] // 2, text_y + text_size[1] // 2), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 255), 8)

        cv2.rectangle(img, (320, 300), (470, 380), (255, 0, 100), -1)
        cv2.line(img, (340, 360), (395, 320), (255, 0, 255), 8)
        cv2.line(img, (395, 320), (450, 360), (255, 0, 255), 8)

        cv2.rectangle(img, (320, 380), (470, 460), (125, 0, 0), -1)
        cv2.line(img, (340, 400), (395, 440), (255, 0, 255), 8)
        cv2.line(img, (395, 440), (450, 400), (255, 0, 255), 8)

        if 320 < cursor[0] < 470 and 300 < cursor[1] < 380 and click:
            sets += 1
            switcher = True

        if 320 < cursor[0] < 470 and 380 < cursor[1] < 460 and click:
            if sets != 0:
                sets -= 1
                switcher = True

        # Break
        cv2.rectangle(img, (10, 480), (320, 640), (200, 100, 0), -1)
        text = f"Break: {break_seconds}"
        text_x, text_y, text_size = get_text_center(text, 10, 480, 320, 640, 3.5, 8)
        cv2.putText(img, text, (text_x - text_size[0] // 2, text_y + text_size[1] // 2), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 255), 8)

        cv2.rectangle(img, (320, 480), (470, 560), (255, 0, 100), -1)
        cv2.line(img, (340, 540), (395, 500), (255, 0, 255), 8)
        cv2.line(img, (395, 500), (450, 540), (255, 0, 255), 8)

        cv2.rectangle(img, (320, 560), (470, 640), (125, 0, 0), -1)
        cv2.line(img, (340, 580), (395, 620), (255, 0, 255), 8)
        cv2.line(img, (395, 620), (450, 580), (255, 0, 255), 8)

        if 320 < cursor[0] < 470 and 480 < cursor[1] < 560 and click:
            break_seconds += 1
            c_second = break_seconds
            switcher = True

        if 320 < cursor[0] < 470 and 560 < cursor[1] < 640 and click:
            if break_seconds != 0:
                break_seconds -= 1
                c_second = break_seconds
                switcher = True

        # Save config
        padding = 10
        cv2.rectangle(img, (800 - padding, 120 - padding), (1000 + padding, 220 + padding), (255, 255, 255), -1)
        cv2.rectangle(img, (800, 120), (1000, 220), (255, 0, 0), -1)
        cv2.putText(img, f"Save", (840, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(img, f"Config", (820, 205), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        if 800 < cursor[0] < 1000 and 120 < cursor[1] < 220 and click:
            json_object[modes[mode]['mode']]['reps'] = reps
            json_object[modes[mode]['mode']]['sets'] = sets
            json_object[modes[mode]['mode']]['break'] = break_seconds

            to_save = json.dumps(json_object, indent=4)
            config_file = open("config.json", "w+")
            config_file.write(to_save)
            config_file.close()
            config_save = True
            screen = "menu"
            print("Configuration saved")

        # Reset config to default
        cv2.rectangle(img, (1050 - padding, 120 - padding), (1250 + padding, 220 + padding), (255, 255, 255), -1)
        cv2.rectangle(img, (1050, 120), (1250, 220), (255, 0, 0), -1)
        cv2.putText(img, f"Reset", (1080, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(img, f"Config", (1070, 205), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        if 1050 < cursor[0] < 1250 and 120 < cursor[1] < 220 and click:
            json_object[modes[mode]['mode']]['reps'] = 15
            json_object[modes[mode]['mode']]['sets'] = 3
            json_object[modes[mode]['mode']]['break'] = 10

            to_save = json.dumps(json_object, indent=4)
            config_file = open("config.json", "w+")
            config_file.write(to_save)
            config_file.close()
            print("Config set to default")
            config_reset = True
            screen = "menu"

        # Start workout
        cv2.rectangle(img, (590 - padding, 485 - padding), (940 + padding, 640 + padding), (255, 255, 255), -1)
        cv2.rectangle(img, (590, 485), (940, 640), (255, 100, 0), -1)
        cv2.putText(img, f"START", (640, 590), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
        if 590 < cursor[0] < 940 and 485 < cursor[1] < 640 and click:
            screen = "workout"

        cv2.rectangle(img, (1180, 0), (1280, 80), (0, 0, 200), -1)
        cv2.putText(img, f"X", (1210, 66), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        if 1180 < cursor[0] < 1280 and 0 < cursor[1] < 80 and click:
            screen = "menu"

        # Save config msg
        if config_save:
            if config_save_timer_msg is False:
                time_msg = time()
                config_save_timer_msg = True

            if config_save_timer_msg:
                diff = int(time() - time_msg)
                if diff != 5:
                    cv2.putText(img, f"Configuration saved", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                else:
                    config_save = False
                    config_save_timer_msg = False

        # Reset config msg
        if config_reset:
            if config_reset_timer_msg is False:
                time_msg = time()
                config_reset_timer_msg = True

            if config_reset_timer_msg:
                diff = int(time() - time_msg)
                if diff != 5:
                    cv2.putText(img, f"Configuration restarted", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                else:
                    config_reset = False
                    config_reset_timer_msg = False
        # screen = "workout"
    if screen == "workout":
        lm_list = get_landmarks(img, draw=True)

        if len(lm_list) != 0:
            if mode not in exceptions_list:
                points = [lm_list[modes[mode]['points'][0]], lm_list[modes[mode]['points'][1]],
                          lm_list[modes[mode]['points'][2]]]

                if stop is False:
                    if len(modes[mode]['angles']) != 0:
                        min_ang, max_ang = modes[mode]['angles']
                        perc, bar, angle = modes[mode]['func'](points, min_angle=min_ang, max_angle=max_ang)
                    else:
                        perc, bar, angle = modes[mode]['func'](points)

                    display_info((10, 150), (10, 230), (10, 640), [(10, 300), (85, 600)], [(10, int(bar)), (85, 600)],
                                 (10, 190), angle, perc, counter, current_set, reps, sets, stage, img)

                if counter == reps:
                    if stop is False:
                        current_set += 1

                    if current_set != sets:
                        stop = True

                        circle_c = (int(w / 2), int(h / 2))
                        cv2.circle(img, circle_c, 190, (255, 0, 125), -1)
                        if c_second > 9:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 155, circle_c[1] + 90), cv2.FONT_HERSHEY_SIMPLEX,
                                        8, info_color, 30)
                        else:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 130, circle_c[1] + 135), cv2.FONT_HERSHEY_SIMPLEX,
                                        13, info_color, 30)
                        sleep(1)
                        c_second -= 1

                        if c_second == 0:
                            stop = False
                            counter = 0
                            c_second = break_seconds
                    else:
                        counter = 0

                if current_set == sets:
                    current_set = 0
                    screen = "menu"

            if mode == 2:
                points_left = [lm_list[11], lm_list[13], lm_list[15]]
                points_right = [lm_list[12], lm_list[14], lm_list[16]]

                if stop is False:
                    perc_ra, bar_ra, angle_ra = get_data_for_bic_curl(points_right, min_angle=40, max_angle=160)
                    perc_la, bar_la, angle_la = get_data_for_bic_curl(points_left, min_angle=40, max_angle=160,
                                                                      second=True)

                    display_info((10, 150), (10, 230), (10, 280), [(10, 300), (85, 600)], [(10, int(bar_ra)), (85, 600)],
                                 (10, 190), angle_ra, perc_ra, counter, current_set, reps, sets, stage, img)

                    display_info((1060, 150), (1060, 230), (1060, 280), [(1060, 300), (1135, 600)],
                                 [(1060, int(bar_la)), (1135, 600)], (1060, 190), angle_la, perc_la, counter2,
                                 current_set, reps, sets, stage2, img)

                if counter == reps and counter2 == reps:
                    if stop is False:
                        current_set += 1

                    if current_set != sets:
                        stop = True

                        circle_c = (int(w / 2), int(h / 2))
                        cv2.circle(img, circle_c, 190, (255, 0, 125), -1)
                        if c_second > 9:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 155, circle_c[1] + 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 8, info_color, 30)
                        else:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 130, circle_c[1] + 135),
                                        cv2.FONT_HERSHEY_SIMPLEX, 13, info_color, 30)
                        sleep(1)
                        c_second -= 1

                        if c_second == 0:
                            stop = False
                            counter = 0
                            counter2 = 0
                            c_second = break_seconds
                    else:
                        counter = 0
                        counter2 = 0

                if current_set == sets:
                    screen = "menu"
                    current_set = 0

            if mode == 5:
                if stop is False:
                    c_time = 0
                    points_arm = [lm_list[11], lm_list[13], lm_list[15]]
                    points_leg = [lm_list[23], lm_list[25], lm_list[27]]
                    arm_angle, leg_angle = get_plank_data(points_arm, points_leg)
                    if stage == "Plank":
                        if set_timer is False:
                            start_time = time()
                            set_timer = True

                        time_diff = int(time() - start_time)
                        if time_diff == 1:
                            counter += 1
                            set_timer = False
                    else:
                        set_timer = False

                    cv2.putText(img, f"Sets: {current_set}/{sets}", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
                    cv2.putText(img, f"Reps: {counter}/{reps}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
                    cv2.putText(img, f"Arm angle: {int(arm_angle)}", (10, 190), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
                    cv2.putText(img, f"Leg angle: {int(leg_angle)}", (10, 230), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)
                    cv2.putText(img, f"Stage: {stage}", (10, 270), cv2.FONT_HERSHEY_PLAIN, 2, info_color, 2)

                if stage == "Plank":
                    if counter == reps:
                        set_timer = False
                        if stop is False:
                            current_set += 1

                        if current_set != sets:
                            stop = True
                            circle_c = (int(w / 2), int(h / 2))
                            cv2.circle(img, circle_c, 190, (255, 0, 125), -1)
                            if c_second > 9:
                                cv2.putText(img, f"{c_second}", (circle_c[0] - 155, circle_c[1] + 90),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            8, info_color, 30)
                            else:
                                cv2.putText(img, f"{c_second}", (circle_c[0] - 130, circle_c[1] + 135),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            13, info_color, 30)
                            sleep(1)
                            c_second -= 1

                            if c_second == 0:
                                stop = False
                                counter = 0
                                c_second = break_seconds
                        else:
                            counter = 0

                    if current_set == sets:
                        current_set = 0
                        screen = "menu"

            if mode == 7 or mode == 8:
                if stop is False:
                    right_arm_angle, right_arm_perc, right_arm_bar, \
                    left_arm_angle, left_arm_perc, left_arm_bar = modes[mode]['func']()

                    display_info((10, 150), (10, 230), (10, 640), [(10, 300), (85, 600)],
                                 [(10, int(right_arm_bar)), (85, 600)],
                                 (10, 190), right_arm_angle, right_arm_perc, counter, current_set, reps, sets, stage, img)

                    display_info((1060, 150), (1060, 230), (1060, 280), [(1060, 300), (1135, 600)],
                                 [(1060, int(left_arm_bar)), (1135, 600)], (1060, 190), left_arm_angle,
                                 left_arm_perc,
                                 counter2, current_set, reps, sets, stage2, img)

                if counter == reps and counter2 == reps:
                    if stop is False:
                        current_set += 1

                    if current_set != sets:
                        stop = True

                        circle_c = (int(w / 2), int(h / 2))
                        cv2.circle(img, circle_c, 190, (255, 0, 125), -1)
                        if c_second > 9:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 155, circle_c[1] + 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 8, info_color, 30)
                        else:
                            cv2.putText(img, f"{c_second}", (circle_c[0] - 130, circle_c[1] + 135),
                                        cv2.FONT_HERSHEY_SIMPLEX, 13, info_color, 30)
                        sleep(1)
                        c_second -= 1

                        if c_second == 0:
                            stop = False
                            counter = 0
                            counter2 = 0
                            c_second = break_seconds
                    else:
                        counter = 0
                        counter2 = 0

                if current_set == sets:
                    screen = "menu"
                    current_set = 0

        # Navbar
        cv2.rectangle(img, (0, 0), (1280, 80), (255, 123, 50), -1)
        cv2.putText(img, f"Mode: {modes[mode]['mode']}", (20, 52), cv2.FONT_HERSHEY_PLAIN, 3, (91, 64, 61), 4)
        cv2.rectangle(img, (1180, 0), (1280, 80), (0, 0, 200), -1)
        cv2.putText(img, f"X", (1210, 66), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        if 1180 < cursor[0] < 1280 and 0 < cursor[1] < 80 and click:
            screen = "menu"
            current_set = 0

    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
        cv2.circle(img, (x1, y1), 20, (255, 0, 255), -1)
        cv2.circle(img, (x2, y2), 20, (255, 0, 255), -1)
        cv2.circle(img, (int((x1 + x2)/2), int((y1 + y2)/2)), 20, cursor_color, -1)
        cv2.line(img, (x1, y1), (x2, y2), cursor_color, 6)

    cv2.imshow("Res", img)
    key = cv2.waitKey(50)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
