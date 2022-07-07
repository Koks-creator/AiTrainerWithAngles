import mediapipe as mp
import cv2
from time import sleep, time
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split  # pozwoli na podzielenie danych na do trenowania i do testow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler  # standaryzuje dane
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score
import math


def get_landmarks(img: np.array, draw=False):
    h, w, _ = img.shape
    landmark_list = []
    pose_data = []

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result_class = pose.process(hsv_img)
    result = result_class.pose_landmarks

    if result:
        if draw:
            mp_draw.draw_landmarks(img, result, mp_pose.POSE_CONNECTIONS)
            pose_data = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in result.landmark]).flatten())

        for landmark in result.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            v = landmark.visibility
            landmark_list.append((x, y, z, v))

    return landmark_list, pose_data


def angle2pt(a, b):
    change_inx = b[0] - a[0]
    change_iny = b[1] - a[1]
    ang = math.degrees(math.atan2(change_iny, change_inx))
    return ang


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


cap = cv2.VideoCapture(r"video path")
# cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

filename = "DumbbellRow.csv"
# class_name = "BicepCurlsUp"  # lewy i prawy up juz byl
stage = ""
counter = 0


def collect_data():
    while True:
        success, img = cap.read()
        if success is False:
            break
        # print()
        img = cv2.resize(img, (1280, 720))
        # img = cv2.flip(img, 1)
        # img = cv2.resize(img, (720, 1280))
        lm_list, pose_data = get_landmarks(img, draw=True)

        if len(lm_list) != 0:
            # angle = angle3pt(lm_list[12], lm_list[14], lm_list[16])
            # angle = 360 - angle if angle > 180 else angle
            #
            # angle2 = angle3pt(lm_list[11], lm_list[13], lm_list[15])
            # angle2 = 360 - angle2 if angle2 > 180 else angle2
            #
            angle = angle2pt(lm_list[12], lm_list[14])
            angle = 360 - angle if angle > 180 else angle

            angle2 = angle2pt(lm_list[11], lm_list[13])
            angle2 = 360 - angle2 if angle2 > 180 else angle2

            cv2.circle(img, lm_list[11][:2], 5, (255, 0, 255), -1)
            cv2.putText(img, f"L: {int(angle2)}, R: {int(angle)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

            if angle > 160:
                # print(angle)
                print(angle)
                cv2.putText(img, f"Saving", (10, 250), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
                class_name = "Up"

                if os.path.exists(filename) is False:
                    columns = ["class"]
                    for val in range(1, 34):
                        columns += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

                    with open(filename, "w") as f:
                        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(columns)

                if os.path.exists(filename):
                    pose_data.insert(0, class_name)
                    with open(filename, "a", newline="") as f:
                        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(pose_data)

        cv2.imshow("Res", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def train_data():
    df = pd.read_csv(filename)

    x = df.drop("class", axis=1)
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

    pipelines = {
        #"lr": make_pipeline(StandardScaler(), LogisticRegression()),
        #"rc": make_pipeline(StandardScaler(), RidgeClassifier()),
        "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
        #"gb": make_pipeline(StandardScaler(), GradientBoostingClassifier())
    }

    fit_models = {}
    print("Starting training")
    for algo, pipeline in pipelines.items():
        print(f"Training with: {algo}")
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        print(f"predict with {algo}")
        yhat = model.predict(x_test)
        print(algo, accuracy_score(y_test, yhat))

    with open("dumbbell_row_model", "wb") as f:
        pickle.dump(fit_models["rf"], f)


def test():
    global stage
    global counter
    with open("model3", "rb") as f:
        model = pickle.load(f)

    while True:
        success, img = cap.read()
        if success is False:
            break

        img = cv2.resize(img, (1280, 720))
        # img = cv2.flip(img, 1)
        # img = cv2.resize(img, (720, 1280))
        lm_list, pose_data = get_landmarks(img, draw=True)

        if len(lm_list) != 0:
            angle = angle3pt(lm_list[11], lm_list[13], lm_list[15])
            angle = 360 - angle if angle > 180 else angle
            # print(angle)

            x = pd.DataFrame([pose_data])
            body_lang_class = model.predict(x)[0]
            body_lang_prob = model.predict_proba(x)[0]
            # print(body_lang_class, body_lang_prob)
            prob = body_lang_prob[np.argmax(body_lang_prob)]
            # print(body_lang_class, prob)
            cv2.putText(img, f"Counter: {counter}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            cv2.putText(img, f"Stage: {stage}", (10, 140), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            cv2.putText(img, f"{body_lang_class}, {prob}", (10, 220), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            if prob > 0.92:
                if "Down" in body_lang_class:
                    stage = "Down"

                if "Up" in body_lang_class and stage == "Down":
                    stage = "Up"
                    counter += 1

                print(stage, counter)

        cv2.imshow("Res", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_from_image():
    global stage
    global counter
    with open("dumbbell_row_model", "rb") as f:
        model = pickle.load(f)

    img = cv2.imread(r"path")
    lm_list, pose_data = get_landmarks(img, draw=True)

    if len(lm_list) != 0:
        angle = angle3pt(lm_list[11], lm_list[13], lm_list[15])
        angle = 360 - angle if angle > 180 else angle
        # print(angle)

        x = pd.DataFrame([pose_data])
        body_lang_class = model.predict(x)[0]
        body_lang_prob = model.predict_proba(x)[0]
        # print(body_lang_class, body_lang_prob)
        prob = body_lang_prob[np.argmax(body_lang_prob)]
        # print(body_lang_class, prob)
        cv2.putText(img, f"Counter: {counter}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
        cv2.putText(img, f"Stage: {stage}", (10, 140), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
        cv2.putText(img, f"{body_lang_class}, {prob}", (10, 220), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
        if prob > 0.92:
            if "Down" in body_lang_class:
                stage = "Down"

            if "Up" in body_lang_class and stage == "Down":
                stage = "Up"
                counter += 1

            print(stage, counter)

    cv2.imshow("Res", img)
    cv2.waitKey(0)


# collect_data()
# train_data()
# test()
# test_from_image()