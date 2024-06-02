"""
所有者:余国龙
程序的作用:
1. 获取摄像头的视频数据
2. 加载手部关节点检测模型
3. 将摄像头数据载入模型做推理
4. 获取每个关键点的绝对值坐标
5. 计算手部动作内容
"""
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import HandLandmarkerResult


class Video:
    def __init__(self, id=0, image_size=196, fps=30):
        self.id = id
        self.image_size = image_size
        self.fps = fps
        self.set_capture()
        self.hand_detection()

    def set_capture(self):
        self.capture = cv2.VideoCapture(self.id)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        # self.capture.set(cv2.CAP_PROP_WIDTH, self.image_size)
        # self.capture.set(cv2.CAP_PROP_HEIGHT, self.image_size)

    def hand_detection(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        # 使用直播模式创建一个手部地标实例：
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./model/hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result)

        self.hand_detector = HandLandmarker.create_from_options(options)
        self.hand_result = None

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.hand_result = result
        if len(self.hand_result.hand_world_landmarks) > 0:
            hand_landmarks = self.hand_result.hand_landmarks[0]
            hand_landmarks_dict = {}
            for index in range(21):
                hand_landmarks_dict[f'{index}'] = ""
                hand_landmarks_dict[f'{index}'] = np.array([hand_landmarks[index].x, hand_landmarks[index].y, hand_landmarks[index].z])
            # 拇指和食指捏合
            # print(np.linalg.norm(hand_landmarks_dict["4"] - hand_landmarks_dict["8"]))
            x = hand_landmarks_dict["0"][0] - hand_landmarks_dict["12"][0]
            y = hand_landmarks_dict["0"][1] - hand_landmarks_dict["12"][1]
            if np.linalg.norm(hand_landmarks_dict["8"] - hand_landmarks_dict["4"]) < 0.05:
                print(f"点击屏幕:{np.linalg.norm(hand_landmarks_dict['8'] - hand_landmarks_dict['4'])}")

            elif x > y and x > 0:
                print("向右")
            elif x > y and x < 0:
                print("向下")
            elif x < y and y < 0:
                print("向左")
            elif x < y and y > 0:
                print("向上")

            else:
                print("未知指令！")
            print("-" * 20)

    def get_time(self):
        return int(time.time() * 1000)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # # Get the top left corner of the detected hand's bounding box.
            # height, width, _ = annotated_image.shape
            # x_coordinates = [landmark.x for landmark in hand_landmarks]
            # y_coordinates = [landmark.y for landmark in hand_landmarks]
            # text_x = int(min(x_coordinates) * width)
            # text_y = int(min(y_coordinates) * height) - MARGIN
            #
            # # Draw handedness (left or right hand) on the image.
            # cv2.putText(annotated_image, f"{handedness[0].category_name}",
            #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        return annotated_image

    def run(self):
        while True:
            ret, frame = self.capture.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.hand_detector.detect_async(mp_image, self.get_time())
            if ret == False:
                break
            if self.hand_result:
                frame = self.draw_landmarks_on_image(frame, self.hand_result)

            cv2.imshow("image", frame)
            state = cv2.waitKey(1)
            if state == 27:
                break

        self.capture.release()
        cv2.destroyWindow()
