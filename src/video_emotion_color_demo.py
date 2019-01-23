
import cv2
import logging, sys, time
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from collections import deque


# weights for each of the last five states
arr_prob = [0.05, 0.1, 0.15, 0.3, 0.4]   #total 1
time_init = time.time()
# function that calculates a weighted relation for each emotion with its own previous states
def weigh_up_emotion(arr):
    real_prob = 0
    arr_weighted = deque()
    #mean_last_states = deque()
    weighted_happy = 0
    weighted_sad = 0
    weighted_angry = 0
    weighted_fear = 0
    weighted_neutral = 0

    for i in range(len(arr)):
        for j in arr[i]:
            arr_weighted = j*arr_prob[i]
            weighted_happy += arr_weighted[3]
            weighted_sad += arr_weighted[4]
            weighted_angry += arr_weighted[0] + arr_weighted[1]
            weighted_fear += arr_weighted[2] + arr_weighted[5]
            weighted_neutral += arr_weighted[6]

    mean_last_states = [weighted_angry, weighted_angry, weighted_fear, weighted_happy, weighted_sad, weighted_fear, weighted_neutral]
    return mean_last_states

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 5
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3] #emotion target size (64, 64)

# starting lists for calculating modes
emotion_window = []

# Initialize logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(sys.argv[0].split('/')[-1])

# starting video streaming
cv2.namedWindow('window_frame')
emotions_collected = deque()
video_capture = cv2.VideoCapture("/home/lfernandez/spaceai/propuesta_disa/video_cortado/inicio_barba.mp4")
happy_max = 0


# Count frames just for debugging purposes
nframe = 0
happiness_diagram = deque()
angriness_diagram = deque()
point = 0

plt.close('all')
f, axarr = plt.subplots(2, 1)
i = 0
x_axis = []
y_axis_happiness = []
y_axis_angriness = []
lista = []
label = False

while True:
    bgr_image = cv2.flip(video_capture.read()[1], 1)
    nframe += 1
    if (nframe % 5) != 0:
        continue

    #bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    logger.debug("frame {}: {} faces detected.".format(nframe, faces.__len__()))
    nface = 1

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        #[[ angry 1.36565328e-01   disgust 2.20752245e-05  fear 8.08805823e-02  happy 6.48011118e-02  sad 6.36823952e-01  surprise 3.33598023e-03  neutral 7.75709748e-02]]

        #Calculating parameters for diagram drawing
        angry_prob = emotion_prediction[0][0] + emotion_prediction[0][1]
        fear_prob = emotion_prediction[0][2] + emotion_prediction[0][5]
        happy_prob = emotion_prediction[0][3]
        sad_prob = emotion_prediction[0][4]

        instant_happiness = happy_prob - sad_prob
        instant_angriness = angry_prob - fear_prob

        # time calculation for x-axis
        now = time.time() - time_init
        # x_axis.append(now)

        # emotion intensity for y-axis
        # y_axis_happiness.append(instant_happiness)
        # y_axis_angriness.append(instant_angriness)

        # Drawing "Emotions during time" diagram
        # axarr[0].set_xlim(now-5.0, now+5.0)
        # axarr[0].set_ylim(-1.0, 1.0)
        # axarr[0].spines['bottom'].set_position('zero')
        # if label == False:
        #     axarr[0].plot(x_axis, y_axis_angriness, c='r', label='angry(+) -> fear(-)')
        #     axarr[0].plot(x_axis, y_axis_happiness, c='b', label='happy(+) -> sad(-)')
        #     label = True
        # else:
        #     axarr[0].plot(x_axis, y_axis_angriness, c='r', label='angry(+) -> fear(-)')
        #     axarr[0].plot(x_axis, y_axis_happiness, c='b', label='happy(+) -> sad(-)')
        # axarr[0].set_xlabel('time')
        # axarr[0].set_ylabel('Emotion intensity')
        # axarr[0].spines["top"].set_visible(False)
        # axarr[0].spines["right"].set_visible(False)
        # axarr[0].grid(False)
        # axarr[0].set_title('Emotions during time')
        # plt.savefig('/home/lfernandez/spaceai/EmDiagram.png')

        #writing data for postprocessing
        with open('/home/lfernandez/spaceai/propuesta_disa/your_file.txt', 'a') as f:
            str_val = str(nframe) + ',' + '{:f}'.format(emotion_prediction[0][0]) + ',' + '{:f}'.format(emotion_prediction[0][1]) + ','
            str_val = str_val + '{:f}'.format(emotion_prediction[0][2]) + ',' + '{:f}'.format(emotion_prediction[0][3]) + ',' + '{:f}'.format(emotion_prediction[0][4]) + ','
            str_val = str_val + '{:f}'.format(emotion_prediction[0][5]) + ',' + '{:f}'.format(emotion_prediction[0][6]) + '\n'
            f.write(str_val)
        f.close()

        # Drawing "Emotional Landscape" diagram
        axarr[1].spines['left'].set_position('zero')
        axarr[1].spines['bottom'].set_position('zero')
        axarr[1].spines["top"].set_visible(False)
        axarr[1].spines["right"].set_visible(False)
        axarr[1].set_xlim(-1.0, 1.0)
        axarr[1].set_ylim(-1.0, 1.0)
        axarr[1].grid(False)
        axarr[1].scatter(instant_happiness, instant_angriness, c='#37b5e4', s=10)
        #axarr[1].set_title('Emotional Landscape')
        axarr[1].annotate('fear',
                          xy=(245, 30), xycoords='figure points')
        axarr[1].annotate('angry',
                          xy=(245, 150), xycoords='figure points')
        axarr[1].annotate('happy',
                          xy=(400, 100), xycoords='figure points')
        axarr[1].annotate('sad',
                          xy=(40, 100), xycoords='figure points')

        # Saving diagram
        plt.savefig('/home/lfernandez/spaceai/EmDiagram.png')


        # if len(emotions_collected) < 4:
        #     emotions_collected.append(emotion_prediction)
        #     emotion_text_weighted = ''
        #     emotion_probability_weighted = 0
        #     continue
        # else:
        #     emotions_collected.append(emotion_prediction)
        #     weighted_face_in_time = weigh_up_emotion(emotions_collected)
        #     emotions_collected = []
        #     #emotions_collected.popleft()

            # #selecting maximum emotion along the last 5 frames
            # emotion_probability_weighted = np.max(weighted_face_in_time)
            # emotion_label_arg_weighted = np.argmax(weighted_face_in_time)
            # emotion_text_weighted = emotion_labels[emotion_label_arg_weighted]
            # emotion_window.append(emotion_text_weighted)

            # #keeping maximum instant of happiness
            # if emotion_text_weighted == 'happy' and happy_max < emotion_probability_weighted:
            #     happy_max = emotion_probability_weighted
            #     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite('../images/predicted_test_image.png', bgr_image)

        logger.debug("frame {}, face {}".format(nframe, nface))

        #logger.debug("frame {}, face {}, emotion: {}, probability: {:5.4f} ".format(nframe, nface, emotion_text_weighted, emotion_probability_weighted))


        if len(emotion_window) > frame_window:
                emotion_window.pop(0)
        # try:
        #     emotion_mode = mode(emotion_window)
        # except:
        #     continue

        #Video printing
        #coloring the selected emotion
        # if emotion_text_weighted == 'angry':
        #     color_weighted = emotion_probability_weighted * np.asarray((255, 0, 0))
        # elif emotion_text_weighted == 'sad':
        #     color_weighted = emotion_probability_weighted * np.asarray((0, 0, 255))
        # elif emotion_text_weighted == 'happy':
        #     color_weighted = emotion_probability_weighted * np.asarray((255, 255, 0))
        # elif emotion_text_weighted == 'fear':
        #     color_weighted = emotion_probability_weighted * np.asarray((0, 255, 255))
        # else:
        #     color_weighted = emotion_probability_weighted * np.asarray((0, 255, 0))
        #
        # color_weighted = color_weighted.astype(int)
        # color_weighted = color_weighted.tolist()

        #drawing box and text in the video
        # draw_text(face_coordinates, rgb_image, emotion_text_weighted,
        #           color_weighted, 0, -45, 1, 1)
        # draw_bounding_box(face_coordinates, rgb_image, color_weighted)

        nface += 1

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
