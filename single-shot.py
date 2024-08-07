# This script is used to look for objects under a specific condition (at least 5 persons etc)
# The script reads a video from a message queue, classifies the objects in the video, and does a condition check.
# If condition is met, the video is being forwarded to a remote vault.

# Local imports
from utils.VariableClass import VariableClass
from utils.ClassificationObject import ClassificationObject

# External imports
import os
import cv2
import time
import torch
from ultralytics import YOLO

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()

if var.TIME_VERBOSE:
    start_time = time.time()
    total_time_preprocessing = 0
    total_time_class_prediction = 0
    total_time_color_prediction = 0
    total_time_processing = 0
    total_time_postprocessing = 0
    start_time_preprocessing = time.time()

# Perform object classification on the media
# initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = YOLO(var.MODEL_NAME).to(device)
if var.LOGGING:
    print(f'3) Using device: {device}')

# Open video-capture/recording using the video-path. Throw FileNotFoundError if cap is unable to open.
if var.LOGGING:
    print(f'4) Opening video file: {var.MEDIA_SAVEPATH}')
cap = cv2.VideoCapture(var.MEDIA_SAVEPATH)
if not cap.isOpened():
    FileNotFoundError('Unable to open video file')

# Initialize the video-writer if the SAVE_VIDEO is set to True.
if var.SAVE_VIDEO:
    fourcc = cv2.VideoWriter.fourcc(*'avc1')
    video_out = cv2.VideoWriter(
        filename=var.OUTPUT_MEDIA_SAVEPATH,
        fourcc=fourcc,
        fps=var.CLASSIFICATION_FPS,
        frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

# Initialize the classification process.
# 2 lists are initialized:
    # Classification objects
    # Additional list for easy access to the ids.
classification_object_list: list[ClassificationObject] = []
classification_object_ids: list[int] = []

# frame_number -> The current frame number. Depending on the frame_skip_factor this can make jumps.
# predicted_frames -> The number of frames, that were used for the prediction. This goes up by one each prediction iteration.
# frame_skip_factor is the factor by which the input video frames are skipped.
frame_number, predicted_frames = 0, 0
frame_skip_factor = int(cap.get(cv2.CAP_PROP_FPS) / var.CLASSIFICATION_FPS)

# Loop over the video frames, and perform object classification.
# The classification process is done until the counter reaches the MAX_NUMBER_OF_PREDICTIONS or the last frame is reached.
MAX_FRAME_NUMBER = cap.get(cv2.CAP_PROP_FRAME_COUNT)
if var.LOGGING:
    print(f'5) Classifying frames')
if var.TIME_VERBOSE:
    total_time_preprocessing += time.time() - start_time_preprocessing
    start_time_processing = time.time()

while (predicted_frames < var.MAX_NUMBER_OF_PREDICTIONS) and (frame_number < MAX_FRAME_NUMBER):

    # Read the frame from the video-capture.
    success, frame = cap.read()
    if not success:
        break

    # Keep the first frame in memory, if the CREATE_BBOX_FRAME is set to True.
    # This is used to draw the tracking results on.
    if var.CREATE_BBOX_FRAME and frame_number == 0:
        bbox_frame = frame.copy()

    # Check if the frame_number corresponds to a frame that should be classified.
    if frame_number > 0 and frame_skip_factor > 0 and frame_number % frame_skip_factor == 0:

        # Perform object classification on the frame.
        # persist=True -> The tracking results are stored in the model.
        # persist should be kept True, as this provides unique IDs for each detection.
        # More information about the tracking results via https://docs.ultralytics.com/reference/engine/results/
        if var.TIME_VERBOSE:
            start_time_class_prediction = time.time()
        results = MODEL.track(
            source=frame,
            persist=True,
            verbose=False,
            conf=var.CLASSIFICATION_THRESHOLD)
        if var.TIME_VERBOSE:
            total_time_class_prediction += time.time() - start_time_class_prediction

        # ###############################################
        # This is where the custom logic comes into play
        # ###############################################
        # Check if the results are not None,
        #  Otherwise, the postprocessing should not be done.
        # Iterate over the detected objects and their masks.

        annotated_frame = frame.copy()
        if results is not None:
            # Loop over boxes and masks.
            # If no masks are found, meaning the model used is not a segmentation model, the mask is set to None.
            for box, mask in zip(results[0].boxes, results[0].masks or [None] * len(results[0].boxes)):

                # Check if object are detected.
                # If no object is detected, the box.id will be None.
                # In this case, the inner-loop is broken. Not calling the object related functions.
                if box.id is None:
                    break

                # Annotate the frame with the classification objects.
                # Draw the class name and the confidence on the frame.

                cv2.rectangle(
                    img=annotated_frame,
                    pt1=(int(box.xyxy.tolist()[0][0]), int(
                        box.xyxy.tolist()[0][1])),
                    pt2=(int(box.xyxy.tolist()[0][2]), int(
                        box.xyxy.tolist()[0][3])),
                    color=(0, 255, 0),
                    thickness=2)

        # Depending on the SAVE_VIDEO or PLOT parameter, the frame is annotated.
        # This is done using a custom annotation function.
        if var.SAVE_VIDEO or var.PLOT:

            # Show the annotated frame if the PLOT parameter is set to True.
            cv2.imshow("YOLOv8 Tracking",
                       annotated_frame) if var.PLOT else None
            cv2.waitKey(1) if var.PLOT else None

            # Write the annotated frame to the video-writer if the SAVE_VIDEO parameter is set to True.
            video_out.write(
                annotated_frame) if var.SAVE_VIDEO else None

        # Increase the frame_number and predicted_frames by one.
        predicted_frames += 1
    frame_number += 1

if var.TIME_VERBOSE:
    total_time_processing += time.time() - start_time_processing
    start_time_postprocessing = time.time()

# Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
if var.TIME_VERBOSE:
    print(
        f'\t - Classification took: {round(time.time() - start_time, 1)} seconds, @ {var.CLASSIFICATION_FPS} fps.')
    print(
        f'\t\t - {round(total_time_preprocessing, 2)}s for preprocessing and initialisation')
    print(
        f'\t\t - {round(total_time_processing, 2)}s for processing of which:')
    print(
        f'\t\t\t - {round(total_time_class_prediction, 2)}s for class prediction')
    print(
        f'\t\t\t - {round(total_time_processing - total_time_class_prediction - total_time_color_prediction, 2)}s for other processing')
    print(
        f'\t\t - {round(total_time_postprocessing, 2)}s for postprocessing')
    print(f'\t - Original video: {round(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS), 1)} seconds, @ {round(cap.get(cv2.CAP_PROP_FPS), 1)} fps @ {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. File size of {round(os.path.getsize(var.MEDIA_SAVEPATH)/1024**2, 1)} MB')

# If the videowriter was active, the videowriter is released.
# Close the video-capture and destroy all windows.
if var.LOGGING:
    print('8) Releasing video writer and closing video capture')
    print("\n\n")

video_out.release() if var.SAVE_VIDEO else None
cap.release()
cv2.destroyAllWindows()