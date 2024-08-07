
from utils.TranslateObject import translate
from utils.VariableClass import VariableClass
import cv2
import time
import re

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


# Human-readable conditions e.g.
# - 5 persons detected
# - 3 cars detected
# - 2 trucks detected

def condition_met(results, text=""):

    count_persons = 0
    count_cars = 0
    count_trucks = 0

    # Loop over boxes and masks.
    # If no masks are found, meaning the model used is not a segmentation model, the mask is set to None.
    for box, mask in zip(results[0].boxes, results[0].masks or [None] * len(results[0].boxes)):

        # Check if object are detected.
        # If no object is detected, the box.id will be None.
        # In this case, the inner-loop is broken. Not calling the object related functions.
        if box.id is None:
            break

        object_name = translate(results[0].names[int(box.cls)])

        if object_name == 'car':
            count_cars += 1
        elif object_name == 'pedestrian':
            count_persons += 1
        elif object_name == 'truck':
            count_trucks += 1

    print(
        f"Persons: {count_persons}, Cars: {count_cars}, Trucks: {count_trucks}")

    counts = {
        "persons": count_persons,
        "cars": count_cars,
        "trucks": count_trucks
    }

    match = re.match(r"(\d+) (\w+) detected", text)
    if match:
        number = int(match.group(1))
        object_type = match.group(2)
        if object_type in counts:
            return counts[object_type] >= number
    return False

# Function to process the frame.


def processFrame(MODEL, frame, video_out, condition):
    # Perform object classification on the frame.
    # persist=True -> The tracking results are stored in the model.
    # persist should be kept True, as this provides unique IDs for each detection.
    # More information about the tracking results via https://docs.ultralytics.com/reference/engine/results/

    total_time_class_prediction = 0
    if var.TIME_VERBOSE:
        start_time_class_prediction = time.time()

    # Execute the model
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

        # We can set a condition...
        # In this case, we are looking for at least 5 persons.
        # If the condition is met, the video is being forwarded to a remote vault.

        is_condition_met = condition_met(
            results, condition)

        if is_condition_met:
            print("Condition met, forwarding video to remote vault")
            return frame, total_time_class_prediction, True
        else:
            print("Condition not met, not forwarding video to remote vault")

        # Annotate the frame with the classification objects.
        # Draw the class name and the confidence on the frame.
        if var.SAVE_VIDEO or var.PLOT:
            for box, mask in zip(results[0].boxes, results[0].masks or [None] * len(results[0].boxes)):
                # Translate the class name to a human-readable format and display it on the frame.
                object_name = translate(results[0].names[int(box.cls)])
                cv2.putText(
                    img=annotated_frame,
                    text=object_name,
                    org=(int(box.xyxy.tolist()[0][0]), int(
                        box.xyxy.tolist()[0][1]) - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2)

                # Draw the bounding box on the frame.
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

    return frame, total_time_class_prediction, False
