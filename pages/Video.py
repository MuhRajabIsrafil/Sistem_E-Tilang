import os
import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from functions import function_system
from stqdm import stqdm
from time import sleep


def load_model(model_path):
    model = YOLO(model_path)
    return model


def first_container(container_input, mydb, mycursor):
    container_input.empty()
    container = container_input.container(border=1)

    address = container.text_area("Address:", max_chars=100, placeholder="Input Address")
    empty_address = container.empty()

    col1, col2 = container.columns(2)

    with col1:
        start_date = st.date_input("Enter Start Date", value=None)
        empty_date = st.empty()

    with col2:
        start_time = st.time_input("Enter Start Time", value=None)
        empty_time = st.empty()

    video_file = container.file_uploader("Upload Video", type=["mp4", "mkv", "avi", "mov", "wmv", "flv", "webm"])
    empty_video = container.empty()

    ori_video_path = "../original_video"

    if video_file is not None:
        video_bytes = video_file.name
        video_upload = open(os.path.join(ori_video_path, video_bytes), mode="wb")

        with video_upload as f:
            f.write(video_file.read())

    generate_btn = container.button(label="Generate")

    if generate_btn:
        if address != "" and start_date is not None and start_time is not None and video_file is not None:
            model_path = "../model/yolov8_model/yolov8_model.pt"
            model = load_model(model_path)

            cap = cv2.VideoCapture(ori_video_path + "/" + video_file.name)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            width_dimension = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_dimension = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            sec = round(frames / fps)

            c_datestamp = datetime.combine(start_date, start_time)

            given_time = c_datestamp.strftime("%d/%m/%Y %H:%M:%S")

            start_time = c_datestamp.strftime("%d-%m-%Y %H_%M_%S")
            convert_time = c_datestamp + pd.DateOffset(seconds=sec)
            end_time = convert_time.strftime("%d-%m-%Y %H_%M_%S")

            make_folder = start_time + " - " + end_time

            if not os.path.join("../results/" + make_folder):
                function_system.MakeFolder(make_folder)
            else:
                folders = os.listdir("../results")
                folders_name = [item for item in folders if os.path.isdir(os.path.join("../results", item))]

                folders_array = []
                folder_amount_array = []
                folder_count = 0
                for folder in folders_name:
                    if make_folder in folder:
                        folder_count += 1
                        folders_array.append(folder)
                        folder_amount_array.append("(" + str(folder_count) + ")")

                for f in folder_amount_array:
                    check_folder = make_folder + " " + f
                    if check_folder not in folders_array:
                        make_folder = check_folder
                        break

                function_system.MakeFolder(make_folder)

            finish_generate = False

            second_container(container_input, cap, model, frames, fps, address, given_time, make_folder, video_file,
                             finish_generate, width_dimension, height_dimension, mydb, mycursor)
        else:
            if address == "":
                empty_address.error("Please Input an Address")

            if start_date is None:
                empty_date.error("Please Insert a Date")

            if start_time is None:
                empty_time.error("Please Insert a Time")

            if video_file is None:
                empty_video.error("Please Insert a Video")


def second_container(container_input, cap, model, frames, fps, address, given_time, make_folder, video_file,
                     finish_generate, width_dimension, height_dimension, mydb, mycursor):
    container_input.empty()
    container = st.empty()
    generate_container = container.container(border=1)

    generate_container.subheader("Original Video")

    generate_container.video(video_file)

    count_image = 0
    time = given_time

    with generate_container:
        for step in stqdm(range(frames), desc="Processing Video"):
            sleep(0.01)

            cap.set(cv2.CAP_PROP_POS_FRAMES, step)
            success, frame = cap.read()

            if success:
                detect_object = model.predict(frame)

                if width_dimension == 3840 and height_dimension == 2160:  # 4k/2160p
                    annotated_frame = function_system.plot_bboxes(frame, detect_object[0].boxes.data, height=950)

                    cv2.putText(annotated_frame[0], "Address: " + address, (10, 75), 5, 3,
                                (32, 38, 53), 3, cv2.LINE_AA)

                    if step % fps == 0:
                        time = function_system.get_time(time)
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 3,
                                    (32, 38, 53), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 3,
                                    (32, 38, 53), 3, cv2.LINE_AA)
                elif width_dimension == 1920 and height_dimension == 1080:  # FHD/1080p
                    annotated_frame = function_system.plot_bboxes(frame, detect_object[0].boxes.data, height=300)

                    cv2.putText(annotated_frame[0], "Address: " + address, (10, 75), 5, 2,
                                (32, 38, 53), 3, cv2.LINE_AA)

                    if step % fps == 0:
                        time = function_system.get_time(time)
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 2,
                                    (32, 38, 53), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 2,
                                    (32, 38, 53), 3, cv2.LINE_AA)
                else:
                    annotated_frame = function_system.plot_bboxes(frame, detect_object[0].boxes.data)

                    cv2.putText(annotated_frame[0], "Address: " + address, (10, 75), 5, 2,
                                (32, 38, 53), 3, cv2.LINE_AA)

                    if step % fps == 0:
                        time = function_system.get_time(time)
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 2,
                                    (32, 38, 53), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_frame[0], "Time: " + time, (10, 150), 5, 2,
                                    (32, 38, 53), 3, cv2.LINE_AA)

                if annotated_frame[1]:
                    for m in annotated_frame[1]:
                        bb_motorcycle = [int(m[0]), int(m[1]), int(m[2]), int(m[3])]

                        crop_bb_no_helmet = function_system.get_middle_coordinates(bb_motorcycle, classes="no_helmet")
                        crop_bb_plate = function_system.get_middle_coordinates(bb_motorcycle, classes="plate")

                        if annotated_frame[2]:
                            for nh in annotated_frame[2]:
                                bb_no_helmet = [int(nh[0]), int(nh[1]), int(nh[2]), int(nh[3])]

                                if crop_bb_no_helmet[0] < bb_no_helmet[0] < bb_no_helmet[2] < crop_bb_no_helmet[2] and \
                                        crop_bb_no_helmet[1] < bb_no_helmet[1] < bb_no_helmet[3] < crop_bb_no_helmet[3]:
                                    crop_motorcycle = annotated_frame[0][bb_motorcycle[1]:bb_motorcycle[3],
                                                      bb_motorcycle[0]:bb_motorcycle[2]]

                                    if annotated_frame[3]:
                                        plate_detected_array = []

                                        for p in annotated_frame[3]:
                                            plate = [int(p[0]), int(p[1]), int(p[2]), int(p[3])]

                                            if crop_bb_plate[0] < plate[0] < plate[2] < crop_bb_plate[2] and \
                                                    crop_bb_plate[
                                                        1] < plate[1] < plate[3] < crop_bb_plate[3]:
                                                plate_detected_array.append(
                                                    True)  # Plate inside the bounding box of motorcycle
                                            else:
                                                plate_detected_array.append(
                                                    False)  # Plate outside the bounding box of motorcycle

                                        if True in plate_detected_array:
                                            resize_plate = []

                                            for plate_count, plate_value in enumerate(annotated_frame[3]):
                                                if plate_detected_array[plate_count]:
                                                    bb_plate = [int(plate_value[0]), int(plate_value[1]),
                                                                int(plate_value[2]), int(plate_value[3])]

                                                    crop_plate = annotated_frame[0][bb_plate[1]:bb_plate[3],
                                                                 bb_plate[0]:bb_plate[2]]

                                                    new_width = int(crop_plate.shape[1] * 5)
                                                    new_height = int(crop_plate.shape[0] * 5)

                                                    resize_plate = cv2.resize(crop_plate, (new_width, new_height))

                                            merge_frame = function_system.get_concat_v_resize(crop_motorcycle,
                                                                                              resize_plate)
                                            merge_frame_2 = function_system.get_concat_h_resize(annotated_frame[0],
                                                                                                merge_frame)
                                        else:
                                            plate_not_detected = "Plate\nNot Detected"

                                            blank_plate = function_system.Make_BG([500, 600, 3])
                                            blank_plate.fill(0)

                                            y_start = int(blank_plate.shape[0] / 2 - 50)
                                            x_start = int(blank_plate.shape[1])
                                            y_increment = 100

                                            for i, line in enumerate(plate_not_detected.split('\n')):
                                                w_, _ = cv2.getTextSize(line, 5, 3, 3)[0]

                                                y = y_start + i * y_increment
                                                cv2.putText(blank_plate, line, (int((x_start - w_) / 2), y), 5, 3,
                                                            (255, 255, 255), 3, cv2.LINE_AA)

                                            merge_frame = function_system.get_concat_v_resize(crop_motorcycle,
                                                                                              blank_plate)
                                            merge_frame_2 = function_system.get_concat_h_resize(annotated_frame[0],
                                                                                                merge_frame)

                                        directory = '../results/' + make_folder
                                        filename = 'results_images_' + str((count_image + 1)) + '.jpg'

                                        cv2.imwrite(os.path.join(directory, filename), merge_frame_2)

                                        sql = "insert into results(address,datetime,image) values(%s,%s,%s)"
                                        val = (address, make_folder, filename)
                                        mycursor.execute(sql, val)
                                        mydb.commit()

                                        count_image += 1
            else:
                break

            if step == frames - 1:
                finish_generate = True

        cap.release()
        cv2.destroyAllWindows()

        if finish_generate:
            st.success("Successfully Generate")
            image_dir = f'../results/{make_folder}'

            sql_file = f'Select image from results where datetime = "{make_folder}"'
            mycursor.execute(sql_file)

            files = mycursor.fetchall()
            fix_files = function_system.fix_array(files)

            images_path_array = []
            for file in fix_files:
                images_path_array.append(image_dir + '/' + file)

            st.subheader("Results")

            num_columns = 5
            cols_image = st.columns(num_columns)

            if len(images_path_array) == 0:
                no_img_html = """
                <div style="text-align: center"> No Image Result In This Folder </div>
                """

                st.markdown(no_img_html, unsafe_allow_html=True)
            else:
                for i, image_path in enumerate(images_path_array):
                    with cols_image[i % num_columns]:
                        st.image(image_path, caption=f'Result Image {i + 1}', use_column_width=True)

            back_container = generate_container.button("Generate More")
            if back_container:
                first_container(container)


def app(mydb, mycursor):
    st.title("Generate Video 📹")
    st.markdown(
        """
            Plays a stored video file. Tracks and detects objects using the **YOLOv8** object detection model.
        """
    )
    st.markdown("<p style='padding-top:20px'></p>", unsafe_allow_html=True)

    container_placeholder = st.empty()
    first_container(container_placeholder, mydb, mycursor)
