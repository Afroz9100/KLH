if __name__ == "__main__":
    faces = encode_faces()
    encoded_faces = list(faces.values())
    faces_name = list(faces.keys())
    video_frame = True

    # Initialize and start multi-thread video input
    # stream from the WebCam.
    # 0 refers to the default WebCam
    video_stream = VideoStream(stream=0)
    video_stream.start()

    while True:
        if video_stream.stopped is True:
            break
        else :
            frame = video_stream.read()

            if video_frame:
                face_locations = fr.face_locations(frame)
                unknown_face_encodings = fr.face_encodings(frame, \
                face_locations)

                face_names = []
                for face_encoding in unknown_face_encodings:
                    # Comapring the faces
                    matches = fr.compare_faces(encoded_faces, \
                    face_encoding)
                    name = "Unknown"

                    face_distances = fr.face_distance(encoded_faces,\
                    face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = faces_name[best_match_index]

                    face_names.append(name)

            video_frame = not video_frame

            for (top, right, bottom, left), name in zip(face_locations,\
            face_names):
                # Draw a rectangular box around the face
                cv2.rectangle(frame, (left-20, top-20), (right+20, \
                bottom+20), (0, 255, 0), 2)
                # Draw a Label for showing the name of the person
                cv2.rectangle(frame, (left-20, bottom -15), \
                (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                # Showing the name of the detected person through 
                # the WebCam
                cv2.putText(frame, name, (left -20, bottom + 15), \
                font, 0.85, (255, 255, 255), 2)
                
                # Call the function for attendance
                Attendance(name)

        # delay for processing a frame 
        delay = 0.04
        time.sleep(delay)

        cv2.imshow('frame' , frame)
        key = cv2.waitKey(1)
        # Press 'q' for stop the executing of the program
        if key == ord('q'):
            break

    video_stream.stop()

    # closing all windows 
    cv2.destroyAllWindows()
