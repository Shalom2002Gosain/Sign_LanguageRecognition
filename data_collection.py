import cv2
import os

# Predefined gesture labels
labels = ["Hello Sir", "I love you", "Okay", "Help", "Yes"]

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"\nCollecting images for gesture: {label}")
    save_path = f"data/{label}"
    os.makedirs(save_path, exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {label} | Saved: {count}/50", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' to Save | 'n' to Next | 'q' to Quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(1)
        if key == ord('s'):
            if count < 50:
                cv2.imwrite(f"{save_path}/{count}.jpg", roi)
                count += 1
                print(f"Saved {count} images for {label}")
            else:
                print(f"Already collected 50 images for {label}")
        elif key == ord('n') or count >= 50:
            print(f"Finished collecting for: {label}")
            break
        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
