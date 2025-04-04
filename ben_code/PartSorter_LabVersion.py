import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
from collections import Counter

''' ___ '''
class ImageProcessor:
    
    # Color masks to isolate Tetrominos
    lower_orange, upper_orange = np.array([5, 150, 150]), np.array([15, 255, 255])
    lower_green, upper_green = np.array([33, 30, 0]), np.array([167, 150, 100])
    lower_gray, upper_gray = np.array([0, 0, 0]), np.array([200, 75, 90])
    
    #@staticmethod
    #def process_colors(image, contour)
    
    @staticmethod
    def process_image(imgBGR):
    
        if isinstance(imgBGR, str):
            imgBGR = cv2.imread(imgBGR, cv2.IMREAD_COLOR)
            if imgBGR is None:
                print(f"Error: Could not load image {imgBGR}")
                return None
            
        if not isinstance(imgBGR, np.ndarray):
            print(f"Error: Not an image")
            return None
        
        hsv = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        
        lab = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (adaptive histogram equalization) to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge and convert back to BGR
        lab_polarized = cv2.merge((l_enhanced, a, b))
        imgBGR_2 = cv2.cvtColor(lab_polarized, cv2.COLOR_LAB2BGR)
        
        # Combine all color masks
        mask = (
            cv2.inRange(hsv, ImageProcessor.lower_orange, ImageProcessor.upper_orange) |
            cv2.inRange(hsv, ImageProcessor.lower_green, ImageProcessor.upper_green) |
            cv2.inRange(hsv, ImageProcessor.lower_gray, ImageProcessor.upper_gray)
        )
        
        # Apply mask to the original frame (masked result)
        masked = cv2.bitwise_and(imgBGR_2, imgBGR, mask=mask)
        
        grayscale = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Blurring suite
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        for i in range(1):
            blurred = cv2.medianBlur(blurred,5)
            blurred = cv2.blur(blurred, (5, 5))
            blurred = cv2.GaussianBlur(blurred,(5,5),0)
        
        _, binary_img = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = binary_img.shape
        min_area = 500  # Adjust based on your shape size
        max_area = width * height * 0.9  # Ignore contours close to full image size
        
        valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        
        if valid_contours:
            largest_contour = max(valid_contours,  key=lambda cnt: cv2.arcLength(cnt, True))
            epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            hu = ImageProcessor.get_hu_moments(smoothed_contour)
            hist = ImageProcessor.extract_histogram(imgBGR, smoothed_contour)
            colormean = ImageProcessor.extract_average_colour(imgBGR, smoothed_contour)
            return smoothed_contour, hu, hist, colormean # Return the largest contour and binary image
        
        else:
            return None
    
    @staticmethod
    def extract_histogram(img, cntr):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cntr], -1, 255, thickness=cv2.FILLED)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    
    @staticmethod
    def extract_average_colour(img, cntr):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cntr], -1, 255, thickness=cv2.FILLED)
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(img, mask=mask)[:3]  # (H, S, V)
        return mean
    
    @staticmethod
    def get_hu_moments(contour):
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # Normalize
        return hu_moments
        
    @staticmethod
    def show_image(img):
        if img is None:
                print(f"Could not load image: {image_path}")
                return
        display_img = img.copy()
        height, width = display_img.shape[:2]
        max_width, max_height = 900, 1600
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(display_img, new_size)

        cv2.imshow(f"Reference: ", resized)
        cv2.waitKey(0)  # Show for 1 second
        cv2.destroyWindow(f"Reference: ")
    
''' '''
class TetrominoClassifier:
    #
    def __init__(self, reference_images):
        self.reference_moments = self.create_reference_shapes(reference_images)
    #    
    def create_reference_shapes(self, reference_images):
        references = {}
        for shape, image_path in reference_images.items():
            
            #contour = self.load_and_preprocess_image(image_path)
            image_data = ImageProcessor.process_image(image_path)
            
            if image_data is not None:
                
                #print("test")
                #print(len(image_data))
                #contour, hu, hist, ave = image_data
                
                references[shape] = image_data
                
                self.show_reference_image_with_contour(shape, image_path, image_data[0])
                
            else:
                print("empty contour @ create_reference_shapes")
                
        return references

    def classify(self, data, image=None):
        best_match = "Unknown"
        best_score = float("inf")

        test_hist = None
        test_avg_hsv = None
        
        contour, hu_moments, test_hist, test_avg_hsv = data
        
        """
        if image is not None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            test_hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
            cv2.normalize(test_hist, test_hist)
            test_avg_hsv = cv2.mean(hsv, mask=mask)[:3]
        """
        for i, shape in enumerate(self.reference_moments):
            #print(i)
            #print(shape)
            ref_data = self.reference_moments[shape]
            ref_cont, ref_moments, ref_hist, ref_hsv = ref_data
            
            shape_score = np.linalg.norm(hu_moments - ref_moments)

            # Histogram comparison
            color_score = 0
            #if test_hist is not None and shape in self.reference_colors:
            #ref_hist = self.reference_colors[shape]
            color_score = cv2.compareHist(test_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)

            # Average HSV comparison
            hsv_score = 0
            #if test_avg_hsv is not None and shape in self.reference_avg_colors:
            #ref_hsv = self.reference_avg_colors[shape]
            hsv_score = np.linalg.norm(np.array(ref_hsv) - np.array(test_avg_hsv)) / 100.0  # normalize range

            # Combine all scores (tune weights)
            total_score = 0.5 * shape_score  + 2.5 * hsv_score

            if total_score < best_score:
                best_score = total_score
                best_match = shape

        return best_match if best_score < 7.0 else "Unknown"

    def show_reference_image_with_contour(self, shape_name, image_path, contour):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return

        display_img = img.copy()
        cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 3)

        # Resize to fit in 1600x900
        height, width = display_img.shape[:2]
        max_width, max_height = 900, 1600
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(display_img, new_size)

        cv2.imshow(f"Reference: {shape_name}", resized)
        cv2.waitKey(1)  # Show for 1 second
        cv2.destroyWindow(f"Reference: {shape_name}")

class TetrominoDetectionApp:
    def __init__(self, root, reference_images):
        self.root = root
        self.cap = cv2.VideoCapture(0)
        self.contour_history = deque(maxlen=10)
           
        self.classifier = TetrominoClassifier(reference_images) # Build Classifier:

        self.label = tk.Label(root)
        self.label.pack()
        self.quit_btn = tk.Button(root, text="QUIT", command=self.close_app, font=("Arial", 12))
        self.quit_btn.pack()
        self.detect()

    def get_nearest_color_label(self, test_hsv):
        best_color = "UNKNOWN"
        best_dist = float("inf")

        for shape, ref_hsv in self.classifier.reference_avg_colors.items():
            ref_color = shape.split("_")[0]  # GREEN, ORANGE, etc.
            dist = np.linalg.norm(np.array(ref_hsv) - np.array(test_hsv))
            if dist < best_dist:
                best_dist = dist
                best_color = ref_color

        return best_color

        
    def interpolate_contour(self, contour, num_points):
        contour = np.vstack([contour, contour[0]])  # close the loop
        dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        cumulative = np.insert(np.cumsum(dists), 0, 0)
        total_length = cumulative[-1]

        if total_length == 0:
            return None

        uniform_distances = np.linspace(0, total_length, num_points)
        result = []

        i = 0
        for dist in uniform_distances:
            while cumulative[i + 1] < dist:
                i += 1
                if i + 1 >= len(cumulative):
                    break

            ratio = (dist - cumulative[i]) / (cumulative[i + 1] - cumulative[i] + 1e-6)
            pt = (1 - ratio) * contour[i] + ratio * contour[i + 1]
            result.append(pt)

        return np.array(result, dtype=np.float32)
        
        
    def average_contours(self, contours, num_points=64):
        if not contours:
            return None

        resampled_contours = []

        for contour in contours:
            # Flatten if needed
            #contour = contour.squeeze()
            if len(contour.shape) != 2 or len(contour) < 3:
                continue

            # Interpolate contour to exactly num_points
            contour = contour.astype(np.float32)
            arc_len = cv2.arcLength(contour, True)
            if arc_len == 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.01 * arc_len, True)
            approx = approx[:, 0, :]  # Remove extra dimension

            # Uniformly interpolate to num_points
            result = self.interpolate_contour(approx, num_points)
            if result is not None:
                resampled_contours.append(result)

        if not resampled_contours:
            return None

        # Average all resampled contours
        avg = np.mean(resampled_contours, axis=0).astype(np.int32)
        return avg.reshape((-1, 1, 2))  # OpenCV contour format

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            print("ERROR DETECTING CAPTURE")
            return
        #ImageProcessor.show_image(frame)
        frame_data = ImageProcessor.process_image(frame)
        if frame_data is not None:
            contour = frame_data[0]
            self.contour_history.append(frame_data[0])
            avg_contour = self.average_contours(self.contour_history)
            
            if avg_contour is not None:
                contour_to_draw = avg_contour
            else:
                contour_to_draw = contour

            shape_label = self.classifier.classify(frame_data, frame)
            
            ##shape_label = "Unknown"
            full_label = shape_label

            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 255, 0) if shape_label != "Unknown" else (0, 0, 255)

            cv2.drawContours(frame, [contour_to_draw], -1, color, 2)
            cv2.putText(frame, full_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert and display in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=img)
        self.label.config(image=img)
        self.label.image = img
        self.root.after(10, self.detect)

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    reference_images = {
        "GREEN_L": "C:/Users/bwroc/Pictures/Camera Roll/green_L.jpg",
        "GREEN_T": "C:/Users/bwroc/Pictures/Camera Roll/green_T.jpg",
        "GREEN_Z": "C:/Users/bwroc/Pictures/Camera Roll/green_Z.jpg",
        "GREY_L": "C:/Users/bwroc/Pictures/Camera Roll/gray_L.jpg",
        "GREY_T": "C:/Users/bwroc/Pictures/Camera Roll/gray_T.jpg",
        "GREY_Z": "C:/Users/bwroc/Pictures/Camera Roll/gray_Z.jpg",
        "ORANGE_L": "C:/Users/bwroc/Pictures/Camera Roll/orange_L.jpg",
        "ORANGE_T": "C:/Users/bwroc/Pictures/Camera Roll/orange_T.jpg",
        "ORANGE_Z": "C:/Users/bwroc/Pictures/Camera Roll/orange_Z.jpg"
    }
    root = tk.Tk()
    root.title("Tetromino Detection")
    app = TetrominoDetectionApp(root, reference_images)
    root.mainloop()

