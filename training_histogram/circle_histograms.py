import os
import cv2
import numpy as np
from glob import glob
import json


# Configuration
CLASS_MAP = {0: 'red', 1: 'white', 2: 'yellow'}
HIST_2D_BINS = [30, 32]           # H, S bins for 2D histogram
HIST_1D_BINS = [30, 32, 32]       # Bins for H, S, V 1D histograms
RANGES_2D = [0, 180, 0, 256]      # H ∈ [0,180], S ∈ [0,256]
RANGES_1D = [0, 180, 0, 256, 0, 256]  # H ∈ [0,180], S ∈ [0,256], V ∈ [0,256]


class HistogramCollector:
    def __init__(self):
        # Initialize empty lists for each color
        self.data = {
            'red':    {'hs2d': [], 'h1d': [], 's1d': [], 'v1d': []},
            'white':  {'hs2d': [], 'h1d': [], 's1d': [], 'v1d': []},
            'yellow': {'hs2d': [], 'h1d': [], 's1d': [], 'v1d': []}
        }
        self.histogram_counts = {'red': 0, 'white': 0, 'yellow': 0}

    def calculate_histograms(self, roi_hsv, mask=None):
        """Calculate and normalize histograms for an HSV ROI, using an optional mask."""
        # 2D HS histogram
        hist_2d = cv2.calcHist([roi_hsv], [0, 1], mask, HIST_2D_BINS, RANGES_2D)
        hist_2d = cv2.normalize(hist_2d, None, alpha=1, norm_type=cv2.NORM_L1)

        # 1D H histogram
        hist_h = cv2.calcHist([roi_hsv], [0], mask, [HIST_1D_BINS[0]], [0, 180])
        hist_h = cv2.normalize(hist_h, None, alpha=1, norm_type=cv2.NORM_L1)

        # 1D S histogram
        hist_s = cv2.calcHist([roi_hsv], [1], mask, [HIST_1D_BINS[1]], [0, 256])
        hist_s = cv2.normalize(hist_s, None, alpha=1, norm_type=cv2.NORM_L1)

        # 1D V histogram
        hist_v = cv2.calcHist([roi_hsv], [2], mask, [HIST_1D_BINS[2]], [0, 256])
        hist_v = cv2.normalize(hist_v, None, alpha=1, norm_type=cv2.NORM_L1)

        return hist_2d, hist_h, hist_s, hist_v

    def process_image(self, img_path, label_path):
        """Process a single image and its corresponding YOLO-format label file."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: Label file {label_path} not found")
            return

        for line in lines:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id not in CLASS_MAP:
                continue

            color = CLASS_MAP[class_id]
            self.histogram_counts[color] += 1

            # Convert YOLO-format (x_center, y_center, width, height) → pixel coords
            xc, yc, bw, bh = np.array(parts[1:]) * [w, h, w, h]
            x1 = max(0, int(xc - bw / 2))
            y1 = max(0, int(yc - bh / 2))
            x2 = min(w, x1 + int(bw))
            y2 = min(h, y1 + int(bh))

            if x2 <= x1 or y2 <= y1:
                continue

            roi_bgr = img[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                continue

            # Create a circular mask inside the bounding box
            roi_h, roi_w = roi_hsv.shape[:2]
            center = (roi_w // 2, roi_h // 2)
            radius = min(center)  # largest circle that fits inside the box
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            # Visualize the bounding box + circle on the original image
            debug_img = img.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # Draw the circle on the ROI region in the debug image
            cv2.circle(debug_img[y1:y2, x1:x2], center, radius, (0, 255, 0), 1)

            cv2.imshow("ROI with Circle", debug_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

            try:
                hs2d, h1d, s1d, v1d = self.calculate_histograms(roi_hsv, mask)

                self.data[color]['hs2d'].append(hs2d.tolist())
                self.data[color]['h1d'].append(h1d.flatten().tolist())
                self.data[color]['s1d'].append(s1d.flatten().tolist())
                self.data[color]['v1d'].append(v1d.flatten().tolist())
            except Exception as e:
                print(f"Error computing histogram for {img_path}: {str(e)}")
                continue

    def save_color_histograms(self, output_dir):
        """Save separate JSON files for each color, containing mean histograms."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Common metadata for all files
        metadata = {
            'hist_2d_bins': HIST_2D_BINS,
            'hist_1d_bins': HIST_1D_BINS,
            'ranges_2d':    RANGES_2D,
            'ranges_1d':    RANGES_1D
        }

        for color in self.data:
            if not self.data[color]['hs2d']:
                print(f"Warning: No samples found for color {color}")
                continue

            # Compute the mean histogram across all collected samples
            mean_hs2d = np.array(self.data[color]['hs2d']).mean(axis=0).tolist()
            mean_h1d  = np.array(self.data[color]['h1d']).mean(axis=0).tolist()
            mean_s1d  = np.array(self.data[color]['s1d']).mean(axis=0).tolist()
            mean_v1d  = np.array(self.data[color]['v1d']).mean(axis=0).tolist()

            color_data = {
                'metadata':     metadata.copy(),
                'sample_count': self.histogram_counts[color],
                'histograms': {
                    'hs2d': mean_hs2d,
                    'h1d':  mean_h1d,
                    's1d':  mean_s1d,
                    'v1d':  mean_v1d
                }
            }

            output_path = os.path.join(output_dir, f"{color}_histograms.json")
            with open(output_path, 'w') as f:
                json.dump(color_data, f, indent=4)
            print(f"Saved {color} histogram data to {output_path}")

        print(f"\nSample counts per color: {self.histogram_counts}")


if __name__ == "__main__":
    collector = HistogramCollector()
    image_dir = "/home/asmar/RAMI/RAMI_ROS1_ws/src/training_histogram/buoys with colors.v1i.yolov8/train/images"
    output_dir = "color_histograms"

    # Iterate over all .jpg images, process each along with its .txt label file
    for img_path in glob(os.path.join(image_dir, "*.jpg")):
        base = os.path.splitext(img_path)[0]
        label_path = f"{base}.txt"
        if os.path.exists(label_path):
            collector.process_image(img_path, label_path)

    # Save the averaged histograms for each color to JSON
    collector.save_color_histograms(output_dir)
    print(f"\nHistogram collection complete. Results saved to {output_dir}/ directory")
    cv2.destroyAllWindows()
