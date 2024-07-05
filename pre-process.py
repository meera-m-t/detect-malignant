import pandas as pd
import cv2
import numpy as np

class HairRemovalAndSegmentation:
    def __init__(self, csv_path, clusters):
        self.csv_path = csv_path
        self.clusters = clusters
        self.data = pd.read_csv(csv_path)
    
    def remove_small_components(self, image, threshold):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = np.zeros((output.shape), dtype=np.uint8)
        component_areas = []
        for i in range(0, nb_components):
            if sizes[i] >= threshold:
                img2[output == i + 1] = 255
                component_areas.append(sizes[i])
        return img2, component_areas
    
    def remove_hair_and_segment(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask1 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 30:
                cv2.drawContours(mask1, [contour], -1, 0, -1)
    
        inverse_mask1 = cv2.bitwise_not(mask1)
        result_image = cv2.bitwise_and(image, image, mask=inverse_mask1)
        
        inpainted_image = cv2.inpaint(result_image, mask1, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        
        gray_inpainted = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
        _, segmented_image = cv2.threshold(gray_inpainted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        segmented_image, component_areas = self.remove_small_components(segmented_image, 500)
        thresh = cv2.adaptiveThreshold(segmented_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)    
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)
    
            scale_factor = 1.5
            radius = int(max(w, h) // 2 * scale_factor)
            

            radius = min(radius, center[0], center[1], image.shape[1] - center[0], image.shape[0] - center[1])
           
            mask = np.zeros_like(gray)
            cv2.circle(mask, center, radius, (255), thickness=-1)
            
            masked_image = cv2.bitwise_and(inpainted_image, inpainted_image, mask=mask)
    
            black_background = np.zeros_like(image)
            black_background[mask == 255] = masked_image[mask == 255]
            
            cv2.imwrite(image_path, black_background)
    
    def process_images(self):
        for index, row in self.data.iterrows():
            if row['Cluster'] in self.clusters:
                self.remove_hair_and_segment(row['Id'])


csv_path = 'datasheets/clustered_data.csv' 
clusters = [0, 1, 2]  

processor = HairRemovalAndSegmentation(csv_path, clusters)
processor.process_images()
