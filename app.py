import streamlit as st
from PIL import Image
import torch
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from io import BytesIO
import base64
import cv2

# Import custom modules
from model import load_model, predict

# Set device
device = torch.device("cpu")

# Page configuration
st.set_page_config(
    page_title="Vision Health Suite",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .download-button-container {
        margin-top: 1rem;
    }
    .sidebar-info {
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4da6ff;
    }
    .test-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .difficulty-easy {
        border-left: 5px solid #28a745;
    }
    .difficulty-medium {
        border-left: 5px solid #fd7e14;
    }
    .difficulty-hard {
        border-left: 5px solid #dc3545;
    }
    .snellen-row {
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 10px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 2px;
    }
    .grid-item {
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .amsler-grid {
        width: 100%;
        aspect-ratio: 1;
        background-image: repeating-linear-gradient(#000 0 1px, transparent 1px 30px),
                         repeating-linear-gradient(90deg, #000 0 1px, transparent 1px 30px);
        background-size: 100% 100%;
    }
    .astigmatism-pattern {
        width: 80%;
        margin: 0 auto;
    }
    .dot {
        height: 20px;
        width: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .contrast-text {
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Color Vision Test Functions
def create_ishihara_plate(number, plate_type="red-green", difficulty="medium"):
    """Generate a simple Ishihara-style color blindness test plate"""
    # Set up plate dimensions
    width, height = 400, 400
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2 - 20
    
    # Create blank white image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Generate random colored dots across the entire plate
    num_dots = 1000
    dot_radius_range = (5, 15)
    
    # Set colors based on color blindness type and difficulty
    if plate_type == "red-green":
        if difficulty == "easy":
            bg_colors = [(200, 150, 150), (150, 200, 150)]  # Reddish and greenish backgrounds
            number_colors = [(100, 0, 0), (0, 100, 0)]      # More distinct red and green
        elif difficulty == "medium":
            bg_colors = [(180, 160, 160), (160, 180, 160)]  # Less contrast
            number_colors = [(130, 40, 40), (40, 130, 40)]  # Less distinct
        else:  # hard
            bg_colors = [(170, 165, 165), (165, 170, 165)]  # Very subtle difference
            number_colors = [(150, 80, 80), (80, 150, 80)]  # Very challenging
    
    elif plate_type == "blue-yellow":
        if difficulty == "easy":
            bg_colors = [(150, 150, 200), (200, 200, 150)]  # Bluish and yellowish backgrounds
            number_colors = [(0, 0, 100), (100, 100, 0)]    # More distinct blue and yellow
        elif difficulty == "medium":
            bg_colors = [(160, 160, 190), (190, 190, 160)]  # Less contrast
            number_colors = [(40, 40, 130), (130, 130, 40)]  # Less distinct
        else:  # hard
            bg_colors = [(165, 165, 180), (180, 180, 165)]  # Very subtle difference
            number_colors = [(80, 80, 150), (150, 150, 80)]  # Very challenging
    
    elif plate_type == "tritan":
        if difficulty == "easy":
            bg_colors = [(150, 180, 200), (200, 180, 150)]  # Blue-ish and yellow-ish
            number_colors = [(50, 100, 150), (150, 100, 50)]  # More distinct
        elif difficulty == "medium":
            bg_colors = [(160, 175, 190), (190, 175, 160)]  # Less contrast
            number_colors = [(80, 110, 140), (140, 110, 80)]  # Less distinct
        else:  # hard
            bg_colors = [(165, 170, 180), (180, 170, 165)]  # Very subtle difference
            number_colors = [(100, 120, 140), (140, 120, 100)]  # Very challenging
    
    elif plate_type == "monochromacy":
        if difficulty == "easy":
            bg_colors = [(150, 150, 150), (200, 200, 200)]  # Grayscale contrast
            number_colors = [(50, 50, 50), (230, 230, 230)]  # High contrast
        elif difficulty == "medium":
            bg_colors = [(160, 160, 160), (190, 190, 190)]  # Medium contrast
            number_colors = [(80, 80, 80), (210, 210, 210)]  # Medium contrast
        else:  # hard
            bg_colors = [(165, 165, 165), (180, 180, 180)]  # Low contrast
            number_colors = [(140, 140, 140), (200, 200, 200)]  # Low contrast
    
    else:  # general color test
        if difficulty == "easy":
            bg_colors = [(200, 150, 200), (150, 200, 150)]  # Purple and green
            number_colors = [(100, 0, 100), (0, 100, 0)]    # More distinct
        elif difficulty == "medium":
            bg_colors = [(180, 160, 180), (160, 180, 160)]  # Less contrast
            number_colors = [(130, 40, 130), (40, 130, 40)]  # Less distinct
        else:  # hard
            bg_colors = [(170, 165, 170), (165, 170, 165)]  # Very subtle difference
            number_colors = [(150, 80, 150), (80, 150, 80)]  # Very challenging
            
    # Create background dots
    for _ in range(num_dots):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        
        # Calculate distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Only place dots within the circle
        if dist <= radius:
            dot_radius = random.randint(*dot_radius_range)
            color = random.choice(bg_colors)
            
            # Draw a circle at (x,y)
            for dx in range(-dot_radius, dot_radius+1):
                for dy in range(-dot_radius, dot_radius+1):
                    if dx**2 + dy**2 <= dot_radius**2:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and np.sqrt((nx - center_x)**2 + (ny - center_y)**2) <= radius:
                            img[ny, nx] = color
    
    # Define the number shape - simple digit patterns
    patterns = {
        '0': [(0, -3), (1, -3), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
              (1, 3), (0, 3), (-1, 3), (-2, 3), (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
              (-2, -3), (-1, -3)],
        '1': [(0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (-1, -2)],
        '2': [(1, -3), (2, -3), (2, -2), (2, -1), (1, 0), (0, 0), (-1, 0), (-2, 1), 
              (-2, 2), (-2, 3), (-1, 3), (0, 3), (1, 3), (2, 3)],
        '3': [(-2, -3), (-1, -3), (0, -3), (1, -3), (1, -2), (1, -1), (0, 0), (1, 1), 
              (1, 2), (1, 3), (0, 3), (-1, 3), (-2, 3)],
        '4': [(1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), 
              (0, 0), (-1, 0), (-2, 0), (0, -1), (0, -2), (0, -3)],
        '5': [(2, -3), (1, -3), (0, -3), (-1, -3), (-2, -3), (-2, -2), (-2, -1), 
              (-1, -1), (0, -1), (1, -1), (2, 0), (2, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 3)],
        '6': [(1, -3), (0, -3), (-1, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), 
              (-1, 3), (0, 3), (1, 3), (2, 2), (2, 1), (1, 0), (0, 0), (-1, 0)],
        '7': [(2, -3), (1, -3), (0, -3), (-1, -3), (-2, -3), (-1, -2), (0, -1), 
              (1, 0), (1, 1), (1, 2), (1, 3)],
        '8': [(0, -3), (1, -3), (2, -2), (2, -1), (1, 0), (0, 0), (-1, 0), 
              (-2, -1), (-2, -2), (-1, -3), (0, -3), (1, 0), (2, 1), (2, 2), 
              (1, 3), (0, 3), (-1, 3), (-2, 2), (-2, 1), (-1, 0)],
        '9': [(1, 0), (0, 0), (-1, 0), (-2, -1), (-2, -2), (-1, -3), (0, -3), 
              (1, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (1, 3), (0, 3), (-1, 3)],
    }
    
    # Get the pattern for the number
    number_str = str(number)
    
    # Adjust scale factor based on number of digits
    scale_factor = 35 if len(number_str) == 1 else 25
    x_offset = -15 * (len(number_str) - 1)
    
    for digit_idx, digit in enumerate(number_str):
        digit_offset_x = x_offset + 30 * digit_idx
        
        for x, y in patterns[digit]:
            # Scale and position the pattern
            pos_x = center_x + (x * scale_factor) + digit_offset_x
            pos_y = center_y + (y * scale_factor)
            
            # Draw dots for this position
            for _ in range(5):  # Multiple dots to ensure visibility
                dot_radius = random.randint(5, 10)
                color = random.choice(number_colors)
                
                for dx in range(-dot_radius, dot_radius+1):
                    for dy in range(-dot_radius, dot_radius+1):
                        if dx**2 + dy**2 <= dot_radius**2:
                            nx, ny = int(pos_x + dx), int(pos_y + dy)
                            if 0 <= nx < width and 0 <= ny < height and np.sqrt((nx - center_x)**2 + (ny - center_y)**2) <= radius:
                                img[ny, nx] = color
    
    # Convert numpy array to PIL Image
    return Image.fromarray(img)

def create_color_simulator(image):
    """Simulate how an image might appear to someone with different types of color blindness"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Define color blindness simulation matrices (simplified)
    # Protanopia (red-blind)
    protanopia = np.array([
        [0.567, 0.433, 0.000],
        [0.558, 0.442, 0.000],
        [0.000, 0.242, 0.758]
    ])
    
    # Deuteranopia (green-blind)
    deuteranopia = np.array([
        [0.625, 0.375, 0.000],
        [0.700, 0.300, 0.000],
        [0.000, 0.300, 0.700]
    ])
    
    # Tritanopia (blue-blind)
    tritanopia = np.array([
        [0.950, 0.050, 0.000],
        [0.000, 0.433, 0.567],
        [0.000, 0.475, 0.525]
    ])
    
    # Apply transformations
    simulated_images = {}
    
    # Reshape to apply transformation
    h, w, _ = img_array.shape
    flat_img = img_array.reshape(-1, 3)
    
    # Apply color matrices
    protanopia_img = np.dot(flat_img, protanopia.T).reshape(h, w, 3).astype(np.uint8)
    deuteranopia_img = np.dot(flat_img, deuteranopia.T).reshape(h, w, 3).astype(np.uint8)
    tritanopia_img = np.dot(flat_img, tritanopia.T).reshape(h, w, 3).astype(np.uint8)
    
    # Generate monochromacy (grayscale)
    grayscale = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    monochromacy_img = np.stack((grayscale,)*3, axis=-1).astype(np.uint8)
    
    # Convert back to PIL Images
    simulated_images["Protanopia (Red-blind)"] = Image.fromarray(protanopia_img)
    simulated_images["Deuteranopia (Green-blind)"] = Image.fromarray(deuteranopia_img)
    simulated_images["Tritanopia (Blue-blind)"] = Image.fromarray(tritanopia_img)
    simulated_images["Monochromacy (No color)"] = Image.fromarray(monochromacy_img)
    
    return simulated_images

# Visual Acuity Test (Snellen Chart) Functions
def create_snellen_chart():
    """Create a simplified Snellen chart for visual acuity testing"""
    snellen_rows = {
        "20/200": {"letters": ["E", "F", "P"], "size": 80},
        "20/100": {"letters": ["F", "D", "Z"], "size": 40},
        "20/70": {"letters": ["P", "T", "C"], "size": 28},
        "20/50": {"letters": ["D", "K", "N"], "size": 20},
        "20/40": {"letters": ["Z", "H", "V"], "size": 16},
        "20/30": {"letters": ["C", "S", "R"], "size": 12},
        "20/25": {"letters": ["O", "N", "D"], "size": 10},
        "20/20": {"letters": ["F", "Z", "P"], "size": 8}
    }
    
    html_chart = "<div style='background-white; padding: 20px; border-radius: 10px;'>"
    
    for acuity, row_data in snellen_rows.items():
        letters = " ".join(row_data["letters"])
        size = row_data["size"]
        html_chart += f"<div class='snellen-row' style='font-size: {size}px;'>{letters}</div>"
    
    html_chart += "</div>"
    
    return html_chart, snellen_rows

# Contrast Sensitivity Test Functions
def create_contrast_sensitivity_test():
    """Create a contrast sensitivity test with letters of decreasing contrast"""
    contrasts = [
        {"level": "100%", "color": "#000000", "bg": "#FFFFFF"},  # Black on white
        {"level": "75%", "color": "#404040", "bg": "#FFFFFF"},   # Dark gray
        {"level": "50%", "color": "#808080", "bg": "#FFFFFF"},   # Medium gray
        {"level": "25%", "color": "#C0C0C0", "bg": "#FFFFFF"},   # Light gray
        {"level": "15%", "color": "#D9D9D9", "bg": "#FFFFFF"},   # Very light gray
        {"level": "10%", "color": "#E6E6E6", "bg": "#FFFFFF"},   # Extremely light gray
        {"level": "5%", "color": "#F2F2F2", "bg": "#FFFFFF"}     # Nearly invisible
    ]
    
    test_letters = ["R", "K", "D", "N", "O", "S", "V", "Z"]
    
    html_test = "<div style='background-color: white; padding: 20px; border-radius: 10px;'>"
    
    for i, contrast in enumerate(contrasts):
        random_letters = random.sample(test_letters, 3)
        letters = " ".join(random_letters)
        html_test += f"""
            <div style='color: {contrast["color"]};
                  background-color: {contrast["bg"]};
                  margin-bottom: 10px;
                  font-size: 36px;
                  font-family: Arial, sans-serif;
                  letter-spacing: 8px;
                  padding: 10px;
                  font-weight: bold;'>
                Row {i+1} ({contrast["level"]} contrast): {letters}
            </div>
        """
    
    html_test += "</div>"
    
    return html_test, contrasts

# Astigmatism Test Functions
def create_astigmatism_chart():
    """Create a sunburst pattern for astigmatism testing"""
    # We'll create this using SVG for precise line drawing
    size = 400
    center = size // 2
    num_lines = 36  # Lines at 5-degree intervals
    
    svg = f"""
    <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
        <circle cx="{center}" cy="{center}" r="5" fill="red" />
    """
    
    for i in range(num_lines):
        angle = i * (360 / num_lines)
        rad = angle * np.pi / 180
        x1 = center
        y1 = center
        x2 = center + (center - 20) * np.cos(rad)
        y2 = center + (center - 20) * np.sin(rad)
        
        svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="2" />'
    
    svg += "</svg>"
    
    return svg

# Peripheral Vision Test functions
def create_peripheral_test():
    """Create a simple peripheral vision test with a central fixation point"""
    width, height = 800, 400
    
    # Create a blank canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a central fixation cross
    center_x, center_y = width // 2, height // 2
    thickness = 2
    size = 10
    
    # Vertical line
    canvas[center_y-size:center_y+size, center_x-thickness:center_x+thickness] = [255, 0, 0]
    # Horizontal line
    canvas[center_y-thickness:center_y+thickness, center_x-size:center_x+size] = [255, 0, 0]
    
    # Convert to PIL Image
    return Image.fromarray(canvas)

# Reaction Time Test functions
def get_encoded_image(color, width=200, height=200):
    """Generate a colored square image and return it as base64 encoded"""
    img = Image.new('RGB', (width, height), color=color)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Load model once and cache
@st.cache_resource
def get_model():
    model_path = r"./model/final_dr_model.pt"
    model_path = r"./model/final_dr_model.pt"
    model, _ = load_model(model_path, device=device)  # Unpack correctly
    return model

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'test_menu' not in st.session_state:
    st.session_state.test_menu = None

# Define navigation function
def navigate_to(destination, test_type=None):
    st.session_state.page = destination
    if test_type:
        st.session_state.test_menu = test_type
    st.rerun()

# Sidebar for app navigation and terminology
with st.sidebar:
    st.title("Vision Health Suite")
    
    # Navigation
    st.header("Navigation")
    nav_options = {
        'home': "🏠 Home",
        'dr_classification': "🩺 Diabetic Retinopathy Classification",
        'vision_tests': "👁️ Vision Tests"
    }
    
    selected_nav = st.radio("Go to:", list(nav_options.values()))
    # Update the page state based on selection
    for key, value in nav_options.items():
        if value == selected_nav:
            if st.session_state.page != key:
                navigate_to(key)
    
    # If on vision tests page, show test selection
    if st.session_state.page == 'vision_tests':
        st.header("Available Tests")
        test_options = {
            'color_vision': "🎨 Color Vision Test",
            'visual_acuity': "👓 Visual Acuity Test",
            'contrast': "◐ Contrast Sensitivity Test",
            'astigmatism': "🔆 Astigmatism Test",
            'peripheral': "👀 Peripheral Vision Test",
            'reaction': "⚡ Reaction Time Test",
            'amsler': "🔲 Amsler Grid Test"
        }
        
        selected_test = st.selectbox("Select test:", list(test_options.values()))
        # Update the test menu state based on selection
        for key, value in test_options.items():
            if value == selected_test:
                if st.session_state.test_menu != key:
                    st.session_state.test_menu = key
                    st.rerun()
    
    # DR terminology explanation (only show on DR page)
    if st.session_state.page == 'dr_classification':
        st.header("Understanding the Terms")
        
        # Explanation of DR classifications
        st.subheader("Diabetic Retinopathy Stages")
        classifications = {
            "No DR": "No visible signs of diabetic retinopathy",
            "Mild DR": "Small areas of balloon-like swelling in the retina's blood vessels",
            "Moderate DR": "More extensive damage to blood vessels, affecting blood supply to retina",
            "Severe DR": "Many blocked blood vessels, signaling the retina to grow new blood vessels",
            "Proliferative DR": "Advanced stage with abnormal new blood vessels growing on the retina"
        }
        
        for term, explanation in classifications.items():
            with st.expander(term):
                st.write(explanation)
        
        # What is DR?
        with st.expander("What is Diabetic Retinopathy?"):
            st.write("Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people with diabetes. It affects blood vessels in the retina (the light-sensitive layer at the back of the eye).")

# Try loading the model (for DR classification)
try:
    if st.session_state.page == 'dr_classification':
        model = get_model()
except Exception as e:
    if st.session_state.page == 'dr_classification':
        st.error(f"Error loading model: {e}")
        st.stop()

# HOME PAGE
if st.session_state.page == 'home':
    st.title("🌟 Welcome to Vision Health Suite")
    
    st.markdown("""
    ## Comprehensive Eye Health Tools
    
    This application offers a range of tools to assess various aspects of vision health:
    
    ### 🩺 **Diabetic Retinopathy Classifier**
    Upload a retinal image to assess the presence and severity of diabetic retinopathy. 
    Our AI model will analyze the image and provide a classification with recommendations.
    
    ### 👁️ **Vision Tests**
    Test different aspects of your visual function:
    - **Color Vision Test** - Detect color blindness and deficiencies
    - **Visual Acuity Test** - Measure your visual clarity at different distances
    - **Contrast Sensitivity Test** - Assess your ability to distinguish subtle contrasts
    - **Astigmatism Test** - Check for irregular curvature of the eye
    - **Peripheral Vision Test** - Evaluate your side vision capabilities
    - **Reaction Time Test** - Measure how quickly your eyes and brain respond to visual stimuli
    - **Amsler Grid Test** - Check for macular degeneration issues
    
    ### ⚠️ Important Disclaimer
    These tests are for informational and educational purposes only. They do not replace comprehensive 
    eye examinations by qualified healthcare professionals. Always consult with an eye care specialist 
    for proper diagnosis and treatment.
    """)


# DIABETIC RETINOPATHY CLASSIFICATION PAGE
elif st.session_state.page == 'dr_classification':
    st.title("Diabetic Retinopathy Classification 🩺")
    st.write("Upload an image of a retina to classify the severity of diabetic retinopathy.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Create tabs for results and additional features
            tabs = st.tabs(["Results & Recommendations", "Color Vision Simulation"])
            
            with tabs[0]:  # Results tab
                # Perform prediction
                CLASSES = ["No DR ✅", "Mild DR 🟡", "Moderate DR 🟠", "Severe DR 🔴", "Proliferative DR ⚠️"]
                predicted_class, confidence, probabilities = predict(model, image, transform=None, 
                                                                   classes=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
                
                # Display results
                st.subheader("Prediction Results")
                st.success(f"**Predicted Class:** {CLASSES[predicted_class]} | **Confidence:** {confidence:.2f}%")
                
                # Severity-Based Recommendations with color coding
                recommendations = {
                    0: ("No Diabetic Retinopathy ✅", "Your retina appears healthy. Maintain regular eye checkups."),
                    1: ("Mild Diabetic Retinopathy 🟡", "Early-stage detected. Consult an ophthalmologist."),
                    2: ("Moderate Diabetic Retinopathy 🟠", "Schedule an appointment and manage blood sugar strictly."),
                    3: ("Severe Diabetic Retinopathy 🔴", "Seek urgent medical attention and consider treatment."),
                    4: ("Proliferative Diabetic Retinopathy ⚠️", "Urgent action required—consult an ophthalmologist immediately.")
                }
                
                # Color codes for different severity levels
                severity_colors = {
                    0: "#28a745",  # Green for No DR
                    1: "#ffc107",  # Yellow
                    2: "#fd7e14",  # Orange
                    3: "#dc3545",  # Red
                    4: "#6c1a1a"   # Dark red
                }
                
                st.markdown(f"""
                <div style="background-color: {severity_colors[predicted_class]}20; 
                           border-left: 5px solid {severity_colors[predicted_class]}; 
                           padding: 10px; border-radius: 5px;">
                    <h3 style="color: {severity_colors[predicted_class]};">{recommendations[predicted_class][0]}</h3>
                    <p>{recommendations[predicted_class][1]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability chart
                st.subheader("Classification Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(CLASSES, probabilities, color=[severity_colors[i] for i in range(len(CLASSES))])
                ax.set_ylabel('Probability')
                ax.set_title('Diabetic Retinopathy Classification Probabilities')
                ax.set_ylim(0, 1.0)
                
                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.1%}', ha='center', va='bottom', fontsize=10)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Next steps based on classification
                st.subheader("Recommended Next Steps")
                
                if predicted_class == 0:
                    st.markdown("""
                    - **Continue Regular Screening**: Even though no signs of DR are present, regular screenings are important, especially for people with diabetes.
                    - **Maintain Diabetes Management**: Keep blood sugar levels, blood pressure, and cholesterol under control.
                    - **Schedule Next Checkup**: Plan for your next annual eye examination.
                    """)
                elif predicted_class == 1:
                    st.markdown("""
                    - **Consult an Ophthalmologist**: Have a professional evaluate these early signs.
                    - **Improve Glucose Control**: Tighten your diabetes management plan.
                    - **Follow-Up Screening**: Get another retinal screening in 6-12 months.
                    - **Monitor for Changes**: Be alert for any changes in vision and report them to your doctor.
                    """)
                elif predicted_class == 2:
                    st.markdown("""
                    - **Ophthalmologist Consultation**: Schedule an appointment in the next 30 days.
                    - **Strict Diabetes Management**: Work closely with your doctor to improve control.
                    - **Consider Treatment Options**: Discuss possible treatments to prevent progression.
                    - **More Frequent Monitoring**: Plan for retinal screenings every 3-6 months.
                    """)
                elif predicted_class == 3:
                    st.markdown("""
                    - **Urgent Specialist Consultation**: See a retina specialist within 1-2 weeks.
                    - **Prepare for Treatment**: Be ready to discuss treatment options like laser therapy.
                    - **Intensify Diabetes Management**: Work with your healthcare team on immediate improvements.
                    - **Frequent Monitoring**: Expect more frequent eye examinations.
                    """)
                else:  # Proliferative DR
                    st.markdown("""
                    - **Immediate Medical Attention**: This is a medical emergency - see a specialist immediately.
                    - **Treatment Planning**: Urgent treatments like laser photocoagulation or anti-VEGF injections may be needed.
                    - **Aggressive Diabetes Management**: Work closely with your endocrinologist.
                    - **Monitor for Complications**: Be vigilant for symptoms of vision loss.
                    """)
                
                # Download report button
                report_data = f"""
                # Diabetic Retinopathy Screening Report
                
                ## Patient Information
                - **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                ## Screening Results
                - **Classification**: {CLASSES[predicted_class]}
                - **Confidence**: {confidence:.2f}%
                
                ## Recommendations
                {recommendations[predicted_class][1]}
                
                ## Important Note
                This automated screening is not a substitute for professional medical advice. 
                Please consult with an eye care specialist for proper diagnosis and treatment.
                """
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_data,
                    file_name=f"dr_screening_report_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )
            
            with tabs[1]:  # Color Vision Simulation
                st.subheader("Color Vision Impact Simulation")
                st.write("This simulation shows how this retinal image might appear to someone with different types of color vision deficiencies.")
                
                # Create simulations
                simulations = create_color_simulator(image)
                
                # Display simulations in a grid
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(simulations["Protanopia (Red-blind)"], caption="Protanopia (Red-blind)", use_container_width=True)
                    st.image(simulations["Tritanopia (Blue-blind)"], caption="Tritanopia (Blue-blind)", use_container_width=True)
                
                with col2:
                    st.image(simulations["Deuteranopia (Green-blind)"], caption="Deuteranopia (Green-blind)", use_container_width=True)
                    st.image(simulations["Monochromacy (No color)"], caption="Monochromacy (No color)", use_container_width=True)
                
                st.info("Note: The appearance of diabetic retinopathy indicators may be less noticeable with certain color vision deficiencies, making regular professional screenings even more important.")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    else:
        # Display sample images when no upload
        st.info("Please upload a retinal image to receive a classification.")
        
        # Explanation with example images
        st.subheader("How It Works")
        st.write("""
        Our system uses deep learning to analyze patterns in retinal images that indicate different 
        stages of diabetic retinopathy. The model has been trained on thousands of classified retinal images.
        
        Diabetic retinopathy is a complication of diabetes that affects the eyes. It's caused by damage to the 
        blood vessels in the tissue at the back of the eye (retina). If left untreated, it can lead to blindness.
        """)

# VISION TESTS PAGE
elif st.session_state.page == 'vision_tests':
    st.title("Vision Tests 👁️")
    
    # Helper function for test disclaimers
    def show_disclaimer():
        st.warning("""
        **Important**: These tests are for informational purposes only and not intended to replace 
        professional medical advice or examination. If you experience vision problems, 
        please consult an eye care professional.
        """)
    
    # COLOR VISION TEST
    if st.session_state.test_menu == 'color_vision':
        st.header("Color Vision Test 🎨")
        show_disclaimer()
        
        st.write("""
        This test helps identify potential color vision deficiencies. Observe each plate and 
        identify the number or pattern you see. Remember, this is a simplified screening test and 
        not a comprehensive diagnostic tool.
        """)
        
        # Difficulty selection
        difficulty = st.select_slider(
            "Select test difficulty:",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        difficulty = difficulty.lower()
        
        # Color blindness type selection
        deficiency_type = st.radio(
            "Select deficiency type to test for:",
            ["Red-Green (Most Common)", "Blue-Yellow", "Total Color Blindness"]
        )
        
        if deficiency_type == "Red-Green (Most Common)":
            test_type = "red-green"
        elif deficiency_type == "Blue-Yellow":
            test_type = "tritan"
        else:
            test_type = "monochromacy"
        
        # Generate test plates
        if st.button("Generate Test Plates"):
            st.session_state['test_numbers'] = [random.randint(1, 9) for _ in range(5)]
            st.session_state['user_answers'] = [0] * 5  # Initialize with zeros
        
        # Initialize answers if they don't exist but test numbers do
        if 'test_numbers' in st.session_state and 'user_answers' not in st.session_state:
            st.session_state['user_answers'] = [0] * len(st.session_state['test_numbers'])
        
        # If test numbers exist in session state, display the test
        if 'test_numbers' in st.session_state:
            st.subheader("What number do you see in each plate?")
            
            # Show the test plates
            for i, num in enumerate(st.session_state['test_numbers']):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    test_plate = create_ishihara_plate(num, plate_type=test_type, difficulty=difficulty)
                    st.image(test_plate, caption=f"Plate {i+1}", use_container_width=True)
                
                with col2:
                    # Create a callback function to update the specific answer
                    def make_update_callback(index):
                        def update_answer():
                            st.session_state['user_answers'][index] = st.session_state[f"plate_{index}"]
                        return update_answer
                    
                    # Use number_input with unique key and current value from session state
                    st.number_input(
                        f"Enter the number you see in plate {i+1}:", 
                        min_value=0, 
                        max_value=9, 
                        value=st.session_state['user_answers'][i],
                        key=f"plate_{i}",
                        on_change=make_update_callback(i)
                    )
            
            # Submit button
            if st.button("Submit Answers"):
                # Calculate score
                correct = sum(1 for a, b in zip(st.session_state['test_numbers'], st.session_state['user_answers']) if a == b)
                total = len(st.session_state['test_numbers'])
                
                st.subheader("Results")
                
                # Display results
                st.write(f"You correctly identified {correct} out of {total} plates.")
                
                # Interpret results
                if correct == total:
                    st.success("🎉 Perfect score! You appear to have normal color vision for this test type.")
                elif correct >= total * 0.8:
                    st.success("👍 Good result! You may have mild color vision deficiency or the test was difficult to see.")
                elif correct >= total * 0.6:
                    st.warning("⚠️ You missed several plates. You may have a moderate color vision deficiency.")
                else:
                    st.error("⚠️ You missed many plates. This could indicate a significant color vision deficiency.")
                
                # Show what the actual numbers were
                st.subheader("Test Key")
                for i, (expected, answered) in enumerate(zip(st.session_state['test_numbers'], st.session_state['user_answers'])):
                    if expected == answered:
                        st.write(f"✅ Plate {i+1}: The number was {expected}, you answered {answered}")
                    else:
                        st.write(f"❌ Plate {i+1}: The number was {expected}, you answered {answered}")
                
                st.info("""
                **What Next?**  
                If you performed poorly on this test, consider:
                1. Taking the test in better lighting conditions
                2. Trying a different difficulty level
                3. Consulting with an eye care professional for a comprehensive color vision assessment
                """)
    
    # VISUAL ACUITY TEST
    elif st.session_state.test_menu == 'visual_acuity':
        st.header("Visual Acuity Test 👓")
        show_disclaimer()
        
        st.write("""
        This test measures your ability to see details at various distances, similar to a Snellen chart 
        used in eye exams. For best results:
        
        1. Position yourself about 10 feet (3 meters) from your screen
        2. If you wear glasses or contacts for distance vision, keep them on
        3. Read the letters from top to bottom, and note the smallest line you can read clearly
        """)
        
        # Create Snellen chart
        snellen_chart, snellen_data = create_snellen_chart()
        
        # Display the chart
        st.markdown(snellen_chart, unsafe_allow_html=True)
        
        # Self-reporting section
        st.subheader("Self Assessment")
        
        smallest_readable = st.select_slider(
            "What's the smallest line you can read clearly?",
            options=list(snellen_data.keys()),
            value="20/50"
        )
        
        # Display results
        st.subheader("Interpretation")
        
        # Simple interpretation logic
        if smallest_readable == "20/20":
            st.success("🎉 You appear to have normal visual acuity!")
        elif smallest_readable in ["20/25", "20/30", "20/40"]:
            st.info("👍 You have mild visual impairment or slight nearsightedness/farsightedness.")
        elif smallest_readable in ["20/50", "20/70"]:
            st.warning("⚠️ You may have moderate visual impairment. Consider an eye examination.")
        else:
            st.error("⚠️ You may have significant visual impairment. An eye examination is recommended.")
        
        st.markdown("""
        **Understanding the Numbers**
        
        The notation (like 20/40) represents visual acuity where:
        - The first number (20) is the test distance in feet
        - The second number is the distance at which a person with normal vision can read the same line
        
        For example, 20/40 means you can see at 20 feet what someone with normal vision can see at 40 feet.
        
        **Important Note**: Screen-based tests have limitations. Factors like screen size, resolution, and 
        viewing distance affect accuracy. This test is only for informational purposes.
        """)
    
    # CONTRAST SENSITIVITY TEST
    elif st.session_state.test_menu == 'contrast':
        st.header("Contrast Sensitivity Test ◐")
        show_disclaimer()
    
        st.write("""
        This test evaluates your ability to distinguish between subtle differences in contrast. 
        Good contrast sensitivity is important for activities like night driving, navigating in fog,
        or reading low-contrast text.
        """)
    
        # Create contrast test
        contrast_test, contrast_data = create_contrast_sensitivity_test()
    
        # Display the test with a more explicit component
        st.subheader("Contrast Test")
        st.write("Try to read all three letters in each row below:")
    
        # Use components.html for better HTML rendering
        from streamlit.components.v1 import html
        html(contrast_test, height=700, scrolling=False)
    
        # Self-reporting section
        st.subheader("Self Assessment")
    
        lowest_readable = st.select_slider(
            "What's the lowest contrast row where you can still read all three letters?",
            options=[f"Row {i+1} ({c['level']} contrast)" for i, c in enumerate(contrast_data)],
            value=f"Row 4 (25% contrast)"
        )
    
        # Extract row number from selection
        selected_row = int(lowest_readable.split()[1])
    
        # Display results
        st.subheader("Interpretation")
    
        # Interpret results
        if selected_row >= 6:
            st.success("🎉 Excellent contrast sensitivity! You can discern very subtle differences in contrast.")
        elif selected_row >= 4:
            st.info("👍 Good contrast sensitivity. This is in the normal range for most people.")
        elif selected_row >= 2:
            st.warning("⚠️ Reduced contrast sensitivity. This may affect activities like night driving.")
        else:
            st.error("⚠️ Significantly reduced contrast sensitivity. Consider consulting an eye care professional.")
    
        st.markdown("""
        **Why Contrast Sensitivity Matters**
    
        While visual acuity tests how well you can see details at high contrast (black letters on white),
        contrast sensitivity measures your ability to detect subtle differences between light and dark areas.
    
        Low contrast sensitivity can:
        - Make night driving difficult or dangerous
        - Make it hard to see in fog, rain, or snow
        - Affect depth perception
        - Make it difficult to distinguish facial expressions
        - Be an early indicator of eye conditions like cataracts, glaucoma, or diabetic retinopathy
    
        If you struggle with this test, especially if you've noticed changes in your vision, 
        consider speaking with an eye care professional.
        """)
    
    # ASTIGMATISM TEST
    elif st.session_state.test_menu == 'astigmatism':
        st.header("Astigmatism Test 🔆")
        show_disclaimer()
        
        st.write("""
        This test helps detect astigmatism, a common vision condition caused by an irregularly shaped cornea
        or lens. The test uses a sunburst pattern - if you have astigmatism, some lines may appear darker, 
        blurrier, or more distinct than others.
        """)
        
        # Create astigmatism chart
        astigmatism_chart = create_astigmatism_chart()
        
        # Display instructions
        st.markdown("""
        **Instructions:**
        1. If you wear glasses or contacts, keep them on
        2. Look at the center dot in the pattern below
        3. Note if any lines appear darker, blurrier, or more distinct than others
        4. If lines in certain directions appear different, this may indicate astigmatism
        """)
        
        # Display the chart
        st.markdown(f"""
        <div class="astigmatism-pattern">
            {astigmatism_chart}
        </div>
        """, unsafe_allow_html=True)
        
        # Self-assessment
        st.subheader("Self Assessment")
        
        option = st.radio(
            "How do the lines appear to you?",
            [
                "All lines appear equally clear and dark",
                "Lines in some directions appear darker or clearer than others",
                "Lines in some directions appear blurrier than others", 
                "The pattern is distorted or some lines appear wavy"
            ]
        )
        
        # Directions selector for those who notice differences
        if option != "All lines appear equally clear and dark":
            directions = st.multiselect(
                "Which directions appear different? (Select all that apply)",
                ["Horizontal (left to right)", "Vertical (top to bottom)", 
                 "Diagonal (top-left to bottom-right)", "Diagonal (top-right to bottom-left)"]
            )
        
        # Submit button
        if st.button("Submit Assessment"):
            st.subheader("Interpretation")
            
            if option == "All lines appear equally clear and dark":
                st.success("🎉 Your results suggest you may not have significant astigmatism.")
            else:
                st.warning("""
                ⚠️ Your results suggest you may have some degree of astigmatism. 
                When lines appear different in certain directions, it often indicates an irregular 
                curvature of the cornea or lens.
                """)
                
                # Additional interpretation based on directions
                if 'directions' in locals() and directions:
                    st.write("Based on the directions you selected, your astigmatism may be:")
                    
                    if "Horizontal (left to right)" in directions:
                        st.write("- Affecting your vision horizontally (With-the-rule or Against-the-rule astigmatism)")
                    
                    if "Vertical (top to bottom)" in directions:
                        st.write("- Affecting your vision vertically (With-the-rule or Against-the-rule astigmatism)")
                    
                    if any("Diagonal" in d for d in directions):
                        st.write("- Oblique astigmatism (affecting diagonal vision)")
            
            st.info("""
            **What Next?**  
            This simple test can only suggest the possibility of astigmatism.
            
            If you suspect you have astigmatism, consider:
            1. Getting a comprehensive eye exam from an optometrist or ophthalmologist
            2. Discussing vision correction options (glasses, contact lenses, or in some cases, refractive surgery)
            3. Regular follow-ups as recommended by your eye care professional
            
            Astigmatism is very common and typically easy to correct with the right prescription.
            """)
    
    # PERIPHERAL VISION TEST
    elif st.session_state.test_menu == 'peripheral':
        st.header("Peripheral Vision Test 👀")
        show_disclaimer()
        
        st.write("""
        This test helps evaluate your peripheral (side) vision, which is important 
        for spatial awareness and detecting movement outside your direct line of sight.
        
        The test will display a central fixation point and random dots or objects 
        in the periphery. Your task is to detect these while keeping your eyes focused on the center.
        """)
        
        # Start test button
        start_test = st.button("Start Test")
        
        if start_test or 'peripheral_test_running' in st.session_state:
            st.session_state['peripheral_test_running'] = True
            
            # Test parameters
            if 'peripheral_correct' not in st.session_state:
                st.session_state['peripheral_correct'] = 0
                st.session_state['peripheral_total'] = 0
                st.session_state['peripheral_positions'] = []
            
            # Display instructions for the ongoing test
            st.markdown("""
            **Instructions:**
            1. Focus your eyes on the red cross in the center of the image
            2. Without moving your eyes from the center, try to detect the dot's position
            3. Select where you saw the dot appear
            """)
            
            # Create a test image with a fixation point
            if 'peripheral_test_image' not in st.session_state or st.session_state['peripheral_total'] < 5:
                base_image = create_peripheral_test()
                
                # Add a dot at a random position if not already generated
                if len(st.session_state['peripheral_positions']) <= st.session_state['peripheral_total']:
                    img_array = np.array(base_image)
                    height, width, _ = img_array.shape
                    
                    # Define possible positions (avoid center)
                    positions = [
                        ("Top Left", (width//4, height//4)),
                        ("Top Right", (width*3//4, height//4)),
                        ("Bottom Left", (width//4, height*3//4)),
                        ("Bottom Right", (width*3//4, height*3//4)),
                        ("Far Left", (width//8, height//2)),
                        ("Far Right", (width*7//8, height//2)),
                    ]
                    
                    # Select a random position
                    position_name, (x, y) = random.choice(positions)
                    st.session_state['peripheral_positions'].append((position_name, (x, y)))
                
                # Get current position
                position_name, (x, y) = st.session_state['peripheral_positions'][st.session_state['peripheral_total']]
                
                # Add dot to the image at the selected position
                draw_img = np.array(base_image)
                cv_radius = 8
                cv_color = (0, 0, 255)  # Blue dot
                cv_thickness = -1  # Fill the circle
                
                # Draw the dot
                draw_img = cv2.circle(draw_img, (x, y), cv_radius, cv_color, cv_thickness)
                
                # Convert back to PIL and display
                peripheral_image = Image.fromarray(draw_img)
                st.image(peripheral_image, use_container_width=True)
                
                # Position selection
                selected_position = st.radio(
                    "Where did you see the dot appear?",
                    ["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Far Left", "Far Right", "I didn't see a dot"]
                )
                
                # Check answer button
                if st.button("Submit Answer"):
                    st.session_state['peripheral_total'] += 1
                    
                    if selected_position == position_name:
                        st.session_state['peripheral_correct'] += 1
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ The dot was actually in the {position_name} position.")
                    
                    # Continue or show results
                    if st.session_state['peripheral_total'] >= 5:
                        # Display final results
                        score = st.session_state['peripheral_correct'] / st.session_state['peripheral_total'] * 100
                        
                        st.subheader("Test Results")
                        st.write(f"You correctly identified {st.session_state['peripheral_correct']} out of {st.session_state['peripheral_total']} positions ({score:.1f}%).")
                        
                        # Interpret results
                        if score >= 80:
                            st.success("🎉 Excellent peripheral vision awareness!")
                        elif score >= 60:
                            st.info("👍 Good peripheral vision. This is in the normal range.")
                        elif score >= 40:
                            st.warning("⚠️ Your peripheral vision awareness may be limited.")
                        else:
                            st.error("⚠️ Significant difficulty with peripheral vision detection. Consider consulting an eye care professional.")
                        
                        # Reset for new test
                        if st.button("Try Again"):
                            del st.session_state['peripheral_test_running']
                            del st.session_state['peripheral_correct']
                            del st.session_state['peripheral_total']
                            del st.session_state['peripheral_positions']
                            st.rerun()
                    else:
                        # Continue to next dot
                        st.button("Next Dot", key="next_peripheral")
            
            else:
                st.error("An error occurred with the peripheral vision test. Please refresh the page and try again.")
        
        else:
            # Show informational content when test is not running
            st.markdown("""
            **Why Peripheral Vision Matters**
            
            Peripheral vision helps you:
            - Detect motion and objects outside your central field of view
            - Maintain spatial awareness and orientation
            - Navigate your environment safely
            - Avoid collisions while walking or driving
            
            Reduced peripheral vision (tunnel vision) can be caused by:
            - Glaucoma
            - Retinitis pigmentosa
            - Certain neurological conditions
            - Stroke or brain injury
            
            This simplified test is not diagnostic but can help you become aware of potential limitations in your peripheral vision.
            """)
    
    # REACTION TIME TEST
    elif st.session_state.test_menu == 'reaction':
        st.header("Visual Reaction Time Test ⚡")
        show_disclaimer()
        
        st.write("""
        This test measures how quickly you can respond to a visual stimulus. Fast reaction times 
        are important for activities like driving, sports, and many everyday tasks.
        
        The test will display a colored square at random intervals. Your task is to click the button 
        as soon as you see the color change.
        """)
        
        # Initialize session state for reaction time test
        if 'reaction_stage' not in st.session_state:
            st.session_state['reaction_stage'] = 'ready'
            st.session_state['reaction_times'] = []
            st.session_state['reaction_start_time'] = None
        
        # Different stages of the test
        if st.session_state['reaction_stage'] == 'ready':
            if st.button("Start Reaction Test", use_container_width=True):
                st.session_state['reaction_stage'] = 'waiting'
                st.session_state['reaction_start_time'] = time.time() + random.uniform(1.5, 4.0)
                st.rerun()
        
        elif st.session_state['reaction_stage'] == 'waiting':
            current_time = time.time()
            
            if current_time < st.session_state['reaction_start_time']:
                # Still waiting - show gray square
                gray_img = get_encoded_image("gray")
                st.markdown(f"""
                <div style="display: flex; justify-content: center; margin: 30px 0;">
                    <img src="{gray_img}" width="200" height="200">
                </div>
                <p style="text-align: center;">Wait for the square to turn green, then click as fast as you can!</p>
                """, unsafe_allow_html=True)
                
                # Force a rerun to update the page
                time.sleep(0.1)
                st.rerun()
            else:
                # Time to show the green square
                st.session_state['reaction_stage'] = 'showing'
                st.session_state['reaction_show_time'] = current_time
                st.rerun()
        
        elif st.session_state['reaction_stage'] == 'showing':
            # Show green square and wait for click
            green_img = get_encoded_image("green")
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 30px 0;">
                <img src="{green_img}" width="200" height="200">
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("CLICK NOW!", use_container_width=True):
                reaction_time = time.time() - st.session_state['reaction_show_time']
                st.session_state['reaction_times'].append(reaction_time)
                st.session_state['reaction_stage'] = 'result'
                st.rerun()
        
        elif st.session_state['reaction_stage'] == 'result':
            # Show result of last test
            last_time = st.session_state['reaction_times'][-1]
            
            st.markdown(f"<h3 style='text-align: center;'>Your reaction time: {last_time*1000:.1f} milliseconds</h3>", unsafe_allow_html=True)
            
            # Interpret the result
            if last_time < 0.2:
                st.success("🎉 Excellent! Your reaction time is very fast.")
            elif last_time < 0.3:
                st.info("👍 Good reaction time! This is in the typical range.")
            elif last_time < 0.5:
                st.warning("⚠️ Your reaction time is a bit slower than average.")
            else:
                st.error("⚠️ Your reaction time is significantly slower than average.")
            
            # Show stats if multiple attempts
            if len(st.session_state['reaction_times']) > 1:
                avg_time = sum(st.session_state['reaction_times']) / len(st.session_state['reaction_times'])
                best_time = min(st.session_state['reaction_times'])
                
                st.markdown(f"""
                **Your Statistics:**
                - Average time: {avg_time*1000:.1f} ms
                - Best time: {best_time*1000:.1f} ms
                - Number of attempts: {len(st.session_state['reaction_times'])}
                """)
            
            # Options for next steps
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Try Again", use_container_width=True):
                    st.session_state['reaction_stage'] = 'ready'
                    st.rerun()
            
            with col2:
                if st.button("Reset Stats", use_container_width=True):
                    st.session_state['reaction_stage'] = 'ready'
                    st.session_state['reaction_times'] = []
                    st.rerun()
       
        # Add informational content
        st.markdown("""
        **About Reaction Times**
       
        Average visual reaction times typically range from:
        - 200-250ms: Very good
        - 250-300ms: Average
        - 300-500ms: Slower than average
        - 500ms+: Significantly delayed
       
        Factors that affect reaction time:
        - Age (reaction time typically increases with age)
        - Fatigue or sleep deprivation
        - Medications or substances
        - Screen lag (note that your device may add some delay)
        - Distractions in your environment
       
        Improving reaction time:
        - Regular physical exercise
        - Adequate sleep
        - Practice and training
        - Maintaining good overall health
        """)
   
    # AMSLER GRID TEST
    elif st.session_state.test_menu == 'amsler':
        st.header("Amsler Grid Test 🔲")
        show_disclaimer()
       
        st.write("""
        The Amsler grid is used to detect problems in central vision, particularly those related to the macula
        (the central part of the retina). This test can help detect early signs of macular degeneration,
        a leading cause of vision loss in older adults.
        """)
       
        # Display instructions
        st.markdown("""
        **Instructions:**
        1. If you wear reading glasses, put them on
        2. Hold your head about 12-14 inches (30-35 cm) from the screen
        3. Cover or close one eye (test each eye separately)
        4. Focus on the central dot in the grid
        5. While looking at the dot, notice if:
           - Any lines appear wavy, distorted, or broken
           - Any areas of the grid appear blurry, dark, or missing
           - The central dot disappears or is difficult to see
        """)
       
        # Display the Amsler grid
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
            <div class="amsler-grid" style="max-width: 400px; max-height: 400px;">
                <div style="position: relative; width: 100%; height: 100%;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                          height: 10px; width: 10px; background-color: red; border-radius: 50%;">
                    </div>
                </div>
            </div>
            <p style="margin-top: 10px; text-align: center;">Focus on the red dot in the center</p>
        </div>
        """, unsafe_allow_html=True)
       
        # Self-assessment
        st.subheader("Self Assessment")
       
        left_eye = st.multiselect(
            "Testing LEFT eye - Did you notice any of the following? (Select all that apply)",
            ["All lines appear straight and regular",
             "Some lines appear wavy or distorted",
             "Some areas appear blurry or missing",
             "The central dot disappears or is hard to see"]
        )
       
        right_eye = st.multiselect(
            "Testing RIGHT eye - Did you notice any of the following? (Select all that apply)",
            ["All lines appear straight and regular",
             "Some lines appear wavy or distorted",
             "Some areas appear blurry or missing",
             "The central dot disappears or is hard to see"]
        )
       
        # Evaluate and provide interpretation
        if st.button("Submit Assessment"):
            st.subheader("Interpretation")
           
            # Function to interpret results
            def interpret_eye_results(eye_results, eye_name):
                if "All lines appear straight and regular" in eye_results and len(eye_results) == 1:
                    st.success(f"✅ Your {eye_name} eye results appear normal.")
                    return True
                else:
                    warnings = []
                    if "Some lines appear wavy or distorted" in eye_results:
                        warnings.append("wavy or distorted lines")
                    if "Some areas appear blurry or missing" in eye_results:
                        warnings.append("blurry or missing areas")
                    if "The central dot disappears or is hard to see" in eye_results:
                        warnings.append("issues seeing the central dot")
                   
                    if warnings:
                        warning_text = ", ".join(warnings)
                        st.warning(f"⚠️ Your {eye_name} eye results show {warning_text}. These may be signs of macular problems.")
                    else:
                        st.info(f"Your {eye_name} eye assessment is incomplete or inconclusive.")
                    return False
           
            # Interpret each eye
            left_normal = interpret_eye_results(left_eye, "LEFT")
            right_normal = interpret_eye_results(right_eye, "RIGHT")
           
            # Overall recommendation
            st.subheader("Recommendations")
           
            if left_normal and right_normal:
                st.info("""
                Your results suggest normal macular function. Continue with regular eye check-ups and:
               
                1. Protect your eyes from UV light with sunglasses
                2. Maintain a healthy diet rich in antioxidants (leafy greens, colorful fruits and vegetables)
                3. Don't smoke, as smoking increases risk of macular degeneration
                4. Consider regular Amsler grid self-tests, especially if you're over 50
                """)
            else:
                st.warning("""
                ⚠️ **Important**: Your results suggest possible issues with macular function.
               
                1. Schedule an appointment with an eye care professional as soon as possible
                2. Bring these test results to your appointment
                3. Don't delay seeking care, as early intervention is important for macular conditions
                4. Continue monitoring with the Amsler grid but get professional evaluation
               
                Abnormalities in this test could indicate conditions like age-related macular degeneration,
                macular edema, or other retinal problems.
                """)
           
            st.info("""
            **What is Macular Degeneration?**
           
            The macula is the central part of the retina responsible for detailed central vision.
            Age-related macular degeneration (AMD) is a leading cause of vision loss in people over 50.
           
            Early detection and treatment can help preserve vision in many cases. Regular eye exams are
            essential, especially if you:
            - Are over 50 years old
            - Have a family history of macular degeneration
            - Smoke cigarettes
            - Have high blood pressure or cardiovascular disease
            """)
   
    # Default case - no test selected
    else:
        st.info("Please select a vision test from the sidebar menu.")
       
        # Display available tests
        st.subheader("Available Vision Tests")
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("""
            **🎨 Color Vision Test**  
            Screen for color blindness and color vision deficiencies.
           
            **👓 Visual Acuity Test**  
            Measure clarity of vision at different distances.
           
            **◐ Contrast Sensitivity Test**  
            Assess ability to distinguish subtle differences in contrast.
           
            **🔆 Astigmatism Test**  
            Check for irregular curvature of the eye or lens.
            """)
       
        with col2:
            st.markdown("""
            **👀 Peripheral Vision Test**  
            Evaluate side vision and visual field awareness.
           
            **⚡ Reaction Time Test**  
            Measure speed of visual processing and response.
           
            **🔲 Amsler Grid Test**  
            Check for signs of macular degeneration or other central vision issues.
            """)
       
        st.info("""
        **How to Use These Tests**
       
        1. Select a test from the sidebar menu
        2. Follow the on-screen instructions
        3. Complete the self-assessment
        4. Review your results and recommendations
       
        Remember that these tests are for informational purposes only and not a substitute for
        professional eye examinations.
        """)

# Add a footer with disclaimer
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Vision Health Suite | Educational Tool | Not for Clinical Use</p>
    <p>Always consult with healthcare professionals for proper diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)
