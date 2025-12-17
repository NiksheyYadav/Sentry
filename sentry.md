# Multimodal Deep learning framework for Real-Time  Mental Health Assessment in Students: Integrating facial expression and Posture Analysis

- ## What is this?
    This is a smart system that can tell if a student is feeling stressed, sad or anxious just by watching them through a web-cam.

- ## How does it work?
        It works by focusing on two things: - 
        1. Face
            -> Is the student smiling frowning, or looking tired?
            -> Are their eyes showing sadness or worry?
        2. Body Posture
            - Are they sitting up straight or slouching (lazy)?
            - Is their head down?
            - Are they fidgeting? (nervous, bitting teeth, etc)

- ## How the system works: -
    ### Part 1
    1. It capture the video from webcam.
        - We will use "OpenCV" library to capture video frame-by-frame like 30 photos per second.
    2. Extract face from each frame: - from each video frame, we need to find where the face is?, so we will just cut out the face portion.
        - MTCNN or Dlib 
            - models to detect face.
            - They draw a box around face and cut it out.
    3. Analyze facial expression: 
        - The cropped face goes into cnn model to filter and recognize the patterns in images. like sad eyes, downturned mouth, lack of expression, tired look.
        - ResNet50 or efficientNet (CNN model) to understand the face on image to recoginze depression/anxiety.
    ### Part 2
    4. Extract Body Pose: - 
        - from the same video frame detect body parts (like shoulders, elbows , hands, etc). Connect these points to create "stick figure or wireframe" skeleton.
        - MediaPipe or OpenPose - Pose estimation models, they identify 33 body points in realtime
    - It will detect: -
        - slouching (head lower than normal)
        - closed posture (arms crossed, shoulders in word)
        - minimal movement
        - Head - down position (looking down for long time)
    5. Analyze Posture Patterns: - 
        - The skeleton figure go through another neural network. This pattern looks at posture pattern overtime.
        - It detect "has student been slouching for 10 minutes straight?"
        ### Option A: 
        - Temporal CNN - looks at sequence of poses (like 30 frame = 1 sec of movement). Detects motion pattern "Student barely oved in 5 min".
        ### Option B:
        - LTSM (Long Short Term Memory) Remembers past poses and compares with current. Good for tracking changes "Student started session upright, now slouching".
    6. Combine face + Posture (512 no. + 512 no.)
        ### Option A
        - Simple Concatenation
        ### Option B
        - Attention Mechanism (better way) System learns which signal is more imp. Ex.: - for this student posture is 70% imp., face is 30%. It will automatically based on what works best.
    7. Make Final Prediction (Fusion Network model)
        - This layer will make the actual decision.
        - After fusion we have 1024 no,
        - Through a series of mathematical transformations (neural network layer), each 512 neurons in the layer-1 look at all 1024 no. . Each neuron calculates a weighted sum.
## Result 
- The project aims to automatically detect stress depression from video by analyzing facial expressions and psoture, providing early alerts to counselors, and supporting mental health monitoring in a fast and non-intrusive way.