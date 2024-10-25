Erica Shepherd
CS5330: Pattern Recognition & Computer Vision
Project 3: Real-time Object 2-D Recognition
Spring 2022

Operating System: Windows 11
IDE: Visual Studio Code

Instructions for running executables:
    - executables were generated using a CMake file with the following instructions:
        =============================================================================
            cmake_minimum_required(VERSION 3.0.0)
            project(proj3 VERSION 0.1.0)

            # links OpenCV & include directories
            find_package(OpenCV REQUIRED)
            include_directories(${OpenCV_INCLUDE-DIRECTORIES})
            set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

            link_libraries(${OpenCV_LIBS})

            include_directories(${include})

            # generates executables
            add_executable(main src/main.cpp)

            set(CPACK_PROJECT_NAME ${PROJECT_NAME})
            set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
            include(CPack)
        =============================================================================
    - main.cpp also uses csv_util that Prof. Maxwell created with a slight modification
        to allow vectors of type double instead of float, a copy was provided
    - the following commands were given in the terminal within the build folder to
        run the program:
        $ mingw32-make.exe 
        $ ./../bin/main

    - input the following commands at the window to view tasks:
        - 't' for thresholded objects (task 1)
        - 'g' for thresholded & transformed objects (task 2)
        - 'r' for colored largest region (task 3)
        - 'd' for original image with oriented bounding box, axis, and second feature
            vector (HU moment 1) on screen (task 4)
                - 'f' for colored region with oriented bounding box & axis
                - 'b' for original image with oriented bounding box & axis only
        - 'n' to add current object on screen to database with user-input label for
            training (task 5)
        - 'm' to add current object on screen to KNN database with user-input label
            for training (task 5/7) // must input k images of the same object consecutively
        - 'c' to classify current object on screen using object database and
            scaled Euclidean distance metric and output label on screen (task 6)
        - 'x' to classify current object on screen using KNN-nearest Neighbor matching
            where k=2 and output label on screen (task 7)
        - 'e' to evaluate performance of first classifier by classifing current object and
            asking user for correct result to save in confusion matrix (task 8)
        - 'k' to evaluate performance of second classifier (KNN) by classifing current 
            object and asking user for correct result to save in confusion matrix (task 8)
        - 'w' prints the confusion matrix (task 8)

    - additional commands:
        - 'p' to pause the video
        - 'o' or any other unbound key to return to original video
        - 's' to save current screen and input image name to commandline
        - 'q' to exit
