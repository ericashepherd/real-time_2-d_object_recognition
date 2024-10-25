/*
    Erica Shepherd
    CS 5330
    Project 3: Real-time Object 2-D Recognition
*/

#include <iostream>
#include <stdio.h>
#include <stack>
#include <string>
#include "csv_util.cpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// thresholding algorithm that separates foreground from background
int threshold(cv::Mat &src, cv::Mat &dst, float l_threshold, float s_threshold){
    int i, j, c;
    cv::Vec3b *ptr;

    // creates destination image copy
    src.copyTo(dst);

    // foreground set to white, background set to black
    int value;
    int min, max;
    float lightness, saturation;

    // iterates through pixels
    for (i=0; i<src.rows; i++){
        ptr = dst.ptr<cv::Vec3b>(i);
        for (j=0; j<src.cols; j++){
            // calculates light value from HLS scale (formula from OpenCV)
            min = max = ptr[j][0];
            for (c=1; c<3; c++){
                if (ptr[j][c] > max){
                    max = ptr[j][c];
                }
                if (ptr[j][c] < min){
                    min = ptr[j][c];
                }
            }
            lightness = ((float)(min + max)) / (2*255);

            // calculates saturation value from HLS scale for low lightness
            if (lightness < 0.5){
                saturation = (float)(max - min) / (max + min);
            }
            else{
                saturation = (float)(max - min) / (2 - (max + min));
            }

            // sets white background to black and objects with
            // enough saturation and low lightness to foreground
            value = 0;
            if (lightness < l_threshold){
                value = 255;
            }
            if (saturation > s_threshold){
                value = 255;
            }

            // sets color of pixel
            for (c=0; c<3; c++){
                ptr[j][c] = value;
            }
        }
    }
    return 0;
}

// grassfire transformation algorithm to grow a given number of times
// if shrink = true, foreground shrinks given number of times
// if shrink = false, foreground grows given number of times
int grassfire_transform(cv::Mat &src, cv::Mat &dst, bool shrink, int times){
    int i, j, c;
    cv::Vec3b *ptr;
    float *ptr_d, *ptr_d2;
    int new_value;
    int value_check, color;

    // creates destination image copy
    src.copyTo(dst);

    // temporary mat to hold Manhattan distances
    cv::Mat distances;
    distances.create(src.size(), CV_32F);

    // adjusts values to shrink or grow based on parameter
    if (shrink){
        // value check (255 if foreground, 0 if background)
        value_check = 255;
        // color (0 for black, 255 for white)
        color = 0;
    }   
    else{
        value_check = 0;
        color = 255;
    }

    // first pass for grassfire transform
    for (i=0; i<src.rows; i++){
        ptr = dst.ptr<cv::Vec3b>(i);
        ptr_d = distances.ptr<float>(i);
        if (i > 0){
            ptr_d2 = distances.ptr<float>(i-1);
        }
        for (j=0; j<src.cols; j++){
            if (ptr[j][0] == value_check){
                if (i > 0){
                    // takes min value and adds 1
                    if (j > 0){
                        ptr_d[j] = std::min(ptr_d[j-1], ptr_d2[j]);
                        ptr_d[j]++;
                    }
                    // edge case for j=0
                    else{
                        ptr_d[j] = (src.rows * src.cols);
                    }
                }
                // edge case for i=0
                else{
                    ptr_d[j] = (src.rows * src.cols);
                }
            }
            else{
                ptr_d[j] = 0;
            }
        }
    }

    // second pass for grassfire transform
    for (i=src.rows-1; i>=0; i--){
        ptr = dst.ptr<cv::Vec3b>(i);
        ptr_d = distances.ptr<float>(i);
        if (i < (src.rows-1)){
            ptr_d2 = distances.ptr<float>(i+1);
        }
        for (j=src.cols-1; j>=0; j--){
            if (ptr[j][0] == value_check){
                if (i < (src.rows-1)){
                    // takes min value and adds 1
                    if (j < (src.cols-1)){
                        new_value = std::min(ptr_d[j+1], ptr_d2[j]);
                    }
                    // edge case for j=max
                    else{
                        new_value = ptr_d2[j];
                    }
                }
                // edge case for i=max
                else{
                    if (j < (src.cols-1)){
                        new_value = ptr_d[j+1];
                    }
                    // right corner case
                    else{
                        new_value = ptr_d[j];
                    }
                }
                new_value++;

                // updates value if less than current
                if (new_value < ptr_d[j]){
                    ptr_d[j] = new_value;
                }
            }
        }
    }

    // updating image with new shrinks/grows
    for (i=0; i<src.rows; i++){
        ptr = dst.ptr<cv::Vec3b>(i);
        ptr_d = distances.ptr<float>(i);
        for (j=0; j<src.cols; j++){
            if (ptr_d[j] <= times){
                for (c=0; c<3; c++){
                    ptr[j][c] = color;
                }
            }
        }
    }

    return 0;
}

// runs connected component analysis on binary image for regions and colors top n regions
// using region growing algorithm
int connected_components(cv::Mat &src, cv::Mat &dst, int n_regions){
    int i, j, k;
    cv::Vec3b *ptrD;
    float *ptrR;
    std::stack<std::pair<int, int>> region_stack;
    std::pair<int, int> seed;
    std::vector<std::vector<int>> regionID_list;
    int regionID = 1;
    int region_size;

    // copies src image to dst
    src.copyTo(dst);

    // temporary mat to hold region map
    cv::Mat region_map;
    region_map.create(src.size(), CV_32F);

    // initializes region map
    for (i=0; i<region_map.rows; i++){
        ptrR = region_map.ptr<float>(i);
        for (j=0; j<region_map.cols; j++){
            ptrR[j] = -1;
        }
    }

    // iterates through image pixels
    for (i=0; i<dst.rows; i++){
        ptrR = region_map.ptr<float>(i);
        ptrD = dst.ptr<cv::Vec3b>(i);
        for (j=0; j<dst.cols; j++){
            // checks for unmarked foreground region
            if (ptrD[j][0] == 255 && ptrR[j] < 0){
                region_size = 0;
                region_stack.push({i, j});
                // adds pixels to stack and explores via DFS
                while(!region_stack.empty()){
                    region_size++;
                    seed = region_stack.top();
                    region_map.at<float>(seed.first, seed.second) = regionID;
                    region_stack.pop();
                    // checks top
                    if (i > 0){
                        if (dst.at<cv::Vec3b>(seed.first - 1, seed.second)[0] == 255 &&
                            region_map.at<float>(seed.first - 1, seed.second) < 0){
                            
                            region_stack.push({seed.first - 1, seed.second});
                        }
                    }
                    // checks bottom
                    if (i < dst.rows-1){
                        if (dst.at<cv::Vec3b>(seed.first + 1, seed.second)[0] == 255 &&
                            region_map.at<float>(seed.first + 1, seed.second) < 0){
                            
                            region_stack.push({seed.first + 1, seed.second});
                        }
                    }
                    // checks left
                    if (j > 0){
                        if (dst.at<cv::Vec3b>(seed.first, seed.second - 1)[0] == 255 &&
                            region_map.at<float>(seed.first, seed.second - 1) < 0){
                            
                            region_stack.push({seed.first, seed.second - 1});
                        }
                    }
                    //checks right
                    if (j < dst.cols-1){
                        if (dst.at<cv::Vec3b>(seed.first, seed.second + 1)[0] == 255 &&
                            region_map.at<float>(seed.first, seed.second + 1) < 0){
                            
                            region_stack.push({seed.first, seed.second + 1});
                        }
                    }
                }
                // saves region size and IDs for sorting later
                regionID_list.insert(regionID_list.end(), {region_size, regionID});
                regionID++;
            }
        }
    }
    
    // sorts in ascending order of region size
    std::sort(regionID_list.begin(), regionID_list.end());
/*
    for (i=0; i<regionID_list.size(); i++){
        printf("regionID: %d, size: %d\n", regionID_list.at(i)[1],  regionID_list.at(i)[0]);
    }
*/ 
    // updates dst with region map values at index 0 for top N regions with the given rank
    for (i=0; i<dst.rows; i++){
        ptrD = dst.ptr<cv::Vec3b>(i);
        ptrR = region_map.ptr<float>(i);
        for (j=0; j<dst.cols; j++){
            for (k=regionID-2; k>(regionID-2)-n_regions; k--){
                int rank = 1;
                if (ptrR[j] == regionID_list.at(k)[1]){
                    ptrD[j][0] = rank;
                }
                rank++;
            }
        }
    }
/*
    printf("top:\n");
    for (i=regionID-2; i>(regionID-2)-n_regions; i--){
        printf("%d\n",i);
        printf("regionID: %d, size: %d\n", regionID_list.at(i)[1],  regionID_list.at(i)[0]);
    }
*/
    return 0;
}

// colors a region with given rgb values given the region map and ID
int color_region(cv::Mat &src, cv::Mat &dst, int regionID, int r, int g, int b){
    int i, j;
    cv::Vec3b *ptr;

    // copies src to dst
    src.copyTo(dst);

    // iterates over dst and updates image with colors
    for (i=0; i<src.rows; i++){
        ptr = dst.ptr<cv::Vec3b>(i);
        for (j=0; j<src.cols; j++){
            if (ptr[j][0] == regionID){
                ptr[j][0] = b;
                ptr[j][1] = g;
                ptr[j][2] = r;   
            }
        }
    }
    return 0;
}

// computes and returns features of a given region map and ID and draws them on given image;
// feature vectors returned: Hu moments 1 & 2, Oriented Bounding Box width/height ratio & 
// percentage filled
std::vector<double> features(cv::Mat &region_map, int regionID, cv::Mat &image){
    int i, j, x, y;
    int x_prime, y_prime;
    int x_prime_min, x_prime_max, y_prime_min, y_prime_max;
    int M_00, M_10, M_01;
    double centroid_x, centroid_y;
    double u_11, u_20, u_02;
    double alpha;
    double major_axis_x, major_axis_y;
    double minor_axis_x, minor_axis_y;
    cv::Vec3b *ptrS, *ptrT;
    std::vector<double> feature_vectors;
     int filled_count = 0;

    // temp Mat to hold Cartesian values for region map
    cv::Mat temp;
    temp.create(region_map.size(), region_map.type());

    // transform i, j values to Cartesian x, y
    for (i=0; i<region_map.rows; i++){
        ptrT = temp.ptr<cv::Vec3b>(i);
        // transform i value to Cartesian y
        ptrS = region_map.ptr<cv::Vec3b>((region_map.rows-1)-i);
        for (j=0; j<region_map.cols; j++){
            if (ptrS[j][0] == regionID){
                ptrT[j][0] = ptrS[j][0];
                filled_count++;
            }
            else{
                ptrT[j][0] = 0;
            }
        }
    }

    // calculate centroid
    M_00 = M_01 = M_10 = 0;
    for (y=0; y<temp.rows; y++){
        ptrT = temp.ptr<cv::Vec3b>(y);
        for (x=0; x<temp.cols; x++){
            if (ptrT[x][0] == regionID){
                M_10 += x;
                M_01 += y;
                M_00++;
            }
        }
    }
    centroid_x = (double)M_10 / M_00;
    centroid_y = (double)M_01 / M_00;
       
    // calculate variances
    u_11 = u_02 = u_20 = 0;
    for (y=0; y<temp.rows; y++){
        ptrT = temp.ptr<cv::Vec3b>(y);
        for (x=0; x<temp.cols; x++){
            if (ptrT[x][0] == regionID){
                u_20 += (double)(x - centroid_x) * (x - centroid_x);
                u_02 += (double)(y - centroid_y) * (y - centroid_y);
                u_11 += (double)(x - centroid_x) * (y - centroid_y);
            }
        }
    }
    u_20 = (double)u_20/M_00;
    u_02 = (double)u_02/M_00;
    u_11 = (double)u_11/M_00;
   
    // calculate alpha (central axis angle)
    alpha = 0.5 * std::atan2(2*u_11, u_20 - u_02);

    // calculate major axis
    major_axis_x = (double)std::cos(alpha);
    major_axis_y = (double)std::sin(alpha);
    // calculate minor axis
    minor_axis_x = (double)std::sin(-alpha);
    minor_axis_y = (double)std::cos(alpha);

    // calculate x/y prime min/max
    x_prime_min = y_prime_min = temp.rows * temp.cols + 1;
    x_prime_max = y_prime_max = x_prime_min * -1;
    for (y=0; y<temp.rows; y++){
        ptrT = temp.ptr<cv::Vec3b>(y);
        for (x=0; x<temp.cols; x++){
            if (ptrT[x][0] == regionID){
                x_prime = ((x-centroid_x)*major_axis_x) + ((y-centroid_y)*major_axis_y);
                y_prime = ((x-centroid_x)*minor_axis_x) + ((y-centroid_y)*minor_axis_y);

                // saves min/max
                if (x_prime > x_prime_max){
                    x_prime_max = x_prime;
                }
                if (x_prime < x_prime_min){
                    x_prime_min = x_prime;
                }
                if (y_prime > y_prime_max){
                    y_prime_max = y_prime;
                }
                if (y_prime < y_prime_min){
                    y_prime_min = y_prime;
                }
            }
        }
    }

    // used for drawing features onto image
    cv::Point pt1, pt2, pt3, pt4, pt5;

    // draws major axis in bright red after unrotating coordinates
    int axis_length = 200;
    pt1 = {(int)centroid_x, (image.rows-1)-(int)centroid_y};
    pt2 = {(int)centroid_x + (int)(axis_length*major_axis_x), (image.rows-1)-((int)centroid_y + (int)(axis_length*major_axis_y))};
    cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    // calculates unrotated oriented bounding box coordinates
    x_prime = x_prime_max;
    y_prime = y_prime_max;
    x = ((x_prime*major_axis_x) - (y_prime*major_axis_y)) + centroid_x;
    y = ((x_prime*major_axis_y) + (y_prime*minor_axis_y)) + centroid_y;
    pt2 = {x, (image.rows-1)-y}; // (x_max, y_max)
    y_prime = y_prime_min;
    x = ((x_prime*major_axis_x) - (y_prime*major_axis_y)) + centroid_x;
    y = ((x_prime*major_axis_y) + (y_prime*minor_axis_y)) + centroid_y;
    pt3 = {x, (image.rows-1)-y}; // (x_max, y_min)

    x_prime = x_prime_min;
    y_prime = y_prime_max;
    x = ((x_prime*major_axis_x) - (y_prime*major_axis_y)) + centroid_x;
    y = ((x_prime*major_axis_y) + (y_prime*minor_axis_y)) + centroid_y;
    pt4 = {x, (image.rows-1)-y}; // (x_min, y_max)
    y_prime = y_prime_min;
    x = ((x_prime*major_axis_x) - (y_prime*major_axis_y)) + centroid_x;
    y = ((x_prime*major_axis_y) + (y_prime*minor_axis_y)) + centroid_y;
    pt5 = {x, (image.rows-1)-y}; // (x_min, y_min)

    // draws the oriented bounding box in image
    cv::line(image, pt2, pt3, cv::Scalar(0, 0, 255), 1);
    cv::line(image, pt2, pt4, cv::Scalar(0, 0, 255), 1);
    cv::line(image, pt3, pt5, cv::Scalar(0, 0, 255), 1);
    cv::line(image, pt4, pt5, cv::Scalar(0, 0, 255), 1);


    // calculates translation/scale/rotation invariant image moments
    double n_20, n_02, n_11, n_30, n_03, n_12, n_21;
    double feature_array[4];
    double denom = ((double)M_00*M_00);
    n_20 = (double)u_20 / denom;
    n_02 = (double)u_02 / denom;
    n_11 = (double)u_11 / denom;

    // adds first two HU moment calcs
    feature_vectors.insert(feature_vectors.begin(), n_20 + n_02);
    feature_vectors.insert(feature_vectors.begin(), (double)((n_20-n_02) * (n_20-n_02)) + (double)4*(n_11 * n_11));

    // calculates bounding box width/height ratio and adds to feature vectors
    double width = std::sqrt( ((pt4.x-pt2.x)*(pt4.x-pt2.x)) + ((((image.rows-1)-pt4.y)-((image.rows-1)-pt2.y))*(((image.rows-1)-pt4.y)-((image.rows-1)-pt2.y))) ); // pt2 & pt4
    double height = std::sqrt( ((pt2.x-pt3.x)*(pt2.x-pt3.x)) + ((((image.rows-1)-pt2.y)-((image.rows-1)-pt3.y))*(((image.rows-1)-pt2.y)-((image.rows-1)-pt3.y))) ); // pt2 & pt3
    feature_vectors.push_back((double) width/height);

    // calculates bounding box percentage filled and adds to feature vectors
    feature_vectors.push_back(filled_count/(double)(width*height));

    return feature_vectors;
}

// classifies a new vector using the object database using scaled Euclidean distance metric
char* euclidean_classify(std::vector<double> new_vector){
    int i, j;
    std::vector<char *> object_names;
    std::vector<std::vector<double>> database;
    
    // reads in data from database
    read_image_data_csv("../database.txt", object_names, database, false);

    // arrays to hold data calculations
    double database_means[database[0].size()];
    double database_var[database[0].size()];
    double sums[database.size()];
    std::vector<std::vector<double>> sum_vectors;

    // initializes arrays
    for (i=0; i<database[0].size(); i++){
        database_means[i] = 0;
        database_var[i] = 0;
    }
    for (i=0; i<database.size(); i++){
        sums[i] = 0;
    }
    
    // calculates means of database
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            database_means[j] += database[i][j];
        }
    }
    for (i=0; i<database[0].size(); i++){
        database_means[i] = (double)database_means[i] / database.size();
    }

    // calculates variances of database
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            database_var[j] += (database[i][j] - database_means[j])*(database[i][j] - database_means[j]);
        }
    }
    for (i=0; i<database[0].size(); i++){
        database_var[i] = (double)database_var[i] / database.size();
    }

    // calculates scaled Euclidean distance
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            sums[i] += (double)((new_vector[j]-database[i][j])*(new_vector[j]-database[i][j])) / database_var[j];
        }
        // adds sums to vector list
        sum_vectors.push_back({sums[i], (double)i});
    }
    
    // sorts in ascending order of sums
    std::sort(sum_vectors.begin(), sum_vectors.end());

    // returns top result
    return object_names.at(sum_vectors.front()[1]);
    
}

// classifies a new vector using the object database using scaled Euclidean distance metric
// with K-nearest neighbor matching where K = 2
char* KNN_classify(std::vector<double> new_vector){
    int i, j;
    std::vector<char *> object_names;
    std::vector<std::vector<double>> database;
    int k = 2;
    
    // reads in data from database
    read_image_data_csv("../database_KNN.txt", object_names, database, false);

    // arrays to hold data calculations
    double database_means[database[0].size()];
    double database_var[database[0].size()];
    double sums[database.size()/k];
    std::vector<std::vector<double>> sum_vectors;

    // initializes arrays
    for (i=0; i<database[0].size(); i++){
        database_means[i] = 0;
        database_var[i] = 0;
    }
    for (i=0; i<database.size()/k; i++){
        sums[i] = 0;
    }
    
    // calculates means of database
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            database_means[j] += database[i][j];
        }
    }
    for (i=0; i<database[0].size(); i++){
        database_means[i] = (double)database_means[i] / database.size();
    }

    // calculates variances of database
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            database_var[j] += (database[i][j] - database_means[j])*(database[i][j] - database_means[j]);
        }
    }
    for (i=0; i<database[0].size(); i++){
        database_var[i] = (double)database_var[i] / database.size();
    }

    // calculates scaled Euclidean distance
    for (i=0; i<database.size(); i++){
        for (j=0; j<database[i].size(); j++){
            sums[i/k] += (double)((new_vector[j]-database[i][j])*(new_vector[j]-database[i][j])) / database_var[j];
        }
        // adds sums to vector list
        if (i%k == k-1){
            sum_vectors.push_back({sums[i/k], (double)std::floor(i/k)});
        }
    }
    
    // sorts in ascending order of sums
    std::sort(sum_vectors.begin(), sum_vectors.end());

    // returns top result
    return object_names.at(sum_vectors.front()[1]*k);
}

// takes in a result, expected result, and adds them to the confusion matrix
int evaluate(char* result, char* expected_result, int matrix[10][10]){
    int i, j;
    int result_index, exp_index;
    std::vector<char *> object_names;
    std::vector<std::vector<double>> database;
    
    // reads in data from database
    read_image_data_csv("../database.txt", object_names, database, false);

    // finds index of results
    for (i=0; i<object_names.size(); i++){
        if (std::strcmp(object_names.at(i), result) == 0){
            result_index = i;
        }
        if (std::strcmp(object_names.at(i), expected_result) == 0){
            exp_index = i;
        }
    }

    // adds them to matrix
    matrix[result_index][exp_index]++;

    return 0;
}

// prints contents of given matrix
void print_matrix(int matrix[10][10]){
    int i, j;

    for (i=0; i<10; i++){
        for (j=0; j<10; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int, char**) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(1);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window

    // initializes temporary and display frames
    cv::Mat frame, displayFrame;
    cv::Mat temp, temp2;

    // saves feature vectors
    std::vector<double> feature_vectors;

    // variable to execute command
    char command = 'o';

    // looping variable to enable pause/resume
    bool pause = false;

    // holds user input
    char object_name[256];

    // displays object name
    char* display_name;

    // holds confusion matrix results
    int confusion_matrix1[10][10];
    int confusion_matrix2[10][10];
    int i, j;

    // initializes matrices
    for (i=0; i<10; i++){
        for (j=0; j<10; j++){
            confusion_matrix1[i][j] = 0;
            confusion_matrix2[i][j] = 0;
        }
    }

    while(1){
        // see if there is a waiting keystroke
        char key = cv::waitKey(10); 

        // exits video loop
        if (key == 'q'){
            break;
        }
        // pauses video
        else if (key == 'p'){
            pause = !pause;
        }
        // saves an image of current screen to given path
        else if (key == 's'){
            std::cout << "Saving Image\nPlease enter image name: \n";
            std::cin >> object_name;
            bool check = cv::imwrite(object_name, displayFrame);
            if (!check){
                    printf("Image not saved");
            }
            else{
                std::cout << "Image saved successfully\n";
            }
        }
        // adds current frame to database
        else if (key == 'n'){
            std::cout << "Adding to database\n";
            std::cout << "Please input the name/label for this object: \n";
            std::cin >> object_name;
            threshold(frame, temp, 0.2, 0.1);
            grassfire_transform(temp, temp2, false, 7);
            grassfire_transform(temp2, temp, true, 5);
            connected_components(temp, temp2, 1);
            frame.copyTo(displayFrame);
            feature_vectors = features(temp2, 1, displayFrame);
            append_image_data_csv("../database.txt", object_name, feature_vectors, 0);
            std::cout << "Input successful.\n";
        }
        // adds current frame to KNN database
        else if (key == 'm'){
            std::cout << "Adding to KNN database\n";
            std::cout << "Please input the name/label for this object: \n";
            std::cin >> object_name;
            threshold(frame, temp, 0.2, 0.1);
            grassfire_transform(temp, temp2, false, 7);
            grassfire_transform(temp2, temp, true, 5);
            connected_components(temp, temp2, 1);
            frame.copyTo(displayFrame);
            feature_vectors = features(temp2, 1, displayFrame);
            append_image_data_csv("../database_KNN.txt", object_name, feature_vectors, 0);
            std::cout << "Input successful.\n";
        }
        // evaluates euclidean classify
        else if (key == 'e'){
            threshold(frame, temp, 0.2, 0.1);
            grassfire_transform(temp, temp2, false, 7);
            grassfire_transform(temp2, temp, true, 5);
            connected_components(temp, temp2, 1);
            frame.copyTo(displayFrame);
            feature_vectors = features(temp2, 1, displayFrame);
            display_name = euclidean_classify(feature_vectors);
            std::cout << "Please input the correct name/label for this object: \n";
            std::cin >> object_name;
            evaluate(display_name, object_name, confusion_matrix1);
            std::cout << "Input successful.\n";
        }
        // evaluates KNN classify
        else if (key == 'k'){
            threshold(frame, temp, 0.2, 0.1);
            grassfire_transform(temp, temp2, false, 7);
            grassfire_transform(temp2, temp, true, 5);
            connected_components(temp, temp2, 1);
            frame.copyTo(displayFrame);
            feature_vectors = features(temp2, 1, displayFrame);
            display_name = KNN_classify(feature_vectors);
            std::cout << "Please input the correct name/label for this object: \n";
            std::cin >> object_name;
            evaluate(display_name, object_name, confusion_matrix2);
            std::cout << "Input successful.\n";
        }
        // prints matrix results
        else if (key == 'w'){
            std::cout << "Euclidean Classify: \n";
            print_matrix(confusion_matrix1);
            std::cout << "\nKNN Classify: \n";
            print_matrix(confusion_matrix2);
        }
        // assigns command key
        else if (key != -1) {
            command = key;
        }

        // captures video if not paused
        if (!pause) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream

            // print error if no frame captured from camera
            if( frame.empty() ) {
                    printf("frame is empty\n");
                    break;
            }    
            
            // checks for commands
            switch(command){
                    // shows thresholded objects
                    case 't':
                        threshold(frame, displayFrame, 0.2, 0.1);
                        break;
                    // shows thresholded & transformed objects
                    case 'g':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, displayFrame, true, 5);
                        break;
                    // shows colored largest region
                    case 'r':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        color_region(temp2, displayFrame, 1, 0, 0, 255);
                        break;
                    // shows largest colored region with bounding box and axis
                    case 'f':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        color_region(temp2, displayFrame, 1, 0, 0, 255);
                        feature_vectors = features(temp2, 1, displayFrame);
                        break;
                    // shows original image with bounding box and axis
                    case 'b':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        frame.copyTo(displayFrame);
                        feature_vectors = features(temp2, 1, displayFrame);
                        break;
                    // shows original image with bounding box and axis
                    // also displays second feature vector value on screen
                    case 'd':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        frame.copyTo(displayFrame);
                        feature_vectors = features(temp2, 1, displayFrame);
                        cv::putText(displayFrame, std::to_string((float)feature_vectors.at(1)), {20, 50}, cv::FONT_ITALIC, 1, cv::Scalar(0, 0, 0), 2);
                        break;
                    // classifies object on screen with euclidean and outputs object name
                    case 'c':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        frame.copyTo(displayFrame);
                        feature_vectors = features(temp2, 1, displayFrame);
                        display_name = euclidean_classify(feature_vectors);
                        cv::putText(displayFrame, display_name, {20, 50}, cv::FONT_ITALIC, 1, cv::Scalar(0, 0, 0), 2);
                        break;
                    // classifies object on screen with KNN and outputs object name
                    case 'x':
                        threshold(frame, temp, 0.2, 0.1);
                        grassfire_transform(temp, temp2, false, 7);
                        grassfire_transform(temp2, temp, true, 5);
                        connected_components(temp, temp2, 1);
                        frame.copyTo(displayFrame);
                        feature_vectors = features(temp2, 1, displayFrame);
                        display_name = KNN_classify(feature_vectors);
                        cv::putText(displayFrame, display_name, {20, 50}, cv::FONT_ITALIC, 1, cv::Scalar(0, 0, 0), 2);
                        break;
                    default:
                        displayFrame = frame;   
                        break;
            }

            // shows new frame
            cv::imshow("Video", displayFrame);
        }
    }
    // exits program
    delete capdev;

    return(0);
}