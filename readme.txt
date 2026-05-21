About: 
    - We have two python files namely
        -convert_data.py
        -cnn.py

    - convert_data.py reads image files, corrects problems, stores data to a .npz file for faster processing.
    - cnn.py: train/test split, trains or evaluates a model on data.

1. Create a virtual environment
 - python -m venv myvenv

2. Install all the packages
 - pip install -r requirements.txt

3. Extract the zip file

4. Run code convert_data.py to convert compress images in npz file:
    - python3 convert_data.py fashion_mnist h w c fashion_mnist.npz 0/1
        - python3 convert_data.py fashion_mnist 28 28 1 fashion_mnist.npz 0 (This extracts the original images from fashion_mnist directory and creates npz zip file)
        - python3 convert_data.py fashion_mnist 28 28 1 fashion_mnist.npz 1 (This extracts the original images fashion_mnist directory, does pre-processing and creates npz zip file)

5. Run code cnn.py to train the images using created npz files, if 0 then it trains/test on the original images whereas 1 used during convert_data.py trains/test on the cleaned data
    Train on DNN:
        - python3 cnn.py fashion_mnist.npz train  
    Test on DNN:
        - python3 cnn.py fashion_mnist.npz test  

    Train on CNN:
        - python3 cnn.py fashion_mnist.npz CNN_train  
    Test on CNN:
        - python3 cnn.py fashion_mnist.npz CNN_test 
