import sys
import numpy as np

from PIL import Image
from numpy import asarray
import numpy as np
import os


def dark_bright(arr, expected_shape=(28, 28)):
    try:
         #Too-dark or too bright
        if arr.mean() < 5 or arr.mean() > 250:
            return True
        return False 
    except Exception:
        return True  

def white_line(arr, expected_shape=(28, 28)):
    try:
        #white line
        if 7140 in np.sum(arr, axis=1):
            return True
        return False 
    except Exception:
        return True        

def main():
    
    if len(sys.argv) != 7:
        print("Usage: python3 convert_data.py fashion_mnist h w c fashion_mnist.npz 0/1")
        sys.exit(1)

    image_dir = sys.argv[1]
    H = int(sys.argv[2])
    W = int(sys.argv[3])
    C = int(sys.argv[4])
    output_npz = sys.argv[5]
    choice = int(sys.argv[6])     # user passes 0 or 1

    if choice == 0:
        original=[]
        label = []
        for k in os.listdir(image_dir):
            p = os.path.join(image_dir, k)
            # print(p)
            # break
            with Image.open(p) as img:
                data = np.asarray(img)
                # print(data)
                # break
                original.append(data)
                label.append(int(k.split('-')[0]))
        original = np.array(original)
        label = np.array(label)

        np.savez(output_npz,
             img_data=original,
             img_lbl=label)

        data = np.load(output_npz)

        print("Displaying array before pre-processing:")
        print(len(data["img_data"]))

    elif choice == 1:
        s=0 
        label = []
        for k in os.listdir(image_dir):
            p = os.path.join(image_dir, k)
            with Image.open(p) as img:
                data = np.asarray(img)
                if not dark_bright(data) and not white_line(data):
                    if s==0:
                        #uncorrupted = np.expand_dims(data, axis=0)
                        uncorrupted = data[None, ...]
                        #add lables for prediction
                        label.append(int(k.split('-')[0]))
                        s=1
                    else:
                        #uncorrupted = np.append(uncorrupted, np.expand_dims(data, axis=0), axis=0)
                        uncorrupted = np.concatenate([uncorrupted, data[None, ...]], axis=0)
                        #add lables for prediction
                        label.append(int(k.split('-')[0]))
        label = np.array(label)

        np.savez(output_npz,
             img_data=uncorrupted,
             img_lbl=label)

        data = np.load(output_npz)

        print("Displaying len of the data: ")
        print(len(data["img_data"]))

    else:
        print("Invalid option! Use 0 or 1.")

if __name__ == "__main__":
    main()
