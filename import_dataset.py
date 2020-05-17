import os
import datasets.hdf5_loader as dataset

def main():
    dataset_path = "./datasets/mnist/"
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path)
    print(dataset_train)
    print(len(dataset_train))
    img, label = dataset_train.get_data(dataset_train.ids[0])
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    num_class = label.shape[0] 
    print(h,w,c)

    # --- create model ---
    #model = Model(config, debug_information=config.debug, is_train=is_train)

    #return config, model, dataset_train, dataset_test

if __name__ == "__main__":
    main()
